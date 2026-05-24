# CUDA / GPU Performance Audit & Implementation — StreamDiffusion

## Context

You asked me to study the two CUDA references in `docs/CUDA_Handbook/` (*The CUDA
Handbook*, Wilt 2013) and `docs/CUDA_Programming_Parallel_processors/` (*Programming
Massively Parallel Processors*, Hwu 5th ed.) **focusing on performance**, then analyze
the StreamDiffusion implementation with MCP search and **identify + implement** potential
performance improvements.

**Decisions captured:**
- **Scope** = Python/PyTorch GPU efficiency (transfers, pinned memory, async overlap,
  redundant allocations, kernel fusion via existing torch ops, runtime flags). **No custom
  CUDA kernels.**
- **Deliverable** = written report **and** implement everything found.
- **Branch base** = cut the new branch from **`feat/cuda-ipc-output`** (it carries the
  `_compat/cuda_ipc` module; the current branch `security/dep-audit-2026-05` does not).

**IPC reality (corrected):** the CUDA-IPC link **is** genuinely zero-copy — the importer
builds *"per-slot zero-copy torch.Tensor views of GPU memory"* via
`__cuda_array_interface__` (`src/streamdiffusion/_compat/cuda_ipc/importer.py:270-303`),
and `CUDAIPCImporter.get_frame()` returns that **GPU tensor** (`cuda_ipc_importer.py:128`,
docstring "Returns tensor"). The integration defeats it: the consumer calls
`get_frame_numpy()` (`Scripts/streamdiffusionTD__Text__td_manager__td.py:682`), forcing a
D2H copy to CPU numpy every frame, after which the input path re-uploads it. **The fix is
in-reach and small** — consume `get_frame()` and keep the tensor on the GPU. The module
lives only on `feat/cuda-ipc-output`; `_compat/td_exporter/` is synced into the
distributed `.tox`, so TD-side changes ship in the binary. Note the consumer
(`td_manager`) lives in the cwd `Scripts/` folder, **outside** the nested git repo.

---

## Book grounding (citations used in the report)

- **CUDA Handbook Ch. 5 §5.1 "Host Memory" (pp. 121–128), Table 5.1 (p. 123):** pinned vs
  pageable host→device = **5523 vs 2951 MB/s (~1.9×)**; pageable copies double-buffer
  through pinned staging, adding CPU overhead.
- **CUDA Handbook Ch. 6 "Streams and Events" §6.1–6.2 (pp. 173–196):** only pinned memory
  is eligible for `cudaMemcpyAsync`; moving a synchronize into a per-launch loop raises
  cost **~3.4 µs → ~100 µs** — per-frame device sync destroys CPU/GPU pipelining.
- **CUDA Handbook Ch. 11 "Streaming Workloads" (pp. 353–362, quote p. 353):** *"If the
  input and output are in device memory, it doesn't make sense to transfer the data back
  to the CPU just to perform one operation."* Bandwidth-bound work is dominated by PCIe;
  minimize/overlap transfers and transfer fewer bytes (smaller dtype).
- **PMPP Ch. 5 (memory-bandwidth/roofline)**, **Ch. 4 §4.7 (occupancy)**, **Ch. 6 §6.1
  (global-memory coalescing)**, **Ch. 18 §18.7 (kernel-launch overhead)** — secondary,
  since this codebase has no hand-written kernels.

---

## Findings (full inventory in the report; ranked by ROI here)

The TensorRT engine layer and the `pipeline.py` per-frame loop are **already heavily
hand-optimized** (buffer reuse, CUDA-graph capture, pinned D2H output, KV-cache prealloc,
in-place CFG batch construction — many "eliminates N mallocs/kernel launches" comments).
The remaining wins concentrate at the **data-transfer boundary** and a few runtime flags.

### Tier 1 — Safe, in-repo, high-confidence → IMPLEMENT

**P1. Set global matmul / cuDNN performance flags once at init.**
Currently none of `torch.backends.cuda.matmul.allow_tf32`, `torch.backends.cudnn.allow_tf32`,
`torch.backends.cudnn.benchmark`, `torch.set_float32_matmul_precision` are set anywhere.
Benefits the PyTorch VAE/fallback paths and all preprocessing convs/`grid_sample` (UNet
itself is TRT, so unaffected — safe). *Grounding: PMPP Ch. 5 roofline; CUDA HB Ch. 6.*
- **File:** `src/streamdiffusion/pipeline.py` `StreamDiffusion.__init__` (~L57) or the
  wrapper init. Set TF32 on, `cudnn.benchmark=True`, `set_float32_matmul_precision("high")`.
- Risk: **low**.

**P2. Stop synchronizing the GPU every frame for the timing EMA.**
`pipeline.py:1127-1130` does `end.record(); end.synchronize()` **every frame** purely to
update `inference_time_ema` (consumed only by the similar-filter sleep at L1088). Per CUDA
HB §6.1 this per-frame host stall breaks pipelining.
- **Fix:** measure on a cadence (e.g. every 16th frame) and keep the EMA fed from those
  samples — preserves the sleep heuristic while removing ~15/16 of per-frame host stalls.
  `start.record()` stays cheap; only the blocking `end.synchronize()` is gated.
- **File:** `src/streamdiffusion/pipeline.py:1060-1130`.
- Risk: **low–med** (verify `inference_time_ema` has no other hard consumer; grep first).

**P3. GPU-native Canny — remove the per-frame GPU→CPU→GPU round-trip.**
`canny.py:72-93` (`_process_tensor_core`) does `gray_tensor.cpu()` → `.numpy()` →
`cv2.Canny` → `torch.from_numpy().to(device)` **every frame**, despite being on the
"tensor" path. `canny.py:8` already carries `#TODO provide gpu native edge detection`.
Sibling `soft_edge.py` already does GPU Sobel via `nn.Conv2d`. *Grounding: CUDA HB Ch. 11
p. 353.*
- **Fix:** implement Sobel-gradient magnitude + hysteresis-style thresholding on GPU
  (reuse the soft_edge Sobel pattern, or `kornia.filters.canny` if kornia is available),
  keyed off the existing `low_threshold`/`high_threshold` params. Keep the `cv2` CPU path
  (`_process_core`) as the PIL fallback.
- Risk: **med** — GPU edges won't be bit-identical to OpenCV; validate visually and keep
  threshold semantics.

### Tier 2 — In-repo transfer reduction → IMPLEMENT

**P4. Convert output on the GPU, transfer uint8 (4× fewer PCIe bytes) via pinned memory.**
`td_manager._send_output_frame:750-761` does `output_image.cpu().numpy()` → CPU CHW→HWC
transpose → CPU `*255 .astype(uint8)`. The float32 tensor is moved over PCIe, then scaled
on the CPU. *Grounding: CUDA HB Ch. 5 (pinned) + Ch. 11 (fewer bytes).*
- **Fix:** do `*255` + `uint8` cast + CHW→HWC **on the GPU**, then a single (pinned, if a
  reusable buffer fits) D2H copy of the uint8 frame. First confirm which output path is
  live — the wrapper already has a pinned uint8-NHWC path at `wrapper.py:928-933`; if that
  path serves TD, the win is to route TD output through it rather than the CPU branch here.
- **Files:** `Scripts/streamdiffusionTD__Text__td_manager__td.py:742-774`,
  `src/streamdiffusion/wrapper.py` output region.
- Risk: **med** (TD output formatting must stay byte-compatible).

**P5. Upload input as uint8 and normalize on the GPU; feed the pipeline's GPU fast path.**
`td_manager` run loop L549-550 does `astype(np.float32)/255.0` **on the CPU** (4× larger
upload) before handing to `wrapper.img2img`. The pipeline already has a GPU-tensor fast
path (`pipeline.py:1064-1079`) that **skips all preprocessing** when handed a normalized
CUDA tensor of the right shape/dtype.
- **Fix:** upload the uint8 frame to the GPU, do `/255` + layout/normalize on-device, and
  pass a GPU tensor so the fast path engages. Applies to the CPU-SHM fallback path (the
  IPC path is P6).
- **Files:** `Scripts/streamdiffusionTD__Text__td_manager__td.py:543-569`, input handoff
  into `wrapper.img2img`.
- Risk: **med**.

### Tier 3 — IPC zero-copy integration (on feat/cuda-ipc-output) → IMPLEMENT

**P6. Restore IPC input zero-copy — the headline fix.** The link is already zero-copy
(`importer.py:270-303` builds zero-copy torch views; `get_frame()` returns a GPU tensor),
but `td_manager.py:682` calls `get_frame_numpy()` → D2H copy every frame, then P5
re-uploads. *Grounding: CUDA HB Ch. 11 p. 353 — the canonical "don't round-trip" case.*
- **Fix:** in the IPC branch of `_get_input_frame` (`td_manager.py:681-685`), call
  `get_frame()` to obtain the GPU tensor; strip alpha + permute to CHW + `/255` +
  dtype/range-normalize **on the GPU**; pass the resulting CUDA tensor straight into the
  pipeline's GPU fast path (`pipeline.py:1064-1079`), bypassing the numpy/CPU input path
  entirely. P5 (CPU-SHM input) remains the fallback when IPC is inactive.
- **Files:** `Scripts/streamdiffusionTD__Text__td_manager__td.py:543-569, 681-685`; verify
  `wrapper.img2img` forwards a CUDA tensor through to `pipeline.__call__` without a CPU cast.
- Risk: **med** (TD boundary; gated behind IPC-active, with CPU-SHM fallback intact).

**P7. Confirm/enable IPC output export (likely already present).** `wrapper.py` on this
branch has `_lazy_init_ipc_exporter` + `use_cuda_ipc_output`; when enabled the wrapper
exports the frame via IPC and `_send_output_frame` receives `None` (`td_manager.py:744-745`).
- **Action:** verify the IPC output path is wired and exercised end-to-end; only if a gap
  exists, route the GPU-side uint8 conversion (P4) into the exporter rather than CPU SHM.
  No reimplementation expected.
- Risk: **low** (mostly verification).

### Not pursued (documented only)
- Hot-loop micro-allocations (`stock_noise` cat `pipeline.py:989`, `x_t_latent_buffer`
  `:996`, CFG combine `:907`, `.float()` upcast in `scheduler_step_batch` `:697-706`): the
  loop is already aggressively pre-allocated; further micro-opt is high-regression-risk,
  low-ROI.
- Custom CUDA kernels: excluded by the chosen scope.

---

## Implementation order

1. **Switch to `feat/cuda-ipc-output`** and implement there. First `git status` — surface
   any uncommitted Task-1 (math-audit) changes before checkout so nothing is lost.
2. **P1 + P2** — `pipeline.py` (flags + sync cadence). Grep `inference_time_ema` consumers
   first.
3. **P3** — GPU-native Canny in `canny.py` (cv2 fallback retained).
4. **P4** — GPU-side output conversion + uint8 D2H in `td_manager`/`wrapper.py` (confirm
   live output path first).
5. **P6** — IPC input zero-copy: `get_frame_numpy()` → `get_frame()` + on-GPU normalize →
   pipeline fast path. **P5** (CPU-SHM uint8 upload) as the non-IPC fallback. **P7** —
   verify IPC output export is wired.
6. **Write the report** to `StreamDiffusion/audit_reports/2026-05-23-cuda-perf-audit.md`
   and copy this plan to `StreamDiffusion/docs/plans/` (per `feedback_save_plans_as_project_files`).

## Verification

- Smoke per memory model targets: **SD-Turbo / SDXL-Turbo, 512×512, 2-step,
  t_index=[32,45], seed=2**.
- Use the existing `src/streamdiffusion/tools/gpu_profiler.py` to measure per-frame ms and
  the removed sync stall, before vs after.
- **P3:** visually compare `cv2.Canny` vs GPU edges on a test frame at the default
  thresholds; confirm ControlNet still tracks.
- **P4/P5:** confirm the TD round-trip output is byte-compatible (resolution/dtype/range)
  and FPS improves; watch the TD textport / `logs/` for regressions.
- Re-run `code-search` for `.cpu()`/`.numpy()` on the per-frame path to confirm the
  round-trips are gone.

---

## Post-implementation corrections (2026-05-24)

**File path corrections.** This plan cited
`Scripts/streamdiffusionTD__Text__td_manager__td.py` as a general reference, but the
correct distinction is:
- `Scripts/streamdiffusionTD__Text__td_manager__td.py` — **canonical TD Text-DAT source**
  (edits here sync to the running .tox immediately; this is where to make changes).
- `StreamDiffusion/StreamDiffusionTD/td_manager.py` — **runtime target** written by TD on
  "Writeconfigs"; **untracked by git**. P4/P5/P6 live in both copies and are not in git
  history — the f631c90 commit message only covers the pipeline.py / canny.py parts.

**P3 GPU Canny hardening (2026-05-24).**
Two correctness bugs fixed in `src/streamdiffusion/preprocessing/processors/canny.py`:
1. `mag / (mag.amax() + 1e-7)` → `(mag / 4.0).clamp(0.0, 1.0)` — fixed per-frame max
   normalization that made thresholds relative/flickering. New constant divisor (≈ max
   Sobel response for a [0,1] step edge) keeps thresholds stable across frames.
2. `edges.unsqueeze(0).expand(3,-1,-1)` → `.repeat(3,1,1)` — fixed non-contiguous
   stride-0 view returned to downstream consumers.

**`use_cuda_ipc_controlnet` was a dead config flag.** At the time this plan was written
(and through the P6 implementation), the config key `use_cuda_ipc_controlnet: true` and
`cuda_ipc_control_shm_name` were emitted by the YAML emitter but had **no producer and no
consumer**: `shmem_out_cn` has only a CPU `SharedMemEXT` (no `cuda_ipc_parexec` exporter),
and neither td_manager copy had a control importer. Meanwhile, `_process_controlnet_frame`
bailed permanently when `control_memory` was `None`, with no lazy-reconnect — the real root
cause of the "no conditioning effect" regression.

**ControlNet CUDA-IPC consumer implemented (2026-05-24).** The Python-side consumer was
added to both td_manager copies:
- `ipc_control_importer` / `_pending_ipc_control_name` state vars + throttled lazy-reconnect.
- `_try_construct_ipc_control_importer()` helper (mirrors `_try_construct_ipc_importer`).
- `_send_back_processed_controlnet()` helper (extracted from inline duplication).
- `_process_controlnet_frame` restructured: IPC path (zero-copy GPU tensor → `[0,1]`
  float16 — **not** `[-1,1]` like the input path) with CPU SHM as a lazy-reconnecting
  fallback. Both branches now survive the startup race with TD's COMP activation.
- **TD-side prerequisite:** `shmem_out_cn` must be wired as a CUDA-IPC Sender publishing to
  `<stream>_control_ipc` (add `CUDAIPCExtension` + `cuda_ipc_parexec` ParExecute DAT,
  mirroring `shmem_out`). Until then, the Python consumer falls back to CPU SHM.
