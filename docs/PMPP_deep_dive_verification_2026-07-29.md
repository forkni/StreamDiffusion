# PMPP "Deep Dive" concept verification against StreamDiffusion

Verification pass against `docs/plans/Welcome to the Deep Dive.md`, a podcast transcript
distilling *Programming Massively Parallel Processors* (4th ed.): the five-point GPU
optimization checklist (occupancy, coalesced global memory access, minimizing control
divergence, tiling into shared memory, privatization to avoid atomics), plus CUDA streams
for host/device overlap and CUDA-aware MPI for multi-node clusters.

Methodology: source audit of `StreamDiffusion/src`, cross-checked against a citation-
verification pass (grep + semantic search, index synced 2026-07-29), then a real measurement
pass (Phase A) — `nsys`/`ncu` profiling of the deployed engine plus two standalone
CUDA-event micro-benchmarks — executed *before* this report was written, so every verdict
below is either backed by a `file:line` or a number, not an assertion.

## 1. Framing

The transcript describes **warp-level CUDA**: how a hand-written kernel should be shaped so
occupancy is high, global loads are coalesced, warps don't diverge, hot data is staged
through shared memory/registers, and inter-thread writes avoid atomics.

**StreamDiffusion has zero hand-written kernels.** No `.cu`/`.cuh` files, no Triton/CuPy/
numba kernels, no `torch.utils.cpp_extension` under `src/` (confirmed by both grep and
semantic search). The actual UNet, VAE, and attention math run inside a TensorRT engine
built from ONNX-traced PyTorch ops; TensorRT's tactic selector — not any code in this repo —
picks the CUDA kernels and their tile sizes, occupancy, and memory-access pattern.

So for items 1–4 of the checklist, "not implemented" and "delegated by design" are different
verdicts, and conflating them would misreport the codebase. Sections 2–4 below classify each
item accordingly. Meanwhile the transcript's *scale-up* half — streams, pinned memory, async
host/device overlap, avoiding round-trips — **is** something this repo's Python layer
controls directly, and it turns out to already be implemented to an unusually high standard.
`pipeline.py:67` even carries a direct citation — `# Grounding: PMPP Ch.5 roofline; CUDA HB
Ch.6 stream overlap` — though precisely: that comment annotates a TF32/`cudnn.benchmark`
flag block, not roofline or stream code at that exact site. It is evidence of *intent and
lineage* (a prior audit worked from the same two books), not evidence that roofline/stream
logic lives at `pipeline.py:67` itself.

## 2. Concept-by-concept

| Transcript concept | Status | Evidence |
|---|---|---|
| 1. Maximize occupancy | **Delegated to TRT; now measured.** Per-arch build profiles set `builder_optimization_level`, `max_num_tactics`, `avg_timing_iterations`, tactic sources; `max_aux_streams` deliberately left to TRT's heuristic (never assigned, confirmed 3×). **Measured (§3):** the dominant UNet kernel family is capped at 2 blocks/SM by shared-memory usage, `warps_active%` 7–14%. This is the real, measured bottleneck. | `acceleration/tensorrt/utilities.py:150-270,279-419,153,188-190,345-348`; `logs/ncu_config_sdtd_profile_cfg_eq7oz4t3_detailed_20260729_025807.ncu-rep` |
| 2. Coalesced global access | **Mostly delegated; measured not-applicable on the hot path.** Python-level analog (layout/contiguity) is handled at TRT bind boundaries. DRAM throughput never exceeds 64% of peak even at its busiest measured instance (§3) — not saturated, so this is not the binding constraint here. **Gap, quantified (§3, A3a):** the BGRA pack does 3 separate strided 1-byte-per-4 writes, duplicated across two near-identical functions; measured ceiling ≈11 µs/frame. **Gap, unquantified:** no `channels_last` anywhere for the PyTorch VAE/preprocessing convs. | `wrapper.py:1131-1133`, `wrapper.py:1191-1193`; `channels_last`/`memory_format`: zero hits, grep + semantic |
| 3. Minimize control divergence | **N/A at Python level** (no warp control). The host-level analog — a data-dependent branch forcing a GPU sync — is already solved: 1-frame-delayed pinned readback in the similar-image filter; `torch.where` gating instead of a branch in `get_nn_feats`; `fi_strength`/`fi_threshold` kept as tensors (no `.item()`) so the CUDA graph stays static. | `image_filter.py:28-60`; `acceleration/tensorrt/models/attention_processors.py:34-35,206-216` |
| 4. Tiling / on-chip reuse | **Delegated to TRT + SDPA flash attention**, plus a real, repo-owned L2-persistence carve-out (Ampere-gated `cudaLimitPersistingL2CacheSize` + one `cudaAccessPolicyWindow` — CUDA permits only one per stream, this is not "per-tensor windows" — pinning the single largest hot attention weight) and circular KV/O caches. **Gap, measured and inverted (§3, A3b):** `get_nn_feats` materializes the full `[B,N,M]` cosine matrix then reduces — the canonical untiled reduction the transcript warns about — but a naive Python-loop "tiled" rewrite measured **1.5×–6.7× slower**, not faster (see §3). | `tools/cuda_l2_cache.py:316-319,352-370` (carve-out + access-policy window; cites corrected 2026-07-29); `acceleration/tensorrt/models/attention_processors.py:9-35`; `pipeline.py:1157-1180` |
| 5. Privatization (avoid atomics) | **Nothing to privatize.** Zero GPU atomics repo-wide (no `scatter_add`/`index_put`/`index_add`/`atomicAdd`/histogram). Every "atomic" grep hit is unrelated: atomic *file* writes for engine caching, and a threading lock — not GPU atomics. The spirit of privatization appears as ping-pong double buffers (allocation at `pipeline.py:629-634` is *unconditional*; only the swap at `pipeline.py:1217-1224` runs inside the `denoising_steps_num > 1` branch — corrected 2026-07-29) and per-worker private buffers. | `pipeline.py:629-634,1217-1224`; `base_orchestrator.py:39-49`; false positives: `utilities.py:455-474`, `fp8_quantize.py:240`, `stream_parameter_updater.py:48` |
| 6. Streams / host-device overlap | **Strongly implemented.** Pinned host staging on both H2D and D2H boundaries (input staging `pipeline.py:423-427`; `_output_pin_buf` readback) with `non_blocking=True` — the IPC *pack* buffer itself is device-resident (a D2D write target), not pinned host memory as an earlier draft implied (corrected 2026-07-29); dedicated streams with `Event`-based cross-stream barriers and `record_stream` allocator hygiene; background stream + thread-pool 1-frame pipelining for pre/post-processing; CUDA-graph capture with fallback; zero-copy CUDA-IPC path to TouchDesigner. **Deliberate exception, found while verifying (§3):** the IPC output boundary is forced to *blocking* export — `_lazy_init_ipc_exporter` overrides `ExportPolicy` to `export_sync=True` unless `CUDALINK_EXPORT_SYNC=0` is set, because the pack buffer is persistent/reused every frame and async export would race the next frame's overwrite (ADR-0001). | `pipeline.py:423-427,1317-1321`; `processors/trt_base.py:89-279`; `processors/ipadapter_embedding.py:34-69`; `wrapper.py:1024-1089,1110-1114,1144-1151`; `docs/adr/0001-cuda-link-as-external-dependency.md` |
| 7. Kernel-launch overhead | **Implemented.** CUDA graphs, pre-allocated buffers, `set_tensor_address` rebinding guarded on the graph fast path. (Buffer allocation itself lives in `allocate_buffers`/`_can_reuse_buffers`, outside the cited doc range — that range documents the pattern.) | `acceleration/tensorrt/utilities.py:1244-1295` |
| 8. CUDA-aware MPI / multi-node | **Absent by design.** No `torch.distributed`, NCCL, or multi-GPU anywhere (grep + semantic search, zero hits). Single-GPU real-time application — correctly out of scope; see §4. | repo-wide search: no hits |
| 9. Constant memory broadcast | **N/A** (no hand-written kernels to place data into `__constant__`). Nearest analog — L2 persistence for hot weights — is implemented (see row 4). | `tools/cuda_l2_cache.py` |
| 10. Output- vs. input-centric decomposition | Conceptual, not directly applicable. The pipeline is inherently output-centric (gather); there is no scatter/histogram stage to reformulate. | — |

## 3. Genuine gaps, measured

Every claim in this section is a number from a real run, with the exact reproduction command
in §6.

### 3.1 Occupancy — the real, measured constraint (item 1/4)

`ncu --set detailed` on the deployed fp16 engine (`StreamDiffusionTD/td_config.yaml`,
`sdxl-turbo--hee14d1f99dde--res-512x512`), 10 kernel launches captured
(`--kernel-regex "cutlass|gemm" --launch-skip 5 --launch-count 10`):

| Kernel variant (grid) | shmem limit | warps_active% | SM thpt% | DRAM thpt% | L1TEX thpt% | LTS thpt% | LTS hit% |
|---|---|---|---|---|---|---|---|
| 128x2 tile, `(24,3,1)` — 5/10 launches | 2 blocks | 7.5–8.4 | 6.5–6.9 | 24.2–25.8 | 27.0–29.6 | 21.8–23.3 | ~80.5 |
| 128x2 tile, `(24,3,3)` | 2 blocks | 13.8–13.9 | 16.5–16.9 | 61.9–63.5 | 37.5–38.1 | 55.8–57.1 | ~80.1 |
| 64x1 tile, `(24,12,1)` | 10 blocks | 18.0–18.2 | 13.8–14.1 | 54.0–55.1 | 31.6–32.0 | 46.0–46.8 | ~80.5–80.8 |
| `fmha_cutlassF_f16` attention | 5 blocks | 8.2 | 2.7 | 5.1 | 16.4 | 3.4 | ~81.0 |

`gpu__compute_memory_throughput` equals `gpu__dram_throughput` exactly on every row — DRAM is
the binding memory pipe for this kernel family, not L1TEX/LTS — but it tops out at 64% even
at the busiest instance and sits at 24–26% for the `(24,3,1)` instances (5/10 launches). **Not
saturated.** The measured limiter is occupancy: the dominant 128x2 tile is capped at **2
blocks/SM by shared-memory usage** (`launch__occupancy_limit_shared_mem=2` vs. hardware max
24), which caps `warps_active%` at 7–14% regardless of memory traffic. SM throughput and warp
occupancy move together; DRAM/L1TEX/LTS throughput scale with grid size (more waves in
flight → more concurrent requests) but never become the bottleneck. L2 sector-hit-rate is a
flat ~80% across every kernel including the memory-light attention kernel — healthy reuse,
not a factor.

**Conclusion:** the transcript's item 2 (coalescing) is measured **not applicable** on this
pipeline's hot path — there's no bandwidth wall to relieve. The real, measured constraint is
items 1/4 (occupancy via TensorRT's shared-memory tile-size choice), which is TRT tactic
selection, not Python-level code this repo controls. This upgrades the earlier `basic`-set
occupancy finding from inference to direct confirmation.

**Amendment (2026-07-29, follow-up pass — A1 register-limiter verdict):** the original
capture exported only the shared-mem/block limiters, so it could not distinguish
"shared-mem-capped defect" from "correctly-tuned register-tiled CUTLASS tile" (PMPP 5e §15.7
treats low occupancy as the *intended* operating point for register tiling, ~255
regs/thread). Re-exporting the same rep offline
(`ncu --import <rep> --page raw --csv --metrics launch__occupancy_limit_registers,launch__occupancy_limit_warps,launch__registers_per_thread,sm__maximum_warps_per_active_cycle_pct`,
saved to `logs/a1_limiters_export_20260729.csv`) resolves it: **this section's conclusion
stands.** The dominant 128x2 GEMM is genuinely shared-memory-limited (2 blocks/SM by smem
vs. 6 by registers; 80 regs/thread) and no captured kernel approaches the 255-register
profile (64/80/128 regs/thread) — the §15.7 register-tiled interpretation does not apply.
Two secondary corrections: (a) the 64x1 tile and `fmha` kernels are actually
*register*-limited (8 vs. 10 and 4 vs. 5 blocks respectively — the table's "shmem limit"
column is not the binding limiter for those two rows), with much looser ceilings (67%/33%
max theoretical warps vs. the dominant kernel's 16.7%); (b) the `(24,3,1)` launch count is
5/10, not 6/10 as originally stated (fixed above). Full evidence:
`docs/profiling/fp8_fi_gates_2026-07-29.md`.

### 3.2 BGRA pack — real but negligible (item 2)

Standalone CUDA-event micro-benchmark (20-iter warmup, 200-iter timed loop,
`torch.cuda.synchronize()` before/after — not `CUDA_LAUNCH_BLOCKING=1`, which would inflate
these numbers ~8.6× per §3.3/§6), correctness-verified with `torch.equal` before timing:
current 3-pass strided write (`wrapper.py:1131-1133`, duplicated at `wrapper.py:1191-1193`)
vs. a single `buf[..., :3] = rgb_hwc.flip(-1)` write.

| Size | current (3-pass) | single `flip(-1)` | delta | speedup |
|---|---|---|---|---|
| 512×512 | 29.6 µs | 25.8 µs | −3.8 µs | 1.15× |
| 1024×1024 | 27.0 µs | 16.3 µs | −10.7 µs | 1.65× |

The gap is real but the ceiling is exactly as small as predicted from `unet_step`'s p50 20.5
ms budget: at most ~11 µs/frame, ≈0.05% of `unet_step`, well under the noise floor of the
`glue.ipc_pack_rgba` p50 itself (0.077 ms, `profiler_logs/sdtd_benchmark_20260729_003521_
stats.json` — that region also includes denormalize/clamp/permute/contiguous/buffer-realloc-
check, not just these 3 writes). The A1 profiler run additionally shows `glue.ipc_pack_
unit_rgba` never fires — the second, duplicated function is dead in the current
`send_controlnet_preview: false` configuration, so it's code debt, not a live perf gap.

**Recommendation:** real, cheap, low-risk one-line fix — worth doing opportunistically when
touching that code, not worth a dedicated effort.

The blocking `exporter.export()` call at the same boundary (`wrapper.py:1036`) was
deliberately **not measured** — it requires a live cuda-link receiver (TouchDesigner or a
mock consumer) on the other end of the named `SharedMemory` to produce a real "publish"
outcome; timing it with no consumer attached would measure failure/backpressure behavior, not
the real cost. This is flagged as an open question, not answered with a misleading number.
Per ADR-0001, that export is forced synchronous because both pack buffers are persistent and
reused every frame — double-buffering `_ipc_pack_buf` is the actual prerequisite for async
export, and a faster pack kernel alone would not unlock overlap at this boundary.

### 3.3 `get_nn_feats` — real gap, negative naive fix (item 4)

`acceleration/tensorrt/models/attention_processors.py:9-35` (reduction at lines 28–35) materializes the full `[B, N, M]`
cosine similarity matrix via `bmm`, then reduces with `.max(dim=-1)` — the canonical untiled
reduction the transcript warns about, and it **does run in production**: contrary to an
earlier draft of this investigation's assumption, feature injection is not simply "off by
default" — `StreamDiffusionWrapper.__init__` defaults it `False` (`wrapper.py:147`), but the
YAML config loader defaults it `True` when the key is absent (`config.py:171-177`, with an
explicit log line), and the shipped production config sets `use_feature_injection: true`,
`fi_strength: 0.754`, `fi_threshold: 0.98` (`td_config.yaml:49-51`). So this reduction is live
on the deployed path.

Same micro-benchmark harness, 4 layer sizes representative of the SDXL UNet's attention
resolutions (B=1, `cache_maxframes=4` ⇒ `M=4N`), correctness-verified (`max_out_diff=0.0000`
vs. the current implementation at every size) before timing a chunked online-argmax reduction
(`chunk=512`, looping `bmm` over `M` in a Python `for` loop) against the current
full-materialize `bmm`+`max`:

| Layer (spatial) | N | M | C | current (full bmm+max) | chunked(512) | result |
|---|---|---|---|---|---|---|
| down, 64×64 | 4096 | 16384 | 320 | 0.570 ms | 2.791 ms | **4.9× slower** |
| mid, 32×32 | 1024 | 4096 | 640 | 0.120 ms | 0.803 ms | **6.7× slower** |
| mid/low, 16×16 | 256 | 1024 | 1280 | 0.132 ms | 0.310 ms | **2.4× slower** |
| bottleneck, 8×8 | 64 | 256 | 1280 | 0.135 ms | 0.203 ms | **1.5× slower** |

The chunked version does cut peak memory (e.g. 191 MB → 61 MB at the largest layer), but a
Python-level loop over `bmm` calls pays a per-chunk kernel-launch-overhead tax that swamps any
bandwidth saved. The full `[B,N,M]` matrix (max 134 MB at the largest layer) already fits
comfortably in GPU memory, and per §3.1, DRAM throughput on the dominant UNet kernels never
exceeds 64% of peak — there was no bandwidth pressure to relieve in the first place.

**Conclusion:** `get_nn_feats` is a genuine but *negatively* actionable PMPP-checklist gap.
It is correctly identified as untiled, but the naive "reproduce tiling as a Python loop" fix
is worse than the status quo. A real win requires a single fused kernel (the FlashAttention/
online-softmax pattern — one kernel launch, tile loop inside CUDA, not a loop of separate ops
each paying launch overhead). This is the load-bearing finding for §5's custom-kernel
evaluation, not a quick win.

### 3.4 `channels_last` — gap, unquantified

No `channels_last`/`memory_format` usage anywhere in the codebase (grep + semantic search,
zero hits). This applies to the PyTorch-side VAE encode/decode and any preprocessing convs
that run outside the TRT engine. Not micro-benchmarked in this pass — no isolated call site
was identified cheaply enough to benchmark standalone the way the other two gaps were: it
would require running the actual VAE forward pass in both memory formats, which is closer to
an implementation experiment than a measurement. Flagged as a real gap with an unknown-but-
plausibly-small ceiling, not measured.

## 4. Deliberate non-goals

- **CUDA-aware MPI / multi-node** — zero `torch.distributed`/NCCL hits, confirmed by grep and
  semantic search. This is a single-GPU, single-process real-time application; multi-node
  distribution is not a relevant axis and adding it would be scope creep, not an optimization.
- **Frame-level pipelining of `encode → UNet → decode`** — the transcript's stream-overlap
  lesson is already applied *within* a frame (§2 row 6), but overlapping stage N's decode with
  stage N+1's encode across frames is not done. Deliberately: this is a real-time interactive
  pipeline, and pipelining across frames would add a frame of latency for throughput that
  isn't the bottleneck (per §3.1, the dominant cost is TRT-internal occupancy, not host
  overlap).
- **Privatization** — there are no GPU atomics anywhere in this codebase to privatize away
  (§2 row 5). Not a gap; there is nothing here for that technique to apply to.

## 5. Custom-kernel evaluation

For each checklist item, what a hand-written kernel would buy vs. cost, given the measured
evidence above:

- **Occupancy / tiling (items 1, 4)** — the only place a custom kernel plausibly wins is
  `get_nn_feats` (§3.3): a fused, single-launch tiled reduction (online max, FlashAttention-
  style — accumulate the running max and its argmax index across `M`-chunks inside one kernel,
  never materializing `[B,N,M]`) would eliminate both the memory-materialize cost and the
  Python-loop launch-overhead tax that made the naive rewrite lose. This is a real,
  well-scoped candidate — but it's traced into ONNX and built into the TRT engine
  (`export_wrappers/unet_unified_export.py:15-52` `_collect_fi_processors`), so it must stay
  ONNX-export-compatible (a TensorRT plugin, not a bare PyTorch/Triton op — TRT will not
  invoke a custom op inside its own kernel graph without one), and any change requires an
  engine rebuild to validate.
- **Coalescing (item 2)** — the BGRA pack (§3.2) is small enough (≤11 µs/frame ceiling) that a
  custom kernel is not justified; a one-line PyTorch fix already captures nearly all of the
  available win at effectively zero engineering cost.
- **Everywhere else** — TensorRT already owns kernel selection and tiling for the UNet/VAE/
  attention math; a hand-written kernel would have to out-tune TRT's own tactic search on the
  same hardware, which is a high-cost, low-confidence bet with no measured evidence (§3.1
  shows the actual constraint is a TRT-internal tile-size choice, not something exposed to
  Python).

**Triton availability, for context:** `tools/install-tensorrt.py:36-37` conditionally
installs the `triton-windows==3.4.0.post21` fork **on Windows only**, and
`acceleration/sfast/__init__.py:21-24` probes for it to enable `config.enable_triton = True`
on the *stable-fast* backend. So Triton codegen may already be active on the
`acceleration: "sfast"` path — but that's compiler-generated code from `torch.compile`, still
zero hand-written kernels, and it is not the TensorRT path used in production (the sfast
backend is an alternative, not the deployed configuration).

**Net recommendation:** the only custom-kernel candidate with a clear, measured case for it is
a fused tiled `get_nn_feats`, and even that requires a TRT-plugin-level investment (ONNX
export compatibility + engine rebuild) to realize — not a quick win. Nothing else on the
checklist has a measured gap large enough to justify hand-written kernel work.

## 6. Measurement appendix

Two tooling bugs were found and fixed in `scripts/profiling/profile_ncu.py` while producing
the numbers above. Both fixes are additive to a dev-only script never imported by the
runtime; no behavior in `src/streamdiffusion` changed.

**Bug 1 — `--kernel-name` needs an explicit `regex:` prefix.** `ncu --help`:
`--kernel-name <name>` is an *exact* match; regex requires the literal `regex:` prefix. The
script previously passed the raw value straight through
(`ncu_cmd += ["--kernel-name", args.kernel_regex]`) even though the flag is *named*
`--kernel-regex`. Every partial kernel name matched zero kernels; ncu exited 0 with
`==WARNING== No kernels were profiled.` — this had never worked in this repo. Fixed at
`profile_ncu.py:197-201`:

```python
if args.kernel_regex:
    _kn = args.kernel_regex
    if not _kn.startswith(("regex:", "base:", "demangled:", "mangled:")):
        _kn = f"regex:{_kn}"
    ncu_cmd += ["--kernel-name", _kn]
```

**Bug 2 — `--set memoryworkload` / `--set source` are not real ncu set names.**
`ncu --list-sets` (this install, 2026.1.1) only defines `basic`, `detailed`, `full`,
`nvlink`, `pmsampling`, `roofline`. Passing an invalid set name does **not** error: ncu prints
`==WARNING== No metrics to collect found in sections.`, runs a trivial "1 pass" per kernel
(vs. the 5–10 passes a real multi-section set needs), and writes a **valid-looking
`.ncu-rep`** with correctly-targeted kernel launches but **zero performance counters** — not
even the always-present `gpu__time_duration.sum`. This is functionally identical, in symptom,
to Bug 1: both produce a successful-exit-code artifact that looks right and contains nothing.
Caught only by diffing column counts against a known-good `basic`-set report. Fixed by
changing `profile_ncu.py`'s `--set` choices to the real set list, substituting `detailed`
(which contains `MemoryWorkloadAnalysis` — the section this investigation's coalescing/DRAM
numbers actually needed) for the invalid `memoryworkload`.

**Guardrail — always sanity-check a `.ncu-rep` before trusting it:**

```bash
ncu --import <rep> --page raw --csv --metrics gpu__time_duration.sum
```

If the value column is blank, the report has no real data, regardless of exit code or file
size. Also verify each individual `--metrics` name against the report header before relying
on it — `ncu --query-metrics` listing a base metric name does not guarantee every plausible
suffix combination resolves; many silently drop with no error (confirmed-working vs.
confirmed-broken metric name lists are in §3.1's source data,
`logs/ncu_config_sdtd_profile_cfg_eq7oz4t3_detailed_20260729_025807_metrics.csv`).

**Working commands used for this report:**

```bash
# Rank kernels by GPU-time share before targeting ncu (nsys, ~1 min, no CUDA_LAUNCH_BLOCKING)
venv/Scripts/python scripts/profiling/profile_nsys.py --target benchmark --config StreamDiffusionTD/td_config.yaml

# Bounded ncu pass on the top kernel family (detailed set = MemoryWorkloadAnalysis + Tile + SourceCounters)
STREAMDIFFUSION_PROFILE_TRT=1 venv/Scripts/python scripts/profiling/profile_ncu.py \
  --config StreamDiffusionTD/td_config.yaml --set detailed \
  --kernel-regex "cutlass|gemm" --launch-skip 5 --launch-count 10 --csv
```

**Further guardrails learned running these:** run `ncu` with a generous timeout (`run_in_
background` or ≥10 min) — a Bash-tool timeout truncating the *log stream* while `ncu` keeps
running to completion looks exactly like a hung/failed run. Always check for the `.ncu-rep`
file on disk before concluding a run failed. **Never cite a timing number captured while
`CUDA_LAUNCH_BLOCKING=1` was set** — `profile_ncu.py` sets this for accurate per-kernel
attribution, and it inflates wall-clock timings ~8.6× (`unet_step` p50 20.488 ms clean vs.
175.808 ms under `ncu` in an earlier run of this investigation). Only
`profiler_logs/sdtd_benchmark_20260729_003521_stats.json` (the A1 run, no `ncu` attached) is
a valid timing baseline in this investigation; every `profiler_logs/*_stats.json` written
*during* an `ncu` run is timing-invalid by construction and must never be quoted as a
performance number.

## 7. Engine-drift note

The currently deployed engine is **FP16** (`StreamDiffusionTD/td_config.yaml` has `fp8:
false`, resolving to `sdxl-turbo--hee14d1f99dde--res-512x512`) — this is the engine measured
throughout §3. An earlier audit (2026-07-10) measured an **FP8** engine
(`sdxl-turbo--fp8v3--h18b4bb48d936--res-512x512`, FP8 GEMMs ≈45% of kernel time,
`cutlass_80_wmma` 16.5%) that still exists on disk but built from a different, no-longer-
matching config.

Attempting to re-run the fp8 measurement for this report surfaced a second, independent
finding: **the checked-in `configs/profiling/profiling_fp8v3.yaml` no longer resolves to the
on-disk fp8 engine.** It resolves to a fresh hash (`h685d6b0fd4fa`) that doesn't exist as a
built engine; disabling feature injection in a scratch copy produces a third, still-different
hash (`h903146e6f7cf`); `get_engine_path`'s cache key also encodes `static_batch_size`
(`wrapper.py:2111`, derived from `t_index_list`/`cfg_type`), and the config's `t_index_list:
[16]` (a single step) likely doesn't match whatever produced the on-disk engine's
`batch_size: 2` (`build_log.jsonl`: built 2026-07-25, 2226.69 s). Reconstructing the exact
original build parameters, or paying a ~37-minute rebuild, was judged out of scope for a
read-mostly measurement pass — so this report measures fp16 only, and flags as an independent
finding that **the 2026-07-10 fp8 baseline is not currently reproducible from the checked-in
config.**

Whether FP16 is the intentionally correct production choice, or the deployment silently
drifted off the FP8 baseline the earlier audit measured, is a genuine open question this
report surfaces rather than assumes. If FP8 is still the intended target, the config drift
(and the missing rebuild provenance for the fp8 engine that IS on disk) should be resolved
before the next fp8 measurement is attempted.

**Amendment (2026-07-29, follow-up pass): provenance CLOSED.** FP8 is the intended
production target — the user rebuilt the fp8 engine the same day (with V2V cached-attn,
feature injection, and ControlNet enabled) and it landed in the **identical** hash dir
`sdxl-turbo--fp8v3--h18b4bb48d936--res-512x512` (build_log.jsonl: 2026-07-25 2226.69 s and
2026-07-29 2087 s, same dir). Independently, reconstructing `get_engine_path`'s canonical
string from `StreamDiffusionTD/td_config.yaml` (fp8 + cached_attn + FI + CN, static batch 2,
`cachef4`, `optlvl4`) SHA1s to exactly `18b4bb48d936` — td_config.yaml *is* the builder
config; the on-disk fp8 engine always had V2V+CN. The stale `configs/profiling/
profiling_fp8v3.yaml` was rewritten in place to that identity, and three pinned arm configs
were added (`configs/profiling/fp8_fi_on.yaml`, `fp16_ab.yaml`, `fi_ablation_off.yaml`, all
`build_engines_if_missing: false`) with an offline SHA1 preflight validating every arm
before any GPU run. Two collateral findings: (a) the fp16 engine measured throughout §3
(`hee14d1f99dde`) was reconstructed by brute force as **batch-1 / unpinned-cache / optlvl2**
— NOT comparable to the fp8 engine's batch-2/cachef4/optlvl4 identity, so a true batch-2
fp16 A/B twin (`h4caa2410d251`) was built 2026-07-29 (616.8 s, 4939 MB) for the FP8-vs-FP16
comparison in `docs/profiling/fp8_fi_gates_2026-07-29.md`; (b) latent hash-key bug,
recorded not fixed: the cache key encodes the *raw* `use_feature_injection` flag
(`wrapper.py:2102`) while the effective value is `use_feature_injection AND use_cached_attn`
(`models.py:514`) — two configs differing only in that interaction can build identical
engines under different hashes.
