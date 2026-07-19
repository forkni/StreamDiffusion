# StreamDiffusion Performance & Best-Practices Audit — 2026-07-10

Read-only audit against CUDA / PyTorch / TensorRT best practices, combining static code review
with measured profiling of the real-time denoising loop. No source files were changed — this
document is the only artifact produced.

**Scope:** `src/streamdiffusion/{pipeline.py,wrapper.py}`, `acceleration/tensorrt/*`, `modules/*`,
`preprocessing/*`. **Config profiled:** StreamDiffusionTD "Quality/FP16" preset — SDXL-turbo,
512×512, img2img, FP8 TensorRT engines, CUDA graphs enabled.

**Repo state at audit time:** branch `refactor/py-code-review-remediation` (2 commits ahead of
`origin/SDTD_032_dev`), HEAD `6718511`. The immediately preceding commit (`5a44f34`, same day)
already remediated exception-hygiene issues in `pipeline.py`, `ipadapter_module.py`,
`controlnet_module.py`, `image_processing_module.py`, `latent_processing_module.py`,
`acceleration/tensorrt/__init__.py`, `engine_manager.py`, and the TRT export wrappers — but **not**
`wrapper.py`, which was untouched. Every finding below was checked against this HEAD; where a
finding sits in a file/region that commit touched, its exact lines were diffed to confirm the
issue is distinct from what was already fixed (see "General code quality" §, findings #4–#10).

---

## Prioritized quick wins

Ranked by (impact × how directly it affects the measured per-frame budget) ÷ effort. "Measured"
column links to the Phase 2 data below; "Static" findings are code-review-only.

| # | Finding | Impact | Effort | Evidence |
|---|---|---|---|---|
| 1 | **~13.6ms/frame (66% of the entire 20.8ms budget) inside `unet_step` is unattributed by any profiler region** — only ~4.6ms of its 18.2ms median is explained by measured children (`trt_infer`×3 + staging + scheduler). The unmeasured remainder is CFG-buffer prep, hook dispatch, and conditioning-lookup Python/CUDA logic that today has zero instrumentation. | **High** — largest single lever on real-time throughput | Small (instrument first, then optimize what's found) | `profiler_logs/*_stats.json`; `pipeline.py:820-1041` (`unet_step`) |
| 2 | `set_tensor_address` is reissued for every I/O tensor on **every frame**, even on the already-captured CUDA-graph replay fast path, defeating part of the point of graph capture (eliminate repeated host-side API calls) | High — recurring per-frame host overhead × every TRT engine in the pipeline | Small — guard with `if not (use_cuda_graph and self.cuda_graph_instance is not None):` | `acceleration/tensorrt/utilities.py:1118-1125` |
| 3 | Engine / timing-cache files are written non-atomically (`open(...,"wb").write()` straight to the final path); an interrupted multi-minute TRT build leaves a truncated file that the next run's `os.path.exists()` check silently treats as a valid cache hit | High — silent corruption / wasted rebuild risk | Small-medium — write to `.tmp` + `os.replace()` | `acceleration/tensorrt/utilities.py:779-789,907`; `builder.py:284-285` |
| 4 | RNG/seed silently resets to the hardcoded default (`seed=2`) on **every runtime prompt change** — `StreamDiffusion.prepare()`'s `generator` parameter defaults to a module-load-time-shared `torch.Generator()` mutable default, and neither `wrapper.py` `prepare()` branch (single-prompt or blending) forwards `self.generator`/`self.current_seed` when calling `stream.prepare()` | High — visible, reproducible loss of user-configured determinism in production | Small — thread `self.generator`/seed through both wrapper `prepare()` call sites; fix the mutable default | `pipeline.py:402-403`; `wrapper.py:493-498,519-524` |
| 5 | `__call__` / `postprocess_image("latent")` return aliased, persistent internal buffers (`_image_decode_buf`, `_prev_image_buf`) that get `.copy_()`'d in place next frame — any caller that retains the reference (queues it, hands to async consumer) sees silent mutation | High — silent data corruption for any downstream buffering consumer | Medium — `.clone()` on public returns of these buffers, or document the aliasing contract loudly | `pipeline.py:1272-1285`; `wrapper.py:1013-1014` |
| 6 | ControlNet's dynamic-shape profile floor (384px) is narrower than UNet's (256px) — a resolution in [256,384) that's valid for UNet will hard-fail specifically on the ControlNet engine when dynamic shapes + ControlNet are combined | Medium-high — real runtime failure risk in a reachable config | Small — lower `min_ctrl_h`/`min_ctrl_w` to 256, or share one constant | `acceleration/tensorrt/models/controlnet_models.py:68-73` vs `models/models.py:160-161` |
| 7 | NSFW safety-checker `.item()` blocks the frame on a synchronous GPU→host readback every frame it's enabled — no async pattern, despite the codebase already having the correct template one file away | High when the (opt-in, default-off) feature is enabled | Small-medium — reuse `image_filter.py`'s 1-frame-delay pinned-buffer pattern | `acceleration/tensorrt/runtime_engines/unet_engine.py:515`; `wrapper.py:1308` |
| 8 | `output_type="pil"` path does a blocking, unpinned `.cpu()` into freshly-allocated pageable memory every frame — the `"np"` path 30 lines above already does this correctly (pinned buffer + `non_blocking=True` + single `Event.record()/synchronize()`) | Medium-high for `"pil"` consumers | Small — route `"pil"` through the same `_output_pin_buf`/`Event` machinery | `wrapper.py:1264` vs `wrapper.py:1019-1037` |
| 9 | Stringly-typed OOM detection (`"out of memory" in str(e).lower()`) duplicated in two engine exception handlers, instead of `isinstance(e, torch.cuda.OutOfMemoryError)` (already used correctly elsewhere in the same file) | High — a PyTorch/TensorRT message-wording change silently breaks the OOM→CPU-fallback path | Small-medium — extract one `_is_oom_error(e)` helper, `isinstance` check first | `wrapper.py:2342-2349,2400-2407` |
| 10 | SDXL pipeline-type-mismatch retry failure is caught and only logged as a warning — execution continues with the wrong (non-SDXL) pipeline object rather than failing fast | High — garbled output for SDXL models on this edge case, with no error surfaced | Small — raise, or fail fast at engine-build time | `wrapper.py:1504-1514` |
| 11 | No recovery path if `cudaGraphLaunch` fails on the fast-replay path — an uncaught `CUASSERT` takes down the entire real-time loop instead of falling back to `execute_async_v3` for one frame | Medium — rare trigger, severe blast radius | Medium — try/except → `reset_cuda_graph()` → fall back, re-capture next call | `acceleration/tensorrt/utilities.py:1125` |
| 12 | `_load_model` is a ~1390-line function combining pipeline load, LoRA fusion, VAE setup, and TRT/xformers/sfast acceleration dispatch for UNet+VAE+ControlNet+IPAdapter+safety-checker — the exact logic this audit was asked to verify is concentrated in one untestable function | High — maintainability/testability of the acceleration-mode selection logic itself | Large — split into `_load_pipe()`, `_setup_lora()`, `_setup_vae()`, `_build_tensorrt_engines()`, `_install_hook_modules()` | `wrapper.py:1321-2712` |

---

## Phase 2 — Measured per-frame budget

**Method:** `scripts/profiling/profile_nsys.py --target benchmark` against the cached SDXL-turbo
FP8 TRT engine, 3 extra warmup + 11 `torch.profiler`-captured + 10 `nsys`-gated frames (24 total
measured `predict_x0_batch` calls). Steady-state wall-clock per frame from the nsys `frame_nsys*`
NVTX spans: **~20.5–21.5ms (~47–49 FPS)**.

**Authoritative timing source.** The repo's own CUDA-event-based profiler
(`src/streamdiffusion/tools/gpu_profiler.py`, exported to `profiler_logs/*_stats.json`) is used
for every number below, **not** the raw `nsys stats --report nvtx_pushpop_sum` output.
Reason: NVTX push/pop ranges nested *inside* a captured CUDA graph fire only at capture time (the
first ~3 warmup passes), not on each graph replay — so their statistics wildly understate
steady-state cost for graph-replayed regions (e.g. `nvtx_pushpop_sum` shows `predict_x0_batch`
med=1.845ms, vs. the CUDA-event stats' correct **18.254ms**). Anyone profiling this pipeline again
should use the `profiler_logs/*_stats.json` export, or wrap regions of interest so they sit outside
the captured graph, not the raw NVTX summary.

### Per-region breakdown (p50, `profiler_logs/sdtd_benchmark_20260710_092608_stats.json`, n=24 unless noted)

| Region | p50 (ms) | p95 (ms) | Total (ms) | % of frame |
|---|---|---|---|---|
| `predict_x0_batch` (= `unet_step`, 1 call/frame) | 18.254 | 31.25 | 633.8 | ~88% |
| ┗ `trt_infer` (n=72, 3/frame) | 1.42 | 30.34 | 688.5 | (nested) |
| ┗ `trt.input_staging` (n=72, 3/frame) | 0.075 | 0.169 | 5.2 | (nested) |
| ┗ `sched.step_batch` (n=48, 2/frame) | 0.06 | 0.097 | 3.4 | (nested) |
| ┗ `sched.rebuild` (n=24) | 0.031 | 0.063 | 0.9 | (nested) |
| **Unattributed remainder inside `unet_step`** | **~13.6** | — | — | **~66%** |
| `encode_image` | 1.222 | 2.093 | 48.9 | ~6% |
| `decode_image` | 1.134 | 1.749 | 34.9 | ~5% |
| `glue.ipc_pack_rgba` | 0.086 | 0.118 | 22.0 | <1% |

`trt_infer`'s three calls (3×1.42ms) plus staging/scheduler sum to **~4.6ms**, against an
`unet_step` median of **18.2ms** — a **~13.6ms/frame gap with no nested `profiler.region()`
call inside it today**. This is the single largest lever in the pipeline: it's 66% of the entire
frame budget, and is exactly the span covered by "General code quality" finding #12 below
(`unet_step`'s CFG-buffer-prep, hook-dispatch loop, and SDXL-conditioning-cache lookup) —
recommend adding `profiler.region()` spans around those three sub-phases before attempting any
optimization, since right now it's not possible to tell whether the cost is Python dispatch
overhead, un-graphed CUDA kernels (RCFG blend, buffer copies), or something else.

### Kernel summary (`nsys stats --report cuda_gpu_kern_sum`, top contributors)

FP8 (`e4m3`) TensorRT GEMM/conv kernels dominate as expected for an FP8-quantized UNet:
`sm89_xmma_gemm_e4m3...` entries account for ~17.6% + 13.8% + 3.7% + 3.3% + ... (roughly 45%+ of
total kernel time across the top 20 rows) — confirms FP8 quantization is effectively applied to
the primary UNet path, no action needed there.

One notable outlier: **`cutlass_80_wmma_tensorop_f16_s161616gemm_f16...` is the single
second-largest kernel at 16.5% (1260 instances)** — this is an **FP16** tensor-core GEMM, not FP8.
Combined with several other `sm80`/`sm75` `f16f16` conv kernels and `fmha_cutlassF_f16` (flash
attention, 2.1%), roughly a quarter of total kernel time runs in FP16 rather than FP8. This audit
could not attribute these kernels to a specific engine (ControlNet vs. VAE vs. a text encoder) —
`ncu`'s source-correlation was blocked (see Limitations) — but given `fp8_quantize.py` exists in
this repo, it's worth checking whether ControlNet and/or the VAE are intentionally left at FP16 or
are simply not yet covered by the FP8 quantization pass. **Follow-up:** run `ncu` with
`--import-source yes` once GPU performance-counter access is available, to attribute this 16.5%
to a specific engine and confirm whether it's a legitimate FP8 candidate.

### Memory-copy summary (`nsys stats --report cuda_gpu_mem_time_sum`)

| Operation | Time (%) | Total (ms) | Count | Max single (ms) |
|---|---|---|---|---|
| H2D memcpy | 97.2 | 2167.3 | 1742 | 338.7 |
| memset | 2.6 | 58.6 | 166 | 51.6 |
| D2D memcpy | 0.1 | 2.4 | 502 | 0.03 |
| D2H memcpy | 0.1 | 1.9 | 1865 | 0.01 |

H2D dominates nsys's GPU-op-time view, but the total (2.17s) **exceeds** the entire measured
24-frame NVTX capture window (~1.7s) — this is almost certainly TRT engine deserialization /
model-weight upload during process startup (`prepare()`), captured because `nsys profile` wraps
the whole Python process, not a per-frame steady-state cost. **Not corroborated against
timestamps in this pass** — treat as a cold-start/engine-load latency note (relevant to
TouchDesigner scene-load / engine-switch UX), not a per-frame throughput finding. D2H per-frame
traffic (1865 copies, avg <1μs) is negligible and consistent with the pinned-buffer output path.

The first frame (`frame_warmup0`) costs ~1.16s — expected CUDA-graph-capture + cuBLAS/cuDNN
heuristic-search tax, paid once at startup per the `WARMUP_RUNS` mechanism. Not a runtime finding;
worth confirming this doesn't re-trigger on prompt/LoRA/engine hot-swap in the TD integration, as
that would surface as a UX stutter rather than a steady-state FPS regression.

### Limitations of this profiling pass

- **`ncu` (Nsight Compute) did not produce results.** First attempt hit `ERR_NVGPUCTRPERM` (GPU
  performance-counter access disabled); after enabling "Allow access to the GPU performance
  counters to all users" in NVIDIA Control Panel → Developer Settings, a retry (`--set basic`,
  `--launch-count 30`) was attempted but **did not bound itself** — it ran for ~105 CPU-minutes
  with the GPU pegged at full utilization and no completion in sight, and was terminated. Root
  cause: `ncu`'s counter collection replays each kernel serially per metric pass, and this
  benchmark loop launches its UNet/ControlNet/VAE work via **captured CUDA graphs**
  (`acceleration/tensorrt/utilities.py:1123-1125`) — profiling graph-launched kernels with `ncu`
  is known to multiply replay cost far beyond a normal (non-graphed) kernel count, so a naive
  `--set basic` pass over the whole benchmark loop doesn't terminate in practical time. Per-kernel
  SOL%, roofline, and occupancy for the FP8 GEMM kernels above are **not measured** — only the
  nsys kernel-time ranking (by wall-clock share) is available.
- `nsys stats --report gpu_gaps` doesn't exist in the installed Nsight Systems 2026.3.1 (the
  profiling README's report name is stale/version-mismatched); no direct GPU-idle-gap report was
  substituted.
- `--cn-scale` defaulted to `0.0` in this capture — ControlNet TRT engine structure is visible in
  the kernel/layer traces, but active per-frame ControlNet compute contribution was **not
  confirmed** in this run. A follow-up capture with `--cn-scale > 0` would be needed to isolate its
  cost.

---

## TensorRT acceleration stack (static review)

Overall maturity: this is a mature, incident-hardened hand-built TensorRT stack (not
`torch_tensorrt`) — GPU-tiered builder config, version/compute-capability/config-aware engine
cache-key naming, a documented (not accidental) single-CUDA-stream correctness fix, FP8
finite-scale validation, and graph-reset-on-buffer-reallocation logic all reflect real
incident-driven engineering. Remaining gaps are concentrated and mostly small/isolated:

1. **Non-atomic engine/timing-cache writes** — see quick win #3.
2. **`set_tensor_address` reissued every frame on the graph-replay fast path** — see quick win #2.
3. **ControlNet/UNet dynamic-shape profile-floor mismatch (384px vs 256px)** — see quick win #6.
4. **No recovery path on `cudaGraphLaunch` failure** — see quick win #11.
5. **`Engine.refit()` has no stream/context synchronization guard** (`utilities.py:565-665`,
   `refitter.refit_cuda_engine()` at `:663`) — violates TensorRT's REFIT requirement that no
   `IExecutionContext` be actively executing during a refit. Confirmed currently unreachable
   (`enable_refit`/`build_enable_refit` default `False`, zero callers of `.refit(` repo-wide) —
   a landmine only if REFIT-based weight hot-swap (e.g. live LoRA switching) is wired up later
   without adding the guard. Impact: low today / high if activated unguarded. Effort: small.
6. **Single shared CUDA stream across UNet/ControlNet/VAE/aux engines** (`wrapper.py:2263`,
   explicitly documented as a deliberate fix for a prior race in `controlnet_engine.py:155-157`).
   Standard multi-stream overlap (e.g. decode frame N-1 while denoising frame N) is traded away
   for correctness simplicity — do not revert casually. Impact: low (correctness fine, pure
   missed-optimization). Effort: large (would need careful multi-stream + `cudaEvent`
   cross-stream sync redesign to reclaim overlap without reintroducing the race).
7. **`opt_batch_size` defaults to `1`** in the public builder entry points (`builder.py:96` and 5
   call sites in `acceleration/tensorrt/__init__.py`), which would mistune the optimization
   profile if ever called without an explicit value. Verified **not currently triggered** — the
   real call path always threads `stream.trt_unet_batch_size` through correctly
   (`wrapper.py:2315` → `engine_manager.py:224-233` → `EngineBuilder.build()`). Dormant footgun
   for a future direct/test-harness caller. Impact: low (dormant). Effort: small — drop the
   default, make it a required kwarg.
8. **FP8 sanity-gate comment mismatches the actual threshold** (`builder.py:307` comment says
   "< 100", code checks `< 500` at `:323`) — cosmetic, update the comment.

---

## PyTorch hot-path / sync-free review (static review)

The per-frame diffusion core is in genuinely good shape: buffers are pre-allocated and reused
(`x_t_latent_buffer`, `_combined_latent_buf`, `_cfg_latent_buf`/`_cfg_t_buf`,
`_stock_noise_bufs` ping-pong, `_image_decode_buf`, `_prev_image_buf`), `pin_memory`/
`non_blocking=True` are applied correctly at every H2D/D2H boundary that matters (input staging,
similarity-filter skip-probability readback, the `"np"` output path), `torch.inference_mode()`
wraps every hot entry point, and the frame-skip similarity filter uses a genuinely sync-free
1-frame-delay pinned-buffer pattern (`image_filter.py`) that should be the template for the two
findings below, not the exception.

- **NSFW safety-checker blocking `.item()` every frame** — see quick win #7.
- **`"pil"` output path's blocking, unpinned `.cpu()`** — see quick win #8.
- **`_tensor_to_pil_safe` does two GPU-tensor truthiness syncs before its eventual `.cpu()`**
  (`preprocessing/preprocessing_orchestrator.py:601,606,610`) — three host syncs where the
  codebase's own `processors/base.py:tensor_to_pil()` (a near-duplicate helper) correctly does
  the `.cpu()` transfer first and compares ranges after, at line `:175-208`. This is a regression
  of a fix that already exists elsewhere in the same codebase. Impact: medium (gated to
  ControlNet/image-hook configs using non-tensor-native processors). Effort: small — reorder to
  match `base.py`'s pattern, or dedupe the two implementations.
- **Systemic, already self-tracked: ~20 of 28 preprocessor classes fall back to a per-frame
  PIL/CPU round-trip** (`preprocessing/processors/base.py:87-109`) — only 8 processors set
  `gpu_native = True`. The code already emits a one-time-per-class warning pointing at this exact
  gap ("[GPU-residency]... Set gpu_native=True..."), so this is not a new discovery, but it is the
  single largest concentration of avoidable per-frame syncs in the codebase, gated behind which
  preprocessor a user selects rather than always active. Worth surfacing to whoever owns that
  backlog item.
- **`_ipc_pack_rgba`/`_ipc_pack_unit_rgba` allocate fresh tensors every frame** (`wrapper.py:1052-
  1061,1092-1099`, `torch.full_like`/`torch.cat`) rather than writing into a persistent buffer like
  every other output path in the same file. Impact: low (small tensors, cheap kernels, but
  avoidable). Effort: small — hoist a persistent BGRA buffer, write in-place.
- **`channels_last` is absent repo-wide, and correctly so for the primary path** — the supported
  real-time deployment runs UNet and VAE through TensorRT, so there are no PyTorch convs in the
  hot path for `channels_last` to attach to. Only relevant if `acceleration="none"`/`"xformers"`
  (PyTorch VAE fallback) or one of the ~20 non-TRT preprocessors above is selected — not a gap in
  the intended architecture.

---

## General code quality & best practices (static review)

Two systemic patterns stand out across `pipeline.py`/`wrapper.py`/`modules/*`: (1) broad,
repeated tolerance for silently-swallowed exceptions that will make production failures invisible
until they surface as a worse downstream symptom, and (2) a handful of very large,
multi-responsibility functions that concentrate most of the acceleration-mode-selection logic this
audit was asked to review, making it hard to verify or safely extend. Findings already covered
above as quick wins are cross-referenced rather than repeated.

- **RNG/seed mutable-default reset** — quick win #4.
- **Aliased buffer returns (`"latent"` output, `prev_image_result`)** — quick win #5.
- **Stringly-typed OOM detection duplicated** — quick win #9.
- **SDXL pipeline-mismatch retry failure silently swallowed** — quick win #10.
- **`_load_model` god function (~1390 lines, 30+ params)** — quick win #12.
- **`cleanup_engines_and_rebuild` targets a hardcoded `"engines"` path**, ignoring the configured
  `engine_dir` (`wrapper.py:3062-3069`) — for any non-default `engine_dir` this OOM-recovery
  method either silently no-ops or, worse, could delete an unrelated `engines/` folder in the cwd.
  Impact: medium (breaks the one method whose entire purpose is OOM recovery, for any custom
  `engine_dir`). Effort: small — `engines_dir = str(getattr(self, "_engine_dir", "engines"))`.
- **Manual `__del__()` invocation in `cleanup_gpu_memory()`** (`wrapper.py:2907-2912,2939-2943`)
  — calling a dunder destructor explicitly does not stop the GC from invoking it again once the
  object is actually collected; native TRT/CUDA cleanup logic inside `__del__` can run twice,
  risking a double-free or a silently swallowed second exception. Impact: medium-high (double
  cleanup of native handles is a classic source of intermittent, hard-to-reproduce crashes).
  Effort: small — call a proper `close()`/`release()` if the wrapper exposes one, or just `del`
  the reference.
- **`_load_lora_with_offline_fallback` swallows every per-candidate exception with zero logging**
  (`pipeline.py:349-371`) — if candidate #2 of 5 fails for an unrelated reason (permissions,
  corruption) while the rest fail with "not found", the caller only ever sees the final "not
  found" — actively misleading during debugging. Impact: medium. Effort: small — add
  `logger.debug` inside the loop. **Not touched by the 2026-07-09 remediation commit.**
- **Bare `except Exception: pass` with zero logging in "Advanced model detection failed" fallback**
  (`wrapper.py:1855-1856`), inconsistent with every sibling handler within 50 lines that does log.
  Impact: medium (both detection paths failing leaves stale `model_type`/`is_sdxl` with no
  diagnostic trail). Effort: small. **In `wrapper.py`, which the 2026-07-09 remediation commit did
  not touch at all.**
- **Per-frame dynamic import + broad try/except-pass chains in the UNet hot-path hook**
  (`modules/ipadapter_module.py:434-521`, `_unet_hook`, invoked once per denoising step) — a
  `from diffusers_ipadapter...` import executes on every call instead of being hoisted to module
  scope, plus five separate `try/except Exception: pass` fallbacks, including one explicitly
  commented "Do not add fallback mechanisms" that nonetheless silently swallows and continues.
  Note: the 2026-07-09 remediation commit touched this same file but at different lines
  (`:360-372`, `set_scale`/attribute-attach try/excepts) — this hot-path hook block was not part
  of that pass. Impact: medium (per-step import overhead when IPAdapter is enabled; transient
  failures invisible in production). Effort: small — hoist the import, add debug logging.
- **Pointless `try/except Exception: pass` wrapping a single `logger.info()` call**
  (`wrapper.py:2103-2106`) — dead defensive code. Impact: low. Effort: small — delete it.
- **Duplicated latent-cache/decode/output-buffer logic between `__call__` and `txt2img`(_sd_turbo)**
  (`pipeline.py:1272-1285` vs `1367-1394`) — the `_image_decode_buf` lazy-allocate-then-`.copy_()`
  pattern is copy-pasted near-verbatim across 2-3 entry points; any fix to it (e.g. quick win #5)
  must be applied in every copy or silently regresses one call path. Impact: medium. Effort:
  small-medium — extract a shared `_decode_and_cache()` helper.
- **`unet_step` mixes five responsibilities in ~220 lines**: CFG buffer prep, SDXL conditioning
  cache, generic hook dispatch, ControlNet/IPAdapter kwarg extraction, TRT-vs-PyTorch calling
  convention (`pipeline.py:820-1041`). This is the exact span identified as the ~13.6ms/frame
  profiling blind spot (quick win #1) — splitting it into `_prepare_cfg_batch()`,
  `_get_sdxl_conditioning()`, `_dispatch_unet_hooks()`, `_call_unet()` would both fix the
  maintainability issue and naturally create the instrumentation boundaries needed to localize
  that cost. Impact: medium (maintainability) / high (unlocks quick win #1). Effort: large.
- **Parameter sprawl**: `_load_model` (30 params), `__init__` (50+ params) — plain
  positional/keyword lists rather than a config object; makes the god-function split above harder
  until addressed. Impact: medium. Effort: large — introduce a `StreamDiffusionConfig` dataclass,
  can be done incrementally alongside the `_load_model` split.
- **Minor/cosmetic**: `__call__`'s `x` parameter is typed without `Optional` despite a `None`
  default (`pipeline.py:1191`); `_get_scheduler_scalings` has no type hints (`pipeline.py:736`).
  Impact: low. Effort: small.

---

## Follow-ups not completed by this audit

1. GPU performance-counter access is now enabled, but a naive `ncu --set basic` pass over the full
   benchmark loop does not terminate in practical time because of CUDA-graph replay cost (see
   Limitations above). To get SOL%/occupancy for the FP8 GEMM kernels and attribute the 16.5%
   FP16 cutlass kernel to a specific engine, retry `ncu` with the workload scoped down — e.g.
   disable CUDA-graph capture for the profiled process (`use_cuda_graph=False` on the engine
   wrappers, if exposed as a config knob) so `ncu` sees ungraphed kernel launches, and/or bound
   the pass tightly with `--launch-skip N --launch-count <10-20>` targeting only the steady-state
   frames, and/or `--kernel-name regex:sm89_xmma_gemm|cutlass` to profile just the GEMM kernels of
   interest instead of the entire per-frame kernel set.
2. Add `profiler.region()` spans inside `unet_step` around CFG-buffer prep, hook dispatch, and
   conditioning lookup to localize the ~13.6ms/frame currently unattributed (quick win #1) —
   this is the highest-value next step and doesn't require any of the above.
3. Correlate H2D memcpy timestamps against `prepare()` vs. per-frame windows to confirm the 2.17s
   H2D total is a one-time load cost, not a steady-state one.
4. Re-run `profile_nsys.py` with `--cn-scale > 0` to isolate ControlNet's actual per-frame compute
   contribution (this run had `--cn-scale 0.0`, the script's default).

---

## Verification

- Repo `git status` shows only this new file as untracked; no source under `src/streamdiffusion/`
  was modified.
- Profiling artifacts referenced above (`profiles/*.txt`, `profiles/*.nsys-rep`,
  `profiler_logs/*_stats.json`) were generated by the repo's own
  `scripts/profiling/profile_nsys.py`/`nsys stats`, not hand-authored.
- Four citations (`utilities.py:1118-1125`, `pipeline.py:402-403`,
  `controlnet_models.py:68-73`, `wrapper.py:1264`) were re-read directly against source and match
  the findings exactly.

### MCP cross-check (2026-07-10, post-publication)

A second, independent verification pass used the repo's semantic code-search index
(`D:\dev\SDTD_032_dev`, 4707 chunks, `last_indexed 2026-07-10T08:16`Z — after the
`5a44f34`/`6718511` commits above, so the index reflects the exact HEAD this report was written
against). Method: `search_code` (k=7) to locate each claim's region, then exact `Grep`/`Read` to
pin the specific cited line(s). ~20 of the ~30 path:line citations in this report were checked,
covering all 12 quick wins plus a sample of the general-findings section.

**Result: no hallucinated citations.** Every checked claim resolves to real code at (or within a
line or two of) its cited location, including the highest-leverage ones: the `unet_step` god-method
(`pipeline.py:826-1041`, cyclomatic complexity 52 — corroborates quick win #1/#12's "unattributed
~13.6ms" and "220-line, five-responsibility function" characterizations independently of the
profiler data); `Engine.refit()` (`utilities.py:565-665`); the graph-replay `set_tensor_address`
loop (`utilities.py:1118-1125`); the single shared `cuda.Stream()` (`wrapper.py:2263`); and the
UNet-vs-ControlNet shape-floor mismatch (`models.py:160` = 256 vs. ControlNet's 384).

Two findings check out as understated, not overstated:
- **Quick win #9 (stringly-typed OOM):** `wrapper.py:2345` and `:2403` do match `"out of memory"
  in error_msg` as described — but the same file already catches the typed
  `torch.cuda.OutOfMemoryError` correctly at `wrapper.py:2029`. The typed exception is known to the
  codebase, making the two string-matched sites more clearly a regression/inconsistency than an
  unavoidable gap.
- **TensorRT finding #8 (FP8 sanity-gate comment mismatch):** confirmed exactly — `builder.py:307`
  comment reads "< 100 means quantization is inactive" while `builder.py:323` gates on `_qdq < 500`.

One characterization was not independently confirmed at the same precision as the rest: the
per-frame *invocation frequency* claimed for the IPAdapter hot-path hook
(`modules/ipadapter_module.py:434-521`) — the function-local imports at `:462`/`:479` are confirmed
present, but this pass did not separately verify the hook fires on every denoising step versus,
e.g., once per `generate()` call.

Also independently re-confirmed via direct grep of `src/streamdiffusion/`: zero occurrences of
`channels_last`/`memory_format` repo-wide, matching the "absent repo-wide" claim in the PyTorch
hot-path section exactly.
