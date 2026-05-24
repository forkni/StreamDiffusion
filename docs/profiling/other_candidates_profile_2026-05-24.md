# Eager-op candidates beyond glue — profiling — 2026-05-24

## Setup

- **GPU:** RTX 4090 (Ada SM 8.9, WDDM)
- **Config:** `StreamDiffusionTD/td_config.yaml` — sdxl-turbo, 512×512, fp8, tensorrt, img2img,
  2-step LCM (`t_index_list: [15,28]`), ControlNet canny, CUDA IPC output
- **Method:** same as `glue_ops_profile_2026-05-24.md` — CUDA-event timing via `profiler.region()`,
  no nsys required, 24 profiled inferences, 10 nsys-gated steps
- **New regions added:** `trt.input_staging`, `sched.step_batch`, `sched.rebuild`

## Results (P50 = steady-state)

| Region | Count | Mean | P50 | P95 | Min | Max | Total |
|---|---|---:|---:|---:|---:|---:|---:|
| `predict_x0_batch` | 24 | 40.02 ms | 28.25 ms | 28.89 ms | 27.91 ms | 310.62 ms | 960.6 ms |
| `unet_step` | 24 | 39.98 ms | 28.21 ms | 28.86 ms | 27.88 ms | 310.51 ms | 959.6 ms |
| `trt_infer` | 72 | 11.22 ms | 1.10 ms | 24.89 ms | 0.78 ms | 172.47 ms | 807.9 ms |
| **`trt.input_staging`** | 72 | 0.91 ms | **0.04 ms** | 2.34 ms | 0.00 ms | 8.66 ms | 65.2 ms |
| `encode_image` | 24 | 1.86 ms | 0.93 ms | 9.59 ms | 0.87 ms | 14.18 ms | 44.7 ms |
| `decode_image` | 24 | 1.64 ms | 1.17 ms | 1.42 ms | 1.14 ms | 12.23 ms | 39.3 ms |
| `glue.ipc_pack_rgba` | 24 | 0.77 ms | 0.07 ms | 0.12 ms | 0.06 ms | 16.72 ms | 18.4 ms |
| **`sched.step_batch`** | 48 | 0.04 ms | **0.04 ms** | 0.08 ms | 0.03 ms | 0.09 ms | 2.0 ms |
| **`sched.rebuild`** | 24 | 0.01 ms | **0.01 ms** | 0.03 ms | 0.00 ms | 0.03 ms | 0.3 ms |

Wall-clock (10 nsys-gated steps): **30.2–30.8 ms/frame (~33 FPS), highly consistent.**

## Analysis

### Candidate A — TRT engine input-staging copies (`utilities.py:993-996`)

The copy loop that stages per-frame data into the engine's address-stable input buffers before CUDA
graph replay measures **P50 = 40 µs = 0.13% of frame time** — 38× below the 5% go/no-go gate.

Count is 72 over 24 inferences = 3 engine calls/inference (UNet + VAE encoder + VAE decoder; the
benchmark bypasses td_manager so the ControlNet engine sees no input).

**Important clarification on torch.profiler `aten::copy_` / `Memcpy DtoD` (2.85 ms/frame in the
profiler table):** the aggregate is dominated by CUDA-graph-internal ops — the CUDA graph captures
attention processor `_curr_key_buf.copy_`/`_curr_value_buf.copy_` calls from
`attention_processors.py:113-114` (once per attention layer per UNet forward = tens of copies), plus
TRT-internal transfer kernels. These are all INSIDE `trt_infer`. The pre-staging loop is a small
fraction of the `aten::copy_` total; our region correctly isolated it as 0.13% of frame.

### Candidate B — LCM scheduler elementwise chains (`pipeline.py`)

| Sub-region | Calls/frame | P50 | % of frame |
|---|---|---|---|
| `sched.step_batch` (×2/frame) | 2 | 40 µs × 2 = 80 µs | 0.27% |
| `sched.rebuild` | 1 | 10 µs | 0.03% |

The broadcast `(N,1,1,1)×(N,4,64,64)` math with fp32 up/downcast runs in **40–10 µs** per call.
Both far below the 5% gate.

`elementwise_kernel` / `Shape_cu` entries visible in the torch.profiler table are similarly TRT-internal
(inside `trt_infer`), not these eager scheduler ops.

## Decision: NO GO (all candidates)

This completes the eager-op optimization audit for the current inference path. Summary of all
profiling passes:

| Candidate | P50 | % of frame | Gate | Decision |
|---|---|---|---|---|
| `glue.ipc_pack_rgba` (denorm→uint8→BGRA) | 80 µs | 0.27% | 5% | NO GO |
| `trt.input_staging` (per-engine copy loop) | 40 µs | 0.13% | 5% | NO GO |
| `sched.step_batch` (LCM elementwise ×2) | 40 µs/call | 0.27% | 5% | NO GO |
| `sched.rebuild` (buffer rebuild) | 10 µs | 0.03% | 5% | NO GO |

**The TRT UNet engine (~28 ms P50, ~93% of frame) is the only optimization surface.** Its internals
are already fused by TensorRT (fp8 quantization, CUDA graph, optimal kernel selection). No
eager-op optimization is actionable on this path at 512×512 / 2-step LCM / WDDM.

If FPS becomes a bottleneck, the next steps are:
- Profile the TRT engine itself with `ncu` on the UNet graph (per `scripts/profiling/profile_ncu.py`)
- Explore resolution or step-count tuning
- Evaluate TRT engine re-build with more aggressive optimization levels

## Artifacts

- CUDA-event stats JSON: `profiler_logs/sdtd_benchmark_20260524_122225_stats.json`
- Chrome trace: `profiler_logs/sdtd_benchmark_20260524_122225_trace.json`
- Prior glue pass: `docs/profiling/glue_ops_profile_2026-05-24.md`
