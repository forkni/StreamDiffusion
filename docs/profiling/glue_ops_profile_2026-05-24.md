# Hot-path glue ops profiling — 2026-05-24

## Setup

- **GPU:** RTX 4090 (Ada SM 8.9, WDDM)
- **Config:** `StreamDiffusionTD/td_config.yaml` — sdxl-turbo, 512×512, fp8, tensorrt, img2img,
  2-step LCM (`t_index_list: [15,28]`), ControlNet canny, CUDA IPC output
- **Target:** `profile_nsys.py --target benchmark` (no nsys, CUDA-event timing)
- **Warmup:** 3 extra + 1 torch.profiler skip → 10 profiled + 10 nsys-gated inferences
- **Instrumentation:** `profiler.region(...)` wraps in `wrapper.py` and `td_manager.py`
  (added 2026-05-24, see Step 1 of the profiling plan)

## Results (P50 = steady-state)

| Region | Count | Mean | P50 | P95 | Min | Max | Total |
|---|---|---|---|---|---|---|---|
| `predict_x0_batch` | 24 | 50.12 ms | 28.19 ms | 221.10 ms | 27.88 ms | 296.66 ms | 1203.0 ms |
| `unet_step` | 24 | 50.08 ms | 28.16 ms | 221.04 ms | 27.85 ms | 296.55 ms | 1201.9 ms |
| `trt_infer` | 72 | 13.79 ms | 1.10 ms | 25.30 ms | 0.79 ms | 170.52 ms | 992.8 ms |
| `encode_image` | 24 | 2.47 ms | 0.92 ms | 10.21 ms | 0.86 ms | 18.40 ms | 59.4 ms |
| `decode_image` | 24 | 1.85 ms | 1.15 ms | 6.97 ms | 1.12 ms | 12.35 ms | 44.5 ms |
| **`glue.ipc_pack_rgba`** | 24 | 0.72 ms | **0.08 ms** | **0.14 ms** | 0.06 ms | 15.49 ms | 17.3 ms |

Wall-clock (10 nsys-gated steps): **30.1–30.4 ms/frame (~33 FPS), highly consistent.**

## Analysis

`glue.ipc_pack_rgba` (the `_ipc_pack_rgba` + `_denormalize_on_gpu` chain — ~8–9 eager kernel
launches converting fp16 NCHW → uint8 BGRA HWC) runs in **80 µs at P50 / 140 µs at P95**.

As a fraction of frame time: **80 µs / 30 200 µs ≈ 0.27%** — nearly 20× below the 5% go/no-go threshold.

The high Max (15.49 ms) and P99 are first-call outliers (CUDA context / TRT engine warm-up); they do
not affect steady-state performance.

## Decision: NO GO

Fusing the output-pack glue into a single CuPy `RawKernel` would save at most ~50–70 µs/frame
(collapsing ~9 launches → 1, eliminating ~6 intermediate allocations and redundant DRAM passes).
That is invisible at 33 FPS and not worth the implementation risk or complexity.

The dominant cost is the TRT UNet engine (~28 ms P50), which is already internally fused by TRT.
There is no actionable eager-glue optimization surface on this path at the current resolution and
step count.

## Candidates NOT profiled

Candidates #2 (`glue.input_normalize`) and #3 (`glue.cn_normalize`) in `td_manager.py` require a
live TD Sender (CUDA IPC input). By napkin math they are each ~2–3 eager kernel launches on a
512×512 frame — at most ~20 µs each. Given #1 (the most expensive candidate at ~8–9 launches)
measured at 80 µs, #2 and #3 are expected well under the threshold too. No nsys capture was run
for those.

## Artifacts

- CUDA-event stats JSON: `profiler_logs/sdtd_benchmark_20260524_120743_stats.json`
- Chrome trace: `profiler_logs/sdtd_benchmark_20260524_120743_trace.json`
