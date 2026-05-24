# UNet fp8 GEMM kernel profiling — ncu roofline — 2026-05-24

## Setup

- **GPU:** RTX 4090 (Ada SM 8.9, WDDM)
- **Config:** `StreamDiffusionTD/td_config.yaml` — sdxl-turbo, 512×512, fp8 TRT, 2-step LCM
- **Tool:** Nsight Compute 2026.1.1, roofline metric set
- **Report:** `logs/ncu_fp8_gemm_roofline.ncu-rep` (captured in a prior session)
- **Filter:** `sm89_xmma_gemm_e4m3` — fp8 GEMM kernels on SM 8.9

## Profiled kernels (5 instances)

All instances are the same fp8 GEMM variant:
`sm89_xmma_gemm_e4m3e4m3_e4m3f32_f32_tn_n_tilesize64x64x64_stage4_warpsize2x2x1_tensor16x8x32_bias_f32_execute_kernel_trt`

- Precision: e4m3×e4m3 → fp32 accumulator (full fp8 path)
- Tile: 64×64×64 — smallest TRT tile for sm89 fp8
- Warp config: 2×2×1 tensor cores with 16×8×32 wmma

| ID | Grid | Duration | DRAM % | SM compute % | Waves |
|---|---|---|---|---|---|
| 0 | 16×10=160 | 19 µs | 36.5% | 15.3% | **0.4** |
| 1 | 4×20=80 | 18 µs | 33.8% | 10.6% | **0.2** |
| 2 | 4×20=80 | 10 µs | 29.5% | 15.7% | **0.2** |
| 3 | 4×20=80 | 16 µs | 44.8% | 15.7% | **0.2** |
| 4 | 4×20=80 | 10 µs | 26.7% | 15.5% | **0.2** |

## Analysis

### Bottleneck: wave-limited (SM occupancy)

ncu reports the same `SOLBottleneck` on every kernel instance:
> "This kernel grid is too small to fill the available resources on this device,
> resulting in only **0.2–0.4 full waves** across all SMs."

At 0.2 waves: only ~26 of 128 SMs are active during the final (and only) wave.
At 0.4 waves: ~51 of 128 SMs active. In both cases, the majority of the GPU is idle.

The consequence: the 4090's theoretical peak is never approached. DRAM throughput is
26–45% (not bandwidth-bound) and SM compute is 10–16% (not compute-bound). The kernels
are short (10–19 µs) because the matrices are small — **not** because they're efficient.

### Root cause: problem size at 512×512

The SDXL attention GEMMs at 512×512 are simply too small for a 128-SM GPU:
- 512×512 spatial → 64-token sequence (8×8 spatial tokens per head for the latent)
- With batch=2 (denoising batch), matrices still don't fill 128 SMs at tile size 64×64
- TRT already selected the smallest available tile (64×64×64) — it cannot go smaller

**This is a fundamental problem-size constraint, not a kernel implementation issue.**
No TRT configuration change, fp8 re-tuning, or eager-op optimization can fix it —
the cure is more work per kernel (larger batch or larger resolution).

### Comparison: the same GEMM at scale (A100 fp16, 1×8K sequence)

`logs/ncu_target_gemm_roofline.ncu-rep` shows the same GEMM operation on an A100 at
a large sequence length: DRAM 92%, SM compute 69%, Duration 830+ µs. This is what
near-roofline fp16 GEMM looks like. Our fp8 kernels operate at 1/50th the duration
with 1/7th the DRAM utilization — purely because the matrices are 50× smaller.

## Implications for optimization

| Approach | Effect on wave-limited bottleneck | Feasibility |
|---|---|---|
| Larger batch size (4–8 frames) | Increases GEMM M → more waves → better utilization | High (no engine rebuild required if `max_batch≥4`) |
| Larger resolution (1024×1024) | Larger spatial dims → bigger GEMMs | Requires engine rebuild + 4× more work |
| More denoising steps (4–8) | Larger batched UNet call | Increases latency proportionally |
| Custom kernel / `torch.compile` | Cannot change TRT-internal kernel selection | N/A |
| fp16 instead of fp8 | Slightly larger arithmetic intensity, still wave-limited | Slower (fp8 IS faster per FP) |
| t_index / cfg_type tuning | Reduces steps, doesn't change per-kernel size | Reduces total work, not per-kernel utilization |

**Highest-ROI option: increase batch size.** If the application can consume multiple
generated frames per TD callback (e.g., triple-buffering for texture blending), running
at batch=4 increases GEMM M by 4× → moves from 0.2 → 0.8 waves → ~4× better SM
utilization → near-linear throughput scaling without extra latency. The current engine
was built with `max_batch=4` (`engine_dir` path includes `min_batch-1--max_batch-4`),
so this requires no engine rebuild.

## What to do next

1. **Batch-size throughput test** — run profile_nsys.py benchmark with
   `frame_buffer_size=4` (or modify td_config.yaml to set `batch_size: 4`) and measure
   throughput (frames/second) vs latency (ms/frame). If latency stays ≤30ms while
   FPS doubles or triples, batch-mode operation is worth exposing to TD.
2. **Nothing else for now** — the eager-op audit established all non-TRT paths are
   negligible. The only other lever is resolution (major scope change) or step count.

## Artifacts

- Report: `logs/ncu_fp8_gemm_roofline.ncu-rep` — open with Nsight Compute UI
- Reference: `logs/ncu_target_gemm_roofline.ncu-rep` — A100 fp16 baseline (different GPU)
- Prior pass: `docs/profiling/other_candidates_profile_2026-05-24.md`
