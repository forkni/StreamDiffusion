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

**Highest-ROI option: increase batch size** (throughput-only — adds input latency; see
verdict section below). `batch_size = denoising_steps_num × frame_buffer_size`
(`pipeline.py:100-101`); production runs batch=2 (`denoising_steps_num=2 ×
frame_buffer_size=1`). `frame_buffer_size=2` reaches batch=4 exactly (2×2), fitting the
engine's `max_batch=4` without rebuild. `frame_buffer_size=4` would require batch=8 and
a rebuild. But filling batch=4 requires feeding 2 input frames per call, which adds input
latency and requires loop + `img2img` rework in the TD path.

## What to do next

1. **Batch-size lever: deferred** — `frame_buffer_size=2` is the correct value to reach
   batch=4, but the live TD loop is 1-in-1-out (`td_manager.py:579/619/622`) and
   `img2img` accepts a single image. Exploiting this requires collecting 2 frames before
   inference + reworking the output path, and adds ~1 frame-interval of input latency.
   Only worthwhile for offline/throughput-oriented modes, not the live interactive stream.
   Full analysis in verdict section below.
2. **Nothing else for now** — the eager-op audit established all non-TRT paths are
   negligible. The only other lever is resolution (major scope change) or step count.

## Artifacts

- Report: `logs/ncu_fp8_gemm_roofline.ncu-rep` — open with Nsight Compute UI
- Reference: `logs/ncu_target_gemm_roofline.ncu-rep` — A100 fp16 baseline (different GPU)
- Prior pass: `docs/profiling/other_candidates_profile_2026-05-24.md`

## Verdict: why there is no latency fix (and what 99% GPU load means)

*Added 2026-05-24 after confirming the TD architecture and completing the full audit.*

### The 99% GPU load observation

During the live stream the GPU reports **~99% utilization** (Task Manager / nvidia-smi
"3D" / "CUDA" graph). This is **not** a contradiction of the wave-limited finding — it
is the other half of the picture.

| Metric | What it measures | Value |
|---|---|---|
| **Temporal utilization** (nvidia-smi) | Fraction of wall-clock time a kernel is running | ~99% |
| **Compute SOL** (ncu) | Of the math the SMs *could* do while a kernel runs, how much is realized | ~15% |
| **SM wave count** (ncu SOLBottleneck) | Full sweeps of all 128 SMs per kernel | 0.2–0.4 |

**99% temporal** means kernels run back-to-back — essentially zero idle gaps between them.
**15% compute SOL / 0.2–0.4 waves** means during each kernel, ~77–102 of 128 SMs are
idle *inside* the kernel.

Consequences for optimization:
- **No time-gap headroom** → any technique that reclaims gaps between kernels (stream
  overlap, launch-latency cuts, eager-op fusion) cannot help. This is exactly why all 4
  profiling candidates (glue, input-staging, scheduler chains, ncu audit = all NO GO)
  were negligible — their combined cost vanishes into rounding error even fused to zero.
- **Large idle-SM headroom *within* kernels** → batch size addresses this: extra frames'
  GEMM blocks fill the SMs sitting idle *during* the same kernel invocation, so 4 frames
  complete in roughly the wall-clock duration of today's single-frame kernel. That is the
  throughput leverage. It does not reduce latency; it increases throughput.

### The TD architectural constraint

The live streaming loop is strictly 1-in-1-out:

```
td_manager._streaming_loop  (td_manager.py:491)
    _get_input_frame()           # reads ONE frame   (line 579)
    wrapper.img2img(one_image)   # single-image call  (line 619)
    _send_output_frame(output)   # writes ONE frame   (line 622)
```

`wrapper.img2img` and `pipeline.__call__` accept a single image; there is no frame queue
or accumulator. The internal `batch_size` refers to the denoising-step batch for
`t_index_list`, not multiple input frames.

Using `frame_buffer_size=2` (batch=4) requires:
1. Collecting 2 input frames before each `img2img` call.
2. Reworking `img2img` and the IPC output path to accept and emit 2 frames per call.
3. **Accepting ~1 extra frame-interval of input-to-output latency** from buffering.

### Final verdict

| Goal | Is the batch-size lever the fix? |
|---|---|
| **Lower per-frame latency** (live interactive use) | **No.** The 28 ms UNet is the problem-size floor at 512×512 on the 4090. No code or config change addresses it. |
| **Higher frame throughput** (offline / slow-mo / multi-output) | **Yes — with rework.** `frame_buffer_size=2` (batch=4, no rebuild) gives ~2× throughput, costs ~1 frame of input latency, and requires the TD loop + `img2img` to support multi-frame batches. |

**For the live interactive img2img stream this pipeline is designed for: there is no fix.**
The 99% temporal load confirms no idle time remains; the wave-limited ncu profile confirms
the remaining waste is inside kernels and only accessible via frame batching. Batching is
a throughput-for-latency trade requiring architectural rework and is not pursued here.

Other levers for context: fewer denoising steps reduces total work but not per-kernel wave
count; lower resolution shrinks GEMMs further (worse occupancy); larger resolution adds 4×
work and 4× waves (near-roofline) but 4× latency.
