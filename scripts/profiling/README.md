# StreamDiffusion Nsight Profiling

## Quick Start

### nsys — GPU timeline (benchmark target, existing cached engine)

Pass `--config` to load the exact same wrapper kwargs as `td_main.py`, guaranteeing a cache
hit — no engine rebuild. The config at `StreamDiffusionTD/td_config.yaml` is the "Quality / FP16"
preset (`stabilityai/sdxl-turbo`, 512×512, fp16, img2img).

```bat
set NSYS="C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.3.2/target-windows-x64/nsys.exe"
%NSYS% profile --trace=cuda,nvtx,cublas --cuda-memory-usage=true ^
    -o profiles/sdtd_quality_fp16 --force-overwrite true ^
    .venv/Scripts/python scripts/profiling/profile_nsys.py --target benchmark ^
        --config StreamDiffusionTD/td_config.yaml

REM Open the report:
"C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.3.2/host-windows-x64/nsys-ui.exe" profiles/sdtd_quality_fp16.nsys-rep

%NSYS% stats --report nvtx_pushpop_trace   profiles/sdtd_quality_fp16.nsys-rep > nvtx_trace.txt
%NSYS% stats --report cuda_kern_exec_trace profiles/sdtd_quality_fp16.nsys-rep > kernel_trace.txt
```

Without `--config` the script falls back to inline defaults (`KBlueLeaf/kohaku-v2.1`) which will
trigger an engine build if no matching cache exists.

### nsys — GPU timeline (td_main production path)

```bat
set NSYS="C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.3.2/target-windows-x64/nsys.exe"

REM Wrap td_main.py directly; SDTD_NSYS_CAPTURE=1 fires start/stop at precise frame boundaries:
set GPU_PROFILER=1
set SDTD_NSYS_CAPTURE=1
set SDTD_NSYS_WARMUP_FRAMES=20
set SDTD_NSYS_CAPTURE_FRAMES=500
%NSYS% profile --trace=cuda,nvtx,cublas --capture-range cudaProfilerApi ^
    -o profiles/sdtd_td_main --force-overwrite true ^
    .venv/Scripts/python StreamDiffusionTD/td_main.py

REM Or let the launcher manage it (no nsys wrapping required for deferred stats):
.venv/Scripts/python scripts/profiling/profile_nsys.py --target td_main --warmup 20 --frames 500
```

### ncu — per-kernel metrics

```bat
REM Basic metrics (2-3× overhead):
.venv/Scripts/python scripts/profiling/profile_ncu.py --target benchmark --set basic

REM Roofline analysis:
.venv/Scripts/python scripts/profiling/profile_ncu.py --target benchmark --set roofline --launch-count 100

REM See the exact command without running:
.venv/Scripts/python scripts/profiling/profile_ncu.py --target benchmark --dry-run
```

---

## What's Instrumented

The following `profiler.region()` names appear in nsys NVTX rows and in the JSON stats file:

| Region | File | Description |
|---|---|---|
| `frame` | `examples/benchmark/single.py`, `wrapper.py` | Full per-frame round-trip |
| `encode_image` | `pipeline.py` | VAE encode: RGB → latent |
| `predict_x0_batch` | `pipeline.py` | Full diffusion denoising block |
| `unet_step` | `pipeline.py` | UNet forward (inside predict_x0_batch) |
| `scheduler_step` | `pipeline.py` | LCM/TCD scheduler step |
| `trt_infer` | `acceleration/tensorrt/utilities.py` | TRT engine execute_async_v3 / graph launch |
| `decode_image` | `pipeline.py` | VAE decode: latent → RGB |
| `d2h_sync` | `wrapper.py` | Device→Host DMA event sync (np output path) |

> **CUDA graph note:** `trt_infer` NVTX markers fire only at graph capture time (first 3 warmup
> frames), not on each replay. Set `GPU_PROFILER_NVTX=0` for events-only mode (graph-safe);
> CUDA-event timings in the JSON stats file are always accurate.

> **CUPTI subscriber note:** When running the benchmark target under nsys, `torch.profiler`
> may print `CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED` — this is benign. nsys and
> torch.profiler both register CUPTI subscribers; CUDA-event timing in `*_stats.json` and
> the nsys GPU timeline are both unaffected.

---

## Environment Variables

| Variable | Effect |
|---|---|
| `GPU_PROFILER=1` | Activate profiler (master switch). Auto-read by `configure()` in wrapper `__init__`. |
| `GPU_PROFILER_NVTX=0` | Disable NVTX ranges (safe with CUDA graphs); CUDA-event timing stays on. |
| `GPU_PROFILER_EVENTS=0` | Disable CUDA-event timing (NVTX only). |
| `STREAMDIFFUSION_PROFILE_TRT=1` | Activate existing TRT IProfiler (per-layer times; disables CUDA graphs). |
| `SDTD_NSYS_CAPTURE=1` | Enable deferred-capture handshake in `td_manager._streaming_loop`. |
| `SDTD_NSYS_WARMUP_FRAMES` | Frames before `cudaProfilerStart` (default: 20). |
| `SDTD_NSYS_CAPTURE_FRAMES` | Frames to capture after warmup (default: 500). |
| `SDTD_PROFILE_JSON` | Path for `profiler.export_stats()` on capture stop. |
| `SDTD_MODEL_ID` | Override model for `profile_nsys.py --target benchmark`. |
| `SDTD_ACCELERATION` | Override acceleration (default: `tensorrt`). |
| `SDTD_WIDTH` / `SDTD_HEIGHT` | Override resolution for `profile_nsys.py --target benchmark`. |
| `NSYS` | Override nsys.exe path (auto-discovered if unset). |
| `NCU` | Override ncu.exe path (auto-discovered if unset). |

---

## Output Convention

| Path | Content |
|---|---|
| `profiles/*.nsys-rep` | Nsight Systems reports — open in `nsys-ui.exe` |
| `profiler_logs/*_trace.json` | Chrome trace from `torch.profiler` — open in Perfetto / `chrome://tracing` |
| `profiler_logs/*_stats.json` | CUDA-event timing stats (mean/p50/p95/p99/min/max/total ms per region) |
| `profiler_logs/*_report.md` | Markdown timing table auto-rendered from stats JSON |
| `logs/ncu_*.ncu-rep` | Nsight Compute reports — open in Nsight Compute UI |
| `logs/ncu_*.csv` | Kernel details CSV (with `--csv` flag) |

All output directories are gitignored. Use the `.nsys-rep` / `.ncu-rep` files directly with the
Nsight UI, or share the `*_stats.json` for lightweight timing comparison.
