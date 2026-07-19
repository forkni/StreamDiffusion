---
applyTo: "src/streamdiffusion/acceleration/tensorrt/**"
---

# TensorRT acceleration review guidance

This directory builds, exports, and runs TensorRT engines (UNet, ControlNet, IP-Adapter, SDXL variants) for
the real-time pipeline. Correctness bugs here are usually silent-corruption bugs (wrong pixels/NaNs), not
crashes — review with that in mind.

## Flag these

- **Zero-copy / buffer-aliasing safety**: any place a device buffer's dtype, shape, or lifetime assumption
  changes — e.g. an engine binding that assumes a buffer outlives a call, or a reused output buffer whose
  shape no longer matches the bound tensor. This is the class of bug the `trt-zero-copy-audit-hardening`
  work targeted; scrutinize buffer ownership across `runtime_engines/unet_engine.py`,
  `runtime_engines/controlnet_engine.py`, and `engine_manager.py`.
- **FP8 / precision preflight gaps**: quantization or precision-selection logic (`fp8_quantize.py`,
  `param_schema.py`-driven preflight) that skips validating the target GPU/engine actually supports the
  requested precision before committing to it.
- **Engine (de)serialization mismatches**: loading a cached `.engine`/`.plan` file without verifying its
  shape, dtype, or builder-config version against the current request — a stale-engine bug surfaces as
  wrong output, not an exception.
- **Export-wrapper drift**: changes to `export_wrappers/*.py` that change a model's input/output signature
  without a matching update to the corresponding `models/*.py` binding shapes.

## Do NOT flag

- `from module import *` in `builder.py` and `runtime_engines/unet_engine.py` — this is intentional and
  explicitly per-file-ignored (`F403`/`F405`) in `pyproject.toml`; don't suggest converting to explicit
  imports in these two files.
- General Python style already covered by `python.instructions.md` / ruff — focus this file's reviews on
  TensorRT-specific correctness (buffers, precision, engine compatibility), not style.
