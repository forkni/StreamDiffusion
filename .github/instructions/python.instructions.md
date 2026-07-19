---
applyTo: "**/*.py"
---

# Python review guidance — StreamDiffusion (real-time img2img for TouchDesigner)

This is a real-time diffusion pipeline running per-frame inside a TouchDesigner render loop, accelerated
with CUDA/TensorRT. Prioritize hot-path correctness and performance over style nitpicks — ruff and pyrefly
already enforce style (see `pyproject.toml`); don't duplicate their findings.

## Flag these

- **Host↔device syncs in per-frame/per-step code paths**: `.item()`, `.cpu()`, `.numpy()`, or
  `print(tensor)` inside anything that runs every frame or every diffusion step (pipeline step loops,
  `wrapper.py` inference calls, TensorRT runtime engines). Each one stalls the CUDA stream.
- **Hard-coded device/dtype**: `.cuda()`, `"cuda:0"`, or a hard-coded dtype instead of respecting the
  configured `torch.device`/dtype. This breaks multi-GPU and CPU-fallback paths.
- **Render-loop-breaking error paths**: any new code on the inference/streaming path that can raise instead
  of degrading gracefully — this pipeline is embedded in a TouchDesigner `.toe` and an uncaught exception
  can crash the host application, not just a request.
- **Bare `raise` inside an `except` that swallows the original cause**: re-raises must use `raise ... from e`
  so the original traceback survives (this repo tightened this recently — see the exception-cause-chaining
  history in `pipeline.py`/`wrapper.py`).
- **New public surface without reuse of existing helpers** in `src/streamdiffusion/utils/` (e.g.
  `diagnostics.py` for error-report generation, `image_utils.py`, `pip_utils.py`) — check before adding a
  parallel implementation.

## Do NOT flag (intentional, repo-wide decisions — see `pyproject.toml` `[tool.ruff.lint]`)

- `typing.Dict` / `typing.List` / `typing.Optional` / `typing.Union` instead of `dict` / `list` / `X | None`
  — kept deliberately (UP006/UP007/UP035/UP045 are ignored repo-wide; no PEP 585/604 rewrite suggestions).
- Single-letter variable names in math/diffusion formulas (`E741` ignored — common and expected here).
- Lines over ~79-100 chars — formatter-controlled; `E501` is off (line-length is 119).
- `torch.device`, `fastapi.Depends`, `fastapi.File` used as function/dataclass default arguments — these are
  explicitly whitelisted as immutable-safe defaults (`extend-immutable-calls` in `pyproject.toml`); don't
  suggest `field(default_factory=...)` or a sentinel workaround for these three.
- Import ordering/grouping nits already covered by ruff's `I` (isort) rules with
  `known-first-party = ["streamdiffusion"]`.
