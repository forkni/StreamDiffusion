# ControlNet CUDA IPC — TRT graph-capture conflict: fix + verification

> **RESOLVED 2026-05-18** — v4 fix applied and verified (cold-start with CN scale=0.577).
> Predecessor: `_plans/2026-05-17_controlnet-ipc-emitter-fix.md` (hypothesis A confirmed, v1 fix failed, v2/v3 partial, v4 final).

## Root cause

TRT's internal `genericReformat::copyPackedRunKernel` — invoked at the CN engine's input boundary to reformat the `controlnet_cond` tensor — submits work to the legacy/NULL CUDA stream during `execute_async_v3`. When the CN engine's polygraphy stream is in CUDA-graph capture mode (`cudaStreamBeginCapture … cudaStreamEndCapture`), that legacy-stream submission violates the capture rules:

> `operation would make the legacy stream depend on a capturing blocking stream`
> `cudaErrorStreamCaptureInvalidated (901)`

The polygraphy `Stream` class (venv `polygraphy/cuda/cuda.py:111`) is created via `cudaStreamCreate` with no flags → **blocking** by default → implicitly synchronizes with legacy. Any GPU op submitted to legacy during the capture window — whether from user code or TRT internals — invalidates the capture.

## Why earlier fixes failed

| Fix | What it did | Why it failed |
|---|---|---|
| v1 | dedicated non-blocking import stream + `wait_stream` bridge | `wait_stream` re-coupled legacy to the pending IPC event — same 901 |
| v2 | `get_frame()` with no `stream=` arg → CPU `cudaEventQuery` poll | Fixes warm-activation (OSC enable mid-stream). Fails cold-start: IPC tensor transforms still queue on legacy before capture — but that's not the real problem |
| Stage A | `CUDALINK_USE_GRAPHS=0` | Disproved; exporter graphs are irrelevant |
| v3 | `torch.cuda.current_stream().synchronize()` before `cudaStreamBeginCapture` | Drains legacy pre-capture. Fails because `genericReformat::copyPackedRunKernel` runs **inside** the capture window — pre-capture drain cannot prevent it |

## Fix (v4)

**`wrapper.py:2208` — `use_cuda_graph=False` for ControlNet engines.**

`use_cuda_graph=True` was hard-coded when constructing every CN TRT engine, regardless of input tensor format or graph-capture compatibility. Setting it to `False` keeps the CN engine in TRT-accelerated mode but skips the CUDA-graph wrapping:

- No `cudaStreamBeginCapture` is ever called on the CN engine stream.
- `genericReformat::copyPackedRunKernel`'s legacy-stream submission is harmless.
- CN inference retains all TRT kernel/tactic optimizations.
- Cost: CN loses the WDDM batch-submission savings (~hundreds of µs per forward on Windows WDDM). Measured impact: steady-state FPS ≈ 18-25 vs 19-28 with graph capture.

### Changes committed

| File | Change |
|---|---|
| `src/streamdiffusion/wrapper.py:2208` | `use_cuda_graph=True` → `use_cuda_graph=False` + inline comment |
| `src/streamdiffusion/acceleration/tensorrt/utilities.py:1018-1022` | Defensive: `torch.cuda.current_stream().synchronize()` before `cudaStreamBeginCapture`, gated on first capture per engine. Addresses the broader polygraphy-blocking-stream structural issue for future TRT engines. |
| `src/streamdiffusion/_compat/cuda_ipc/cuda_ipc_exporter.py:593` | `mode=0 → mode=1` (ThreadLocal capture hardening) — committed `07045be` |
| `src/streamdiffusion/_compat/cuda_ipc/cuda_graphs.py:46-47` | Docstring correction — committed `07045be` |
| `Scripts/StreamDiffusionTD__Text__StreamDiffusionExt__td.py` | YAML emitter emits `use_cuda_ipc_controlnet` + `cuda_ipc_control_shm_name` — committed `07045be` |
| `StreamDiffusionTD/td_manager.py` | v2 runtime fix: `get_frame()` with no `stream=` arg (gitignored; live via `Scripts/` sync) |

## Verification (live, 2026-05-18)

Cold-start `.toe` with `controlnet_scale: 0.577`, `use_cuda_ipc_controlnet: true`:
- `CUDA IPC control ready (zero-copy GPU): shm=StreamDiffusionTD_512-512_control_ipc` ✓
- No `[E] IExecutionContext::enqueueV3`, no `901`, no "legacy stream depend on capturing blocking stream" ✓
- CN scale applies immediately from frame 1 ✓
- Steady-state FPS sustained ✓

## Deferred follow-up

Investigate whether the CN engine's `controlnet_cond` input tensor can be produced in a format that avoids the `genericReformat` boundary (explicit `TensorIOFormat` constraints at build time, or providing the tensor already in CHW float32 on the engine stream). If so, `use_cuda_graph=True` for CN could be safely re-enabled.
