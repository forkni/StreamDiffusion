# ControlNet CUDA IPC — CUDA Graph Capture Conflict (session 2026-05-17)

> **RESOLVED 2026-05-18** — Hypothesis A confirmed. Fix applied and committed. See `_plans/2026-05-18_controlnet-ipc-stream-capture-fix.md`.

> Continuation of `_plans/2026-05-17_controlnet-zero-copy.md`. Emitter fixed so activation survives stream restart. New error class observed: TRT CN engine fails with `cudaErrorStreamCaptureInvalidated (901)` when IPC import runs inside the graph-capture window.

## 🟡 Session state (2026-05-17 end of session)

- ✅ Patches 1-5 to `StreamDiffusionTD/td_manager.py` intact
- ✅ Emitter patch applied to `Scripts/StreamDiffusionTD__Text__StreamDiffusionExt__td.py` (CN block after `use_cuda_ipc_input` + `cuda_ipc_control_shm_name` inside `td_settings`)
- ✅ Activation marker confirmed: `CUDA IPC control ready (zero-copy GPU): shm=StreamDiffusionTD_512-512_control_ipc`
- ✅ CN importer auto-detected `(512, 512, 4) uint8` — correct for TD canny TOP
- ❌ TRT CN engine forward fails — `cudaErrorStreamCaptureInvalidated (901)`
- ❌ Emitter patch and td_config.yaml NOT committed (waiting on error resolution)

## Error (23:42:28-29)

```
[E] IExecutionContext::enqueueV3: Error Code 1: Myelin (Platform Cuda error)

streamdiffusion.modules.controlnet_module - ERROR - controlnet forward failed:
  CUDA ERROR: cudaErrorStreamCaptureInvalidated (901)
  call_summary: cond_shape=(2, 77, 2048), img_shape=(2, 3, 512, 512), scale=0.6, is_sdxl=True, is_trt=True

Traceback:
  controlnet_module.py:488  _unet_hook: down_samples, mid_sample = cn(...)
  controlnet_engine.py:135  __call__: outputs = self.engine.infer(...)
  utilities.py:1028         infer: self.graph = CUASSERT(cudart.cudaStreamEndCapture(stream.ptr))
RuntimeError: CUDA ERROR: cudaErrorStreamCaptureInvalidated (901)

TouchDesignerManager - ERROR - Error updating parameters:
  CUDA error: operation would make the legacy stream depend on a capturing blocking stream
```

## Root-cause hypothesis

The TRT CN engine captures a CUDA graph on its own stream (`cudaStreamBeginCapture` … `cudaStreamEndCapture` at `utilities.py:1028`). Our `_get_control_frame_cuda_ipc()` calls:

```python
gpu_frame = self._cuda_ipc_control_importer.get_frame(stream=torch.cuda.current_stream())
```

`get_frame()` issues `cudaStreamWaitEvent` against the IPC slot's event. This touches the stream **during or adjacent to TRT's capture window**, which:

- Either drags the legacy/null stream into a dependency with the capturing stream (hypothesis A)
- Or records an event on the IPC stream that the capturing stream can't reference (hypothesis B)
- Or invalidates the capture from a previous call, and `cudaStreamEndCapture` returns 901 (hypothesis C)

The **input** IPC importer uses the same code path and never errors — suggesting timing is the differentiator. Input is fetched before any TRT capture starts; CN frame is fetched after `update_control_image()` and inside the hook that triggers the CN engine capture.

## Files to read at next session start

| Order | File | Location | Why |
|---|---|---|---|
| 1 | `cuda_ipc_importer.py` | `src/streamdiffusion/_compat/cuda_ipc/` | `get_frame()` stream-wait implementation; any `cudaStreamIsCapturing` guard |
| 2 | `utilities.py` | `src/streamdiffusion/acceleration/tensorrt/` | Lines 1000-1035: `infer()` capture begin/end, which stream |
| 3 | `controlnet_engine.py` | `src/streamdiffusion/acceleration/tensorrt/runtime_engines/` | Lines 120-140: when capture begins relative to input setup |
| 4 | `controlnet_module.py` | `src/streamdiffusion/modules/` | Lines 470-500: `_unet_hook` — timing of CN forward vs control-image update |
| 5 | `td_manager.py` | `StreamDiffusionTD/` | Lines 875-921: `_process_controlnet_frame` — call ordering |

## Candidate fixes (verify hypothesis before choosing)

- **a) Dedicated import stream** — pass a non-`current_stream()` argument to `get_frame()`, one that is never captured. Sync to engine stream once after. Low risk if importer signature supports it.
- **b) Capture-mode guard** — before `cudaStreamWaitEvent`, check `cudaStreamIsCapturing(stream)`. If capturing, use `cudaEventWaitExternal` flag or wait on a side-channel stream and pass result through an explicit event.
- **c) Reorder fetch before capture window** — pull CN frame at the top of the per-frame loop (before the diffusion step), cache the tensor, hand it to the orchestrator. The `process_tensor` branch already accepts a pre-fetched CUDA tensor.
- **d) Disable CUDA graph capture for CN engine only** — `CUDALINK_USE_GRAPHS=0` or per-engine flag in engine config. Temporary workaround; measure perf cost.

Options (a) and (c) are the cleanest structural fixes.

## Quick-revert if CN is needed immediately

Set `Usecudaipccontrolnet` TD COMP par to `False` (if par exists on the COMP), or comment out the two emitter lines:

```python
# yaml_content += f'use_cuda_ipc_controlnet: {str(use_ipc_controlnet).lower()}\n'
# yaml_content += f"  cuda_ipc_control_shm_name: '{stream_name}_control_ipc'\n"
```

Note: reverting to legacy path also requires re-adding a legacy CN SHM Out TOP in the .toe (was removed when the CUDA-Link Sender was added).

## Commit (deferred)

After the stream-capture conflict is resolved and live verification passes:

```powershell
./scripts/git/commit_enhanced.sh --no-venv "feat: emit ControlNet CUDA IPC activation keys in stream-start YAML"
```

Branch: `feat/cuda-ipc-output`, PR target: `SDTD_031_dev`.
