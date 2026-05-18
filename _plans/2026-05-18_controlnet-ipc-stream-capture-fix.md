# ControlNet CUDA IPC — TRT graph-capture conflict: fix + verification

> **RESOLVED 2026-05-18** — v2 fix applied and live-verified.
> Predecessor: `_plans/2026-05-17_controlnet-ipc-emitter-fix.md` (hypothesis A confirmed, v1 fix failed, v2 succeeded).

## Root cause

`CUDAIPCImporter.get_frame(stream=<any GPU stream>)` issues `cudaStreamWaitEvent` against the producer's IPC event. That GPU-side wait remains pending until the TD-side exporter records its event (next render tick). When `get_frame()` returns immediately, the supplied stream has unresolved pending GPU work.

Any stream that carries this pending wait — directly or via `wait_stream()` — propagates the dependency to PyTorch's legacy/NULL stream (handle `0x0`). Shortly after, the TRT ControlNet engine calls `cudaStreamBeginCapture(engine_stream, ThreadLocal)` on its **blocking** polygraphy stream. Blocking streams implicitly synchronise with the legacy stream. Beginning a CUDA graph capture on a blocking stream while the legacy stream has pending GPU work returns `cudaErrorStreamCaptureInvalidated (901)`.

## Why v1 failed

v1 routed `get_frame()` through a dedicated non-blocking stream (`_ipc_import_stream`) and bridged back via `current_stream().wait_stream(ipc_import_stream)`. The `wait_stream` call recorded a new event on `ipc_import_stream` queued **behind** the still-pending `wait_event`, then issued a second `cudaStreamWaitEvent(legacy, new_event)`. The legacy stream still had pending GPU work — same coupling, same 901.

## Fix (v2)

**Drop the GPU-side wait entirely.** Call `get_frame()` with no `stream=` argument. This routes through `_wait_for_slot()` (`cuda_ipc_importer.py:852`) which uses CPU-side `cudaEventQuery` polling — no GPU stream involvement, no pending state. When `get_frame()` returns, the producer event has fired (CPU-confirmed) and the data is coherent in GPU memory. The TRT capture proceeds with zero GPU pending state on the legacy stream.

`CUDALINK_EXPORT_SYNC=1` (default) means the producer CPU-blocks after `record_event`, so by the time the consumer polls, the event is already signalled — `query_event()` returns True on the first call, making v2's CPU sync essentially free at steady state.

### Changes

| File | Change |
|---|---|
| `StreamDiffusionTD/td_manager.py` | Removed `_ipc_import_stream` field, `_get_ipc_import_stream()` helper, and `wait_stream` bridges. Both `_get_input_frame_cuda_ipc` and `_get_control_frame_cuda_ipc` call `get_frame()` with no `stream=` argument. Lives in `dotsimulate/StreamDiffusionTD` repo (gitignored here); runtime-active via `Scripts/` sync. |
| `src/streamdiffusion/_compat/cuda_ipc/cuda_ipc_exporter.py:593` | `mode=0 → mode=1` (ThreadLocal) — defensive multi-engine hardening. Committed in this repo. |
| `src/streamdiffusion/_compat/cuda_ipc/cuda_graphs.py:46-47` | Docstring correction (ThreadLocal preferred in multi-engine processes). Committed in this repo. |
| `D:\dev\SD_3_0_1\test_Install_dev\StreamDiffusion\Scripts\StreamDiffusionTD__Text__StreamDiffusionExt__td.py` | YAML emitter emits `use_cuda_ipc_controlnet` + `cuda_ipc_control_shm_name` (applied 2026-05-17, outside this repo). |

## Verification (live, 2026-05-18)

**Python side (`00:42:02` — Uptime `03:03`):**
- `CUDA IPC control ready (zero-copy GPU): shm=StreamDiffusionTD_512-512_control_ipc`
- No `[E] IExecutionContext::enqueueV3`, no `901`, no "legacy stream depend on capturing blocking stream"
- `OSCHandler: ControlNet enabled via OSC` → `_update_controlnet_config: scale 0.0 → 0.246 → 0.577`
- Steady-state FPS 19-28 sustained over 3+ minutes

**TouchDesigner side (Receiver consuming SD output IPC):**
- All 3 slots opened, `event=YES`
- `[DIAG] import_frame #1-5: stream_wait=0.03ms copyCUDAMemory=0.08-0.10ms` — full round-trip healthy
- Activation barrier cycled through deactivate/reactivate cleanly
