# ControlNet zero-copy CUDA-IPC consumer + P3 Canny hardening + plan-doc fix

## Context

**Why this work exists.** Two threads converged:

1. **Verification of the CUDA perf plan** (`docs/plans/2026-05-24-cuda-perf-plan.md`, P1–P7): all
   seven items are implemented in code, but with caveats — `td_manager.py` is **untracked by git**
   (so P4/P5/P6 are not reviewable and the f631c90 commit message overstates what landed), the plan
   doc's file paths/line numbers are **stale** (it cites `Scripts/streamdiffusionTD__Text__td_manager__td.py`
   and old line numbers), and P3 (GPU Canny) carries two latent correctness bugs.

2. **ControlNet regression** — "no conditioning effect" (runs without error, output ignores the
   control input), using the **passthrough** preprocessor, "worked before, broke recently."

**Root cause of the CN regression (confirmed by reading the live code, high confidence).**
ControlNet conditioning never reaches the UNet because the control image is never delivered to the
Python side:

- The TD ControlNet sender COMP `shmem_out_cn` publishes the control image over **CPU shared memory**
  only — its Execute DAT calls `SharedMemEXT.sendData` (`Scripts/shmem_out_cn__Execute__execute__td.py:32`),
  and unlike `shmem`/`shmem_out` it has **no `cuda_ipc_parexec` exporter**. So the config keys
  `use_cuda_ipc_controlnet: true` and `cuda_ipc_control_shm_name: '<stream>_control_ipc'`
  (emitted by `StreamDiffusionExt__td.py:3776,3794`, present in the live `td_config.yaml:84,94`) are
  **dead** — nothing produces or consumes that IPC channel.
- On the Python side, `_process_controlnet_frame` read only the CPU-SHM `control_memory`
  (name `input_mem_name + '-cn'`) and bailed permanently when it was `None`, with **no lazy-reconnect**
  — in contrast to `_process_ipadapter_frame`, which reconnects every frame. When the input path
  moved to CUDA IPC (cuda-link v1.5.1, commits b9130e2 / f631c90), COMP activation timing shifted
  so the `-cn` buffer no longer exists at `_initialize_memory_interfaces` time, and it was never
  retried. Net effect: `controlnet_images[0]` stayed `None` → the gate in `controlnet_module.py`
  (`if cn is not None and img is not None and scale > 0`) silently dropped the residuals. No error,
  no conditioning — exactly the symptom.

**Decision (user).** ControlNet should run over the **CUDA-IPC zero-copy path**, mirroring the input
importer. The fix is on the **Python side** (`td_manager`); the user owns the TD `.toe` wiring of
`shmem_out_cn` as an IPC Sender. The Python consumer degrades gracefully (CPU-SHM fallback with
lazy-reconnect) so it is safe regardless of TD-side readiness.

**Feasibility confirmed.** The preprocessing orchestrator already accepts a CUDA tensor as the control
image and routes it through the "tensor" variant (`preprocessing_orchestrator.py:716-720`), and
`BasePreprocessor.validate_tensor_input` (`base.py:107-137`) accepts `(1,3,H,W)`/`(3,H,W)`/`(H,W,3)`,
moves to device/dtype, and **preserves `[0,1]`** (divides by 255 only if `max > 1`). Passthrough's
`_process_tensor_core` is a no-op (`passthrough.py:43-47`). So a `[0,1]` CUDA tensor flows end-to-end
with zero CPU round-trip.

---

## Implementation (completed 2026-05-24)

### Part A — ControlNet CUDA-IPC zero-copy consumer

Both td_manager copies edited in lockstep:
- `StreamDiffusion/StreamDiffusionTD/td_manager.py` — runtime target (untracked).
- `Scripts/streamdiffusionTD__Text__td_manager__td.py` — canonical TD Text-DAT source.

Changes inside `TouchDesignerManager`:

1. **State vars** (after `:89`): `self.ipc_control_importer=None`,
   `self._pending_ipc_control_name=None`, `self._ipc_control_last_retry=0.0`,
   `self._ipc_control_connected_logged=False`. Reuses `self._ipc_importer_cls`.

2. **Init** in `_initialize_memory_interfaces`: reads `cuda_ipc_control_shm_name` from
   `td_settings`; gated on `use_cuda_ipc_controlnet AND use_controlnet`; probe + construct via
   `_try_construct_ipc_control_importer`. CPU `control_memory` fallback retained.

3. **`_try_construct_ipc_control_importer()`**: sibling of `_try_construct_ipc_importer`,
   assigns `self.ipc_control_importer` on success.

4. **`_send_back_processed_controlnet()`**: extracted from inline duplication; called by both
   IPC and CPU-SHM branches.

5. **`_process_controlnet_frame` restructured**:
   - IPC path: throttled lazy-reconnect (1s), `get_frame()` → GPU tensor → `(1,3,H,W)` →
     **`[0,1]` float16** (NOT `[-1,1]` like the input path) → `update_control_image`.
   - CPU-SHM fallback: lazy-reconnect (mirrors IPAdapter pattern) → numpy uint8/255 → same.

6. **Cleanup**: `ipc_control_importer.cleanup()` added alongside input importer cleanup.

`td_main.py` — no change needed.

**TD-side prerequisite (user-owned):** wire `shmem_out_cn` as a CUDA-IPC Sender publishing to
`<stream>_control_ipc` (add `CUDAIPCExtension` + `cuda_ipc_parexec` ParExecute DAT, mirroring
`shmem_out`; engine from pip `cuda_link` / `td_exporter/CUDAIPCExtension.py` canonical source). Until wired, the Python
consumer logs "waiting for TD Sender" and uses CPU-SHM fallback.

### Part B — P3 GPU Canny hardening (`canny.py`)

1. `mag = mag / (mag.amax() + 1e-7)` → `(mag / 4.0).clamp(0.0, 1.0)`: fixed per-frame max
   normalization; constant divisor (≈ max Sobel response for a full-contrast step edge in [0,1]
   input) keeps thresholds stable across frames.
2. `edges.unsqueeze(0).expand(3,-1,-1)` → `.repeat(3,1,1)`: fixed non-contiguous stride-0 view.

### Part C — Perf plan doc update (`docs/plans/2026-05-24-cuda-perf-plan.md`)

Added "Post-implementation corrections" section documenting: file path corrections (Scripts/ is
canonical source), td_manager untracked status, dead `use_cuda_ipc_controlnet` flag, and the
implemented ControlNet consumer.

---

## Verification

- **Smoke**: SD-Turbo / SDXL-Turbo, 512×512, 2-step, t_index=[32,45], seed=2,
  passthrough CN, `conditioning_scale≈0.44`, `enabled: true`.
- **CN restored (IPC active)**: expect log "CUDA IPC control connected" + visible conditioning
  effect; `controlnet_module.controlnet_images[0] is not None` mid-stream.
- **Graceful fallback (IPC down)**: CPU-SHM lazy-reconnect engages, CN still conditions.
- **Zero-copy check**: no `.cpu()`/`.numpy()` in the per-frame IPC CN branch.
- **P3**: stable edge density across frames; contiguous output tensor.
