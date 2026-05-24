# ControlNet output throttle fix — gate the per-frame preview send-back

## Context

After the ControlNet CUDA-IPC consumer landed (commit `2a9ed08`), ControlNet conditions
correctly — but the main output visibly throttles/stutters. The throttle is a side effect of
the fix, not a coincidence.

**Root cause.** Every frame, `_process_controlnet_frame` (both IPC branch at `:1002` and CPU
branch at `:1025`) calls `_send_back_processed_controlnet()`, which forwards the preprocessed
control image to `_send_processed_controlnet_frame()`. That method at **`td_manager.py:906`**
does:

```python
frame_np = processed_tensor.cpu().numpy()   # unpinned, BLOCKING D2H every frame
```

`.cpu()` on a non-pinned tensor implicitly synchronizes the CUDA stream, stalling the host in
series with diffusion every frame.

**Why it only appeared now.** Before the fix, `controlnet_images[0]` was always `None`, so the
guard at `td_manager.py:835` (`controlnet_images[0] is not None`) made the send-back a silent
no-op. Now that CN populates `controlnet_images[0]`, the blocking copy fires every frame.
`control_processed_memory` was always created (`:418-430`), so the copy always had a destination.

**Why the main output path was fine.** `_send_output_frame` (`:841`) uses the P4 pattern:
GPU-side uint8 + reused pinned buffer + `non_blocking=True` + one sync (`:861-865`). The CN
preview path never got that treatment — it was the naive blocking pattern.

**What the send-back is for.** Display-only: ships the preprocessed control image over
`control_processed_memory` shared memory so TD can show a preview. Not needed for diffusion
(the UNet reads `controlnet_images` directly). With **passthrough** the preview == input (fully
redundant).

---

## Implementation (completed 2026-05-24)

Both td_manager copies edited in lockstep:
- `StreamDiffusion/StreamDiffusionTD/td_manager.py` — runtime target (untracked).
- `Scripts/streamdiffusionTD__Text__td_manager__td.py` — canonical TD Text-DAT source.

### 1. Gate `_send_back_processed_controlnet` behind `send_controlnet_preview` (default OFF)

Added early return at the top of the method (`:826`), covering both call sites:

```python
if not self.config.get('send_controlnet_preview', False):
    return
```

Because the key is absent from `td_config.yaml` by default, `False` is returned immediately —
the stall is eliminated without requiring any YAML/TD-side change.

### 2. Skip the preview SHM allocation when disabled

Wrapped the `control_processed_memory` create/connect block (`:416-430`) in
`if self.config.get('send_controlnet_preview', False):`. When disabled,
`control_processed_memory` stays `None`, providing a second guard in
`_send_processed_controlnet_frame` (`:900`).

### 3. Emit `send_controlnet_preview` from the YAML emitter (default false)

Added to `Scripts/StreamDiffusionTD__Text__StreamDiffusionExt__td.py` (after
`use_cuda_ipc_controlnet` at `:3782`), mirroring the same pattern:

```python
send_cn_preview = False
try:
    send_cn_preview = bool(self.ownerComp.par.Sendcontrolnetpreview.eval())
except AttributeError:
    pass
yaml_content += 'send_controlnet_preview: {str(send_cn_preview).lower()}\n'
```

Emitted top-level (lands in `self.config`). TD-side: add a `Sendcontrolnetpreview` toggle par
to the COMP to expose it in the UI. Absent the par, defaults to `false`.

---

## To enable the preview later

Set `send_controlnet_preview: true` in `td_config.yaml` (for a one-off test) or add the
`Sendcontrolnetpreview` toggle par to the COMP and let the emitter handle it. Best used only
with a real preprocessing CN (Canny, etc.) where the preview differs from the input.

---

## Verification

- **Throttle gone:** run SD-Turbo / SDXL-Turbo, 512×512, 2-step, t_index=[32,45], seed=2,
  passthrough CN, `conditioning_scale≈0.44`. Confirm FPS recovers and stutter is gone while
  CN still conditions visibly.
- **No D2H on hot path:** grep per-frame CN path for `.cpu()`/`.numpy()` — none should execute
  with preview off.
- **Preview works when enabled:** set flag to `true` with a Canny preprocessor; confirm TD
  receives the edge image on `<stream>_out-cn-processed`.
