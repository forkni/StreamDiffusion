# True zero-copy GPU input for ControlNet — close the last per-frame CPU detour

> **Hand-off from `2026-05-17_zero-copy-gpu-input.md` (PR'd as `02911e5`, both Phase 1 + Phase 2 stream-sync hardening landed cleanly).** Main input is fully zero-copy. ControlNet input still takes the CPU detour every frame it's active. Same recipe applies — different config wiring, different format target.

## Context

After commit `02911e5` (zero-copy GPU input v1 + Phase 2 stream-sync hardening), the main img2img input is true zero-copy end-to-end. But when ControlNet is enabled, every frame still pays the old CPU cost on a parallel SHM channel:

| Direction | Channel | Transport | Payload handling |
|---|---|---|---|
| SD → TD (output) | `cuda_ipc_shm_name` | CUDAIPCExporter | **GPU end-to-end** (zero-copy) |
| TD → SD (main input) | `cuda_ipc_input_shm_name` | CUDAIPCImporter (Phase 2 GPU-fence) | **GPU end-to-end** (zero-copy) |
| TD → SD (controlnet) | `control_mem_name` = `<input>-cn` | **Legacy SharedMemory mmap** | **HWC uint8 → CPU float-cast → PIL roundtrip inside orchestrator** |
| TD → SD (ipadapter) | `ipadapter_mem_name` = `<input>-ip` | Legacy SHM mmap | Out of scope (OSC-triggered, preprocessor PIL detour — see Out of scope) |
| SD → TD (CN preprocessed) | `<output>-cn-processed` | Legacy SHM mmap | Out of scope (separate refactor) |

When ControlNet is active, `_process_controlnet_frame` runs **every frame** in the streaming loop (`td_manager.py:818`). For multi-CN setups the wrapper-side loop fires N updates per frame.

### What's already in place (verified by MCP search + Reads)

- **Tensor fast-path in orchestrator exists**: `PreprocessingOrchestrator.prepare_control_image` at `src/streamdiffusion/preprocessing/preprocessing_orchestrator.py:268-281` detects `isinstance(control_image, torch.Tensor)` and routes through `_process_tensor_input` (lines 636-656). For preprocessor-less inputs (passthrough), the path is: `unsqueeze(0) if dim==3 → .to(device=self.device, dtype=self.dtype)`. **GPU-only, no CPU roundtrip.** It expects **NCHW float [0,1]**.
- **Single-index entry from TD**: `ControlNetModule.update_control_image_efficient` at `src/streamdiffusion/modules/controlnet_module.py:152-203` calls `process_sync(image, preprocessors, scales, W, H, index)` per CN. `td_manager.py:841-842` loops over CN indices passing the same `control_frame`. Building one GPU tensor and reusing it across that loop is correct.
- **Importer API is identical to input direction**: `CUDAIPCImporter(shm_name=..., debug=False)` + `is_ready()` + `get_frame(stream=...)`. We already use it for input at `td_manager.py:677`. Same constructor, same lifecycle. Phase 2 stream-sync (`stream=torch.cuda.current_stream()`) applies the same way.
- **TD-side resize already happens**: `td_manager.py:824-832` reads `(height, width, 3)` from config — TD must already downsample CN frames to model resolution before SHM write. The new CUDA-Link Sender comp inherits the same constraint — no new size negotiation needed.

### What the current code does (the waste)

`StreamDiffusionTD/td_manager.py:818-842` (`_process_controlnet_frame`, hot path):

```python
control_frame = np.ndarray((height, width, 3), dtype=np.uint8, buffer=self.control_memory.buf)
if control_frame.dtype == np.uint8:
    control_frame = control_frame.astype(np.float32) / 255.0    # ← WASTE: CPU rescale
# ... per-CN loop:
for cn_idx in range(num_controlnets):
    self.wrapper.update_control_image(cn_idx, control_frame)    # numpy HWC float [0,1]
```

Then inside the orchestrator (current path for numpy input), `_convert_to_tensor` does:

```python
control_image = (control_image * 255).astype(np.uint8)         # ← WASTE: round-trip up
control_image = Image.fromarray(control_image)                  # ← WASTE: PIL allocation
control_tensor = self._cached_transform(control_image).unsqueeze(0)  # ← WASTE: ToTensor() rescales + H2D
control_tensor.to(device=self.device, dtype=self.dtype)
```

Net effect per CN per frame: `mmap read → CPU rescale → PIL allocation → ToTensor (rescale + H2D) → device cast`. With N CNs the orchestrator deduplicates input via `_last_input_frame is` identity check (`preprocessing_orchestrator.py:298-304`) so the H2D itself runs once, but the **per-frame CPU work** is unavoidable on the current path.

## Approach

Mirror the main-input plan exactly:
- New SHM channel `cuda_ipc_control_shm_name` (default `<input_mem_name>_control_ipc`)
- New gating flag `use_cuda_ipc_controlnet` (default `False` — fully backward compatible)
- New importer field `self._cuda_ipc_control_importer`, lazy-initialized on first frame
- New reader `_get_control_frame_cuda_ipc()` that returns **NCHW float32 [0,1]** GPU tensor (NOT [-1,1] like the main input — different format because the orchestrator's tensor passthrough path doesn't re-normalize)
- `_process_controlnet_frame` tries the IPC path when gated, falls back to legacy SHM otherwise

The TD-side .toe edits (adding a new CUDA-Link Sender comp publishing to `_control_ipc`) are manual work for the user — the plan calls out the requirement but doesn't deliver TD network changes (the .toe is a binary file, not Scripts/).

### The GPU transform (single chained op)

```python
# gpu_frame: HWC float32 BGRA on GPU, range [0,1] from Importer
# target:    NCHW float32 RGB [0,1] on GPU (orchestrator handles dtype cast)
nchw = (
    gpu_frame[..., [2, 1, 0]]      # HWC float32 RGB [0,1]   (drop alpha + BGR→RGB)
    .permute(2, 0, 1)              # CHW float32 RGB [0,1]
    .unsqueeze(0)                  # NCHW (N=1)
    .contiguous()                  # contiguous strides
)
```

**Key difference from main input**: no `mul(2).sub_(1)` scale to [-1,1], no `.to(dtype=...)`. The orchestrator's `_process_tensor_input` does the dtype cast itself at line 647/656 (`return ...to(device=self.device, dtype=self.dtype)`). Sending float32 keeps the contract simple — let the orchestrator decide when to downcast.

### Capability gating risk (called out for verification)

If any CN preprocessor lacks `process_tensor` (`preprocessing_orchestrator.py:641`), the fallback at line 664-665 forces `.cpu()` and a PIL roundtrip — **the zero-copy gain evaporates for that CN**. Most pure-passthrough CNs and several built-in preprocessors implement `process_tensor`; some image-analysis ones may not. Verification step (below) includes checking which preprocessors are configured.

## Code changes

### Patch 1 — `StreamDiffusionTD/td_manager.py:60-65` — config + importer field

```python
self.use_cuda_ipc_output = self.config.get('use_cuda_ipc_output', False)
self.use_cuda_ipc_input = self.config.get('use_cuda_ipc_input', False)
self.use_cuda_ipc_controlnet = self.config.get('use_cuda_ipc_controlnet', False)   # NEW
self.cuda_ipc_input_shm_name = self.td_settings.get('cuda_ipc_input_shm_name')
self.cuda_ipc_control_shm_name = self.td_settings.get('cuda_ipc_control_shm_name') # NEW
self._cuda_ipc_importer = None  # lazy-init on first frame
self._cuda_ipc_control_importer = None  # NEW: lazy-init on first CN frame
```

### Patch 2 — `StreamDiffusionTD/td_manager.py:392-410` — cleanup

Mirror the existing `_cuda_ipc_importer` cleanup block:

```python
if self._cuda_ipc_control_importer is not None:
    try:
        self._cuda_ipc_control_importer.cleanup()
    except Exception:
        pass
    self._cuda_ipc_control_importer = None
```

### Patch 3 — `StreamDiffusionTD/td_manager.py:705` — new probe helper

After existing `_probe_ipc_input_shm` (lines 705-717), add a sibling:

```python
def _probe_ipc_control_shm(self) -> bool:
    """Return True iff TD has created the CN IPC SharedMemory."""
    if not self.cuda_ipc_control_shm_name:
        return False
    try:
        from multiprocessing.shared_memory import SharedMemory
        shm = SharedMemory(name=self.cuda_ipc_control_shm_name)
        shm.close()
        return True
    except (FileNotFoundError, Exception):
        return False
```

### Patch 4 — `StreamDiffusionTD/td_manager.py` — new method `_get_control_frame_cuda_ipc`

Insert as a peer to `_get_input_frame_cuda_ipc` (currently around line 667-703):

```python
def _get_control_frame_cuda_ipc(self) -> Optional["torch.Tensor"]:
    """Read one CN frame from TD's CUDA IPC channel and return a GPU torch.Tensor
    matching the orchestrator's tensor passthrough contract: NCHW float32 RGB [0,1] on CUDA.
    Returns None if importer not ready (caller falls back to legacy SHM path).
    """
    if self._cuda_ipc_control_importer is None:
        if not self._probe_ipc_control_shm():
            return None
        from streamdiffusion._compat.cuda_ipc import CUDAIPCImporter
        try:
            self._cuda_ipc_control_importer = CUDAIPCImporter(
                shm_name=self.cuda_ipc_control_shm_name,
                debug=False,
            )
        except Exception as e:
            logger.warning(f"CUDAIPCImporter (control) init failed: {e}")
            self._cuda_ipc_control_importer = None
            return None
        if not self._cuda_ipc_control_importer.is_ready():
            self._cuda_ipc_control_importer = None
            return None
        logger.info(f"CUDA IPC control ready (zero-copy GPU): shm={self.cuda_ipc_control_shm_name}")

    gpu_frame = self._cuda_ipc_control_importer.get_frame(stream=torch.cuda.current_stream())
    if gpu_frame is None:
        return None

    # Orchestrator's _process_tensor_input handles dtype cast — we just normalize layout and channels.
    return (
        gpu_frame[..., [2, 1, 0]]       # HWC float32 RGB [0,1]  (drop alpha, BGR→RGB)
        .permute(2, 0, 1)               # CHW float32 RGB [0,1]
        .unsqueeze(0)                   # NCHW (N=1)
        .contiguous()
    )
```

### Patch 5 — `StreamDiffusionTD/td_manager.py:818-842` — branch in `_process_controlnet_frame`

Modify the body to try the IPC path when gated, fall back to legacy SHM otherwise:

```python
def _process_controlnet_frame(self) -> None:
    """Process ControlNet frame data (per-frame updates)"""
    if not self.config.get('use_controlnet', False):
        return

    control_frame = None

    # Fast path: CUDA IPC (zero-copy GPU tensor) if gated and TD emitter is up
    if self.use_cuda_ipc_controlnet:
        control_frame = self._get_control_frame_cuda_ipc()

    # Legacy fallback: SHM mmap → numpy HWC → CPU float-cast
    if control_frame is None:
        if not self.control_memory:
            return
        try:
            width = self.config['width']
            height = self.config['height']
            control_frame = np.ndarray((height, width, 3), dtype=np.uint8, buffer=self.control_memory.buf)
            control_frame = control_frame.astype(np.float32) / 255.0
        except Exception as e:
            logger.error(f"Error reading ControlNet SHM: {e}")
            return

    try:
        # Update ControlNet image for all active CNs (each runs its own preprocessor)
        cn_module = getattr(self.wrapper.stream, '_controlnet_module', None) if hasattr(self.wrapper, 'stream') else None
        num_controlnets = len(cn_module.controlnets) if cn_module is not None else 1
        for cn_idx in range(num_controlnets):
            self.wrapper.update_control_image(cn_idx, control_frame)

        # Send the processed image back to TD (unchanged — out of scope for this plan)
        try:
            if (hasattr(self.wrapper, 'stream') and
                hasattr(self.wrapper.stream, '_controlnet_module') and
                self.wrapper.stream._controlnet_module is not None):
                controlnet_module = self.wrapper.stream._controlnet_module
                if (hasattr(controlnet_module, 'controlnet_images') and
                    len(controlnet_module.controlnet_images) > 0 and
                    controlnet_module.controlnet_images[0] is not None):
                    processed_tensor = controlnet_module.controlnet_images[0]
                    self._send_processed_controlnet_frame(processed_tensor)
        except Exception as processed_error:
            logger.debug(f"Could not extract processed ControlNet image: {processed_error}")

    except Exception as e:
        logger.error(f"Error processing ControlNet frame: {e}")
```

### Patch 6 — `StreamDiffusionTD/td_config.yaml` — sample config (documentation)

Add commented-out reference values for the user to copy when they wire up the TD-side Sender:

```yaml
# CUDA IPC: ControlNet (set to true once TD has a CUDA-Link Sender publishing to this name)
use_cuda_ipc_controlnet: false
td_settings:
  cuda_ipc_input_shm_name: StreamDiffusionTD_512-512_input_ipc
  # cuda_ipc_control_shm_name: StreamDiffusionTD_512-512_control_ipc   # NEW (commented-out by default)
```

### TD-side manual work (called out, NOT in this PR)

For `use_cuda_ipc_controlnet=true` to actually work, the user must:

1. In the .toe network, add a second `CUDA-Link` Sender comp parallel to the existing input one
2. Wire it to the CN preview TOP (the same source that currently feeds the `-cn` SHM)
3. Set its shm name to match `cuda_ipc_control_shm_name` (default `StreamDiffusionTD_512-512_control_ipc`)
4. Flip `use_cuda_ipc_controlnet: true` in `td_config.yaml`

If the user skips these steps and flips the flag, the SD side will log "CUDAIPCImporter (control) init failed" and gracefully fall back to legacy SHM — no breakage.

### What we explicitly do NOT touch

- **IPAdapter path** — OSC-triggered (`td_manager.py:882` early-exit on `ipadapter_update_requested`), and `IPAdapterEmbeddingPreprocessor._process_tensor_core` (`processors/ipadapter_embedding.py:55-59`) forces a PIL roundtrip anyway. Transport-only zero-copy would buy ~nothing. Deferred until either cadence changes OR preprocessor is refactored.
- **CN preprocessed return path** (`_send_processed_controlnet_frame` + `<output>-cn-processed` SHM) — separate SD→TD direction, separate refactor, separate PR.
- **`PreprocessingOrchestrator._process_tensor_input` PIL fallback** at line 664-665 — if a preprocessor lacks `process_tensor`, that branch defeats the win. Out of scope: changing the orchestrator. Mitigation: documented as a verification step below.
- **Preprocessor `process_tensor` implementations** — adding GPU paths to preprocessors that lack one. Per-preprocessor refactor, separate work.
- **`_compat/cuda_ipc/`** — no changes; the existing API is sufficient.
- **`.toe` network edits** — manual user work, documented above but not part of this PR.

## Verification

After applying all six patches in the running SD venv (no rebuild needed — pure Python; Scripts/ edits live-reload per `[[project_scripts_dir_purpose]]`):

### 1. Smoke test — import contract

```powershell
venv\Scripts\python -c "from StreamDiffusionTD.td_manager import TouchDesignerManager; print('OK')"
```

Must print `OK`. Any `SyntaxError`/`NameError`/`ImportError` means a patch is wrong — stop and re-read.

### 2. Backward-compatibility test — legacy SHM still works

With `use_cuda_ipc_controlnet: false` (default) and CN enabled, relaunch the .toe. Expect:
- Legacy `-cn` SHM still serves frames (no regression vs baseline)
- No new log markers
- Same FPS / quality as before this PR

### 3. CUDA IPC opt-in test (requires TD-side Sender setup)

After user adds a CUDA-Link Sender comp publishing to `StreamDiffusionTD_512-512_control_ipc` and flips `use_cuda_ipc_controlnet: true`:
- SD log shows `CUDA IPC control ready (zero-copy GPU): shm=StreamDiffusionTD_512-512_control_ipc`
- ControlNet preview in TD shows correct colors (BGR→RGB shuffle on GPU instead of CPU)
- No `_get_control_frame_cuda_ipc:` errors in log
- If Sender comp missing or wrong name: clean fallback to legacy SHM with one-time `init failed` warning, no crash

### 4. Preprocessor capability check

Before claiming the win, verify the active CN preprocessors actually have `process_tensor`:

```powershell
venv\Scripts\python -c "
from streamdiffusion.preprocessing.processors import REGISTRY
for name, cls in REGISTRY.items():
    has = hasattr(cls, 'process_tensor')
    print(f'{name}: process_tensor={has}')
"
```

For any preprocessor used in the user's config WITHOUT `process_tensor`, the zero-copy gain evaporates (CPU PIL fallback at `preprocessing_orchestrator.py:664-665`). That's a follow-up item, not a blocker.

### 5. Performance verification

With CN active and `use_cuda_ipc_controlnet=true`, compare against the legacy-SHM baseline:
- **Steady-state `total_time`**: expected ~0.3-0.8ms lower (CN's CPU rescale + PIL roundtrip + H2D eliminated)
- **`total_time` jitter**: should tighten when CN active (one fewer per-frame CPU detour)
- **CN preview latency**: subjective TD-side check — visible reduction in N-CN-multi setups

Optional `nsys` check: should show zero `cudaMemcpyAsync HtoD` calls between consecutive `cudaGraphLaunch`es originating from the CN code path.

## Commit

Per `[[feedback_pr_branch_convention]]`, branch stays at `feat/cuda-ipc-output` (current head: `02911e5`), PR target `SDTD_031_dev`.

```powershell
./scripts/git/commit_enhanced.sh --no-venv `
  "feat: zero-copy GPU input for ControlNet via CUDA IPC (transport parity with main input)"
```

Then save the plan as a project file per `[[feedback_save_plans_as_project_files]]`:
- Copy this file to `StreamDiffusion/_plans/2026-05-17_controlnet-zero-copy.md`

Note: `StreamDiffusionTD/td_manager.py` is gitignored (lives in companion `dotsimulate/StreamDiffusionTD` repo). Only the plan file and `src/`-side touches (if any) are committable in this repo. The `td_manager.py` patches go through the companion repo per `[[project_td_release_flow]]`.

## Critical files

| File | Lines | Change |
|---|---|---|
| `StreamDiffusionTD/td_manager.py` | 60-65 | Patch 1 — add `use_cuda_ipc_controlnet`, `cuda_ipc_control_shm_name`, `_cuda_ipc_control_importer` |
| `StreamDiffusionTD/td_manager.py` | 392-410 | Patch 2 — cleanup `_cuda_ipc_control_importer` |
| `StreamDiffusionTD/td_manager.py` | 705-717 | Patch 3 — add `_probe_ipc_control_shm` helper |
| `StreamDiffusionTD/td_manager.py` | (new method near 667-703) | Patch 4 — add `_get_control_frame_cuda_ipc` |
| `StreamDiffusionTD/td_manager.py` | 818-842 | Patch 5 — branch `_process_controlnet_frame` on IPC vs SHM |
| `StreamDiffusionTD/td_config.yaml` | (config schema) | Patch 6 — sample config keys (commented) |

Reused unchanged (verified):

- `src/streamdiffusion/_compat/cuda_ipc/cuda_ipc_importer.py:903` — `get_frame(stream=...)` already supports GPU-side fence
- `src/streamdiffusion/preprocessing/preprocessing_orchestrator.py:268-281, 636-656` — tensor fast-path already exists
- `src/streamdiffusion/modules/controlnet_module.py:152-203` — `update_control_image_efficient` accepts tensors via per-index passthrough
- `src/streamdiffusion/wrapper.py:2390-2399` — `update_control_image` already forwards tensors unchanged

## Out of scope (documented for future work)

- **IPAdapter zero-copy** — see Approach. Two reasons it's deferred: (a) OSC-triggered cadence vs per-frame, so the optimization saves a few CPU ms per *user trigger* not per frame; (b) `IPAdapterEmbeddingPreprocessor._process_tensor_core` actively defeats the tensor fast path by converting back to PIL for the CLIP image processor. Worth doing only after the preprocessor is refactored to keep tensors on GPU.
- **CN return path zero-copy** (`_send_processed_controlnet_frame` → `<output>-cn-processed` SHM) — separate SD→TD direction; mirror the existing main-output `CUDAIPCExporter` pattern.
- **`PreprocessingOrchestrator._process_tensor_input` PIL fallback** — when a preprocessor lacks `process_tensor`, line 664-665 falls back to `.cpu()`. Not blocking this PR, but the per-preprocessor `process_tensor` work is a real follow-up.
- **TD-side .toe Sender comp wiring** — manual user work (the .toe is a binary file, edited in TouchDesigner not git).
