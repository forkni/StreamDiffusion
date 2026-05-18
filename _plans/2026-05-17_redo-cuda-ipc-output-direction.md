# Redo CUDA IPC integration — rename SD-side to cuda-link vocabulary, wire output direction

## Context

Phase 1 (restore) is complete. The working tree is back to a clean post-Quantization state with `_compat/cuda_ipc/` and `_compat/td_exporter/` re-vendored as pristine copies from `F:\RD_PROJECTS\COMPONENTS\cuda-link` (commit `92989fc`, version `1.4.1`). No IPC glue remains in `wrapper.py`, `config.py`, `td_config.yaml.example`, `StreamDiffusionTD/td_manager.py`, or `StreamDiffusionTD/td_config.yaml`. The abandoned plan that mapped cuda-link Sender→SD "input" / Receiver→SD "output" is archived at `_plans/archive/2026-05-17_cuda-ipc-input-perspective_ABANDONED.md`.

Phase 2 (this plan) redoes the integration with cuda-link's vocabulary kept verbatim, renaming SD-side names to match. The previous attempt's input-direction (Importer) first-connect noise at `cuda_ipc_importer.py:808` is sidestepped for this round by scoping to the output direction only (SD's Exporter), which was the proven 16-25 FPS path in abandoned Phase 2.1.

User decisions captured this session:
- **Scope**: Rename + output IPC only. Input direction (Importer) deferred to a follow-up.
- **Naming**: `td_export_shm_name` / `td_import_shm_name` (TD's perspective — mirrors what user types into TD's `Ipcmemname` param).
- **Migration**: Hard rename, no backwards-compat shim. Users update their deployed `td_config.yaml` manually.
- **SHM name format (TD-side, already in place)**: `parent.SDTD.par.Streamoutname + '_input_ipc'` for the SHM SD's Exporter writes into (TD reads as input); `parent.SDTD.par.Streamoutname + '_output_ipc'` for the SHM TD writes into (SD's Importer would read, out of scope this round). The TD StreamDiffusionExt computes these — SD just receives the final strings verbatim.

## Target end state

- Two new YAML keys: `td_settings.td_export_shm_name` (TD writes, SD reads — out-of-scope wiring) and `td_settings.td_import_shm_name` (SD writes, TD reads — CUDA IPC wired this round). The old `input_mem_name` / `output_mem_name` keys are gone from both `td_config.yaml` and `td_config.yaml.example`.
- `TouchDesignerManager.__init__` takes `td_export_shm_name` / `td_import_shm_name` positional params. All internal SHM-name references renamed.
- `StreamDiffusionWrapper` gains an opt-in CUDA IPC fast-path in `postprocess_image` (or its caller) for the SD→TD direction. Falls through to legacy CPU SHM behavior when IPC is disabled.
- A new `use_cuda_ipc` YAML flag controls the IPC fast-path (default `false` for safe rollout).
- `td_main.py` drops the `_{int(time.time())}` uniquifier on the output name — `Streamoutname`-based naming is already per-COMP unique.
- StreamDiffusionExt YAML emitter inside the .toe binary writes the two new keys with the IPC-suffix values (user edits inside TouchDesigner — out of repo).
- Smoke test passes: `from streamdiffusion.wrapper import StreamDiffusionWrapper` (assuming pre-existing `controlnet_aux` is installed, unrelated). End-to-end TD test: open `.toe`, configure `Streamoutname`, see frames flow at ≥16 FPS with `use_cuda_ipc: true` in `td_config.yaml`.

## Execution (in order)

All paths relative to `D:/dev/SD_3_0_1/test_Install_dev/StreamDiffusion/StreamDiffusion/` unless absolute.

### Step 1 — Rename SD-side YAML keys (mechanical)

**`StreamDiffusionTD/td_config.yaml`** (current L73-74) and **`configs/td_config.yaml.example`** (current L95-96): replace

```yaml
  input_mem_name: 'StreamDiffusionTD_512-512'
  output_mem_name: 'StreamDiffusionTD_512-512_out'
```

with

```yaml
  # cuda-link CUDA IPC shared-memory names (auto-emitted by TD StreamDiffusionExt)
  # td_export_shm_name: TD writes here (Sender/ExportBuffer); SD's CUDAIPCImporter reads. Not wired this round.
  # td_import_shm_name: SD writes here (CUDAIPCExporter); TD reads (Receiver/ImportBuffer).
  td_export_shm_name: 'StreamDiffusionTD_512-512_output_ipc'
  td_import_shm_name: 'StreamDiffusionTD_512-512_input_ipc'

  # CUDA IPC output toggle (false = legacy CPU multiprocessing.shared_memory)
  use_cuda_ipc: false
```

### Step 2 — Rename `TouchDesignerManager` ctor + internal refs

Edit `StreamDiffusionTD/td_manager.py` (authoritative — copied directly from `.toe` binary into this dir per [[project_scripts_dir_purpose]]; gitignored). Scripts/ is the OLD stale sync dir — do **not** touch.

- **L40**: signature change
  - Before: `def __init__(self, config, input_mem_name: str, output_mem_name: str, debug_mode=False, osc_reporter=None)`
  - After:  `def __init__(self, config, td_export_shm_name: str, td_import_shm_name: str, debug_mode=False, osc_reporter=None)`
- **L41-42**: `self.input_mem_name = input_mem_name` → `self.td_export_shm_name = td_export_shm_name`; same for `td_import_shm_name`.
- **L98-100**: derived names — replace `self.input_mem_name` with `self.td_export_shm_name` (the suffixed `-cn`/`-cn-processed`/`-ip` names tag onto whatever-data-flows-into-SD, which is TD's export side).
- **L314-322** (macOS Syphon block): `sender_name=self.td_import_shm_name, input_name=self.td_export_shm_name` (Syphon perspective is unaffected by the rename — it's just two names; updating identifiers).
- **L330-346**: CPU SHM connect/create — `self.td_export_shm_name` replaces `self.input_mem_name`; `self.td_import_shm_name` replaces `self.output_mem_name`.
- **L358**: `control_processed_mem_name = self.td_import_shm_name + '-cn-processed'`.
- **L534**: `_send_output_frame(output_image)` call site — no change, only the buffer name's identifier changes.
- **L643-740** (`_send_output_frame`): inspect for any `self.output_mem_name` references — rename to `self.td_import_shm_name`.
- **L673** (`buffer=self.output_memory.buf`): no rename needed (attribute name `output_memory` is a Python attr, not a SHM identifier — keep as-is; it's the CPU SHM handle that may be `None` when IPC active).

Verification: `grep -nE "(input_mem_name|output_mem_name)" StreamDiffusionTD/td_manager.py` returns 0 hits.

### Step 3 — Drop timestamp uniquifier in `td_main.py`

Edit `StreamDiffusionTD/td_main.py` L399-415:

```python
# Before
input_mem = td_settings.get('input_mem_name', 'input_stream')
base_output_name = td_settings.get('output_mem_name', 'sd_to_td')
output_mem = f"{base_output_name}_{int(time.time())}"
# ...
self.manager = TouchDesignerManager(yaml_config, input_mem, output_mem, ...)

# After
td_export_shm_name = td_settings['td_export_shm_name']  # required, no fallback per hard-rename
td_import_shm_name = td_settings['td_import_shm_name']
# ...
self.manager = TouchDesignerManager(
    yaml_config,
    td_export_shm_name,
    td_import_shm_name,
    debug_mode=debug_mode,
    osc_reporter=osc_reporter,
)
```

- **L450**: log line — `print(f"\033[38;5;80mMemory: \033[37m{td_export_shm_name} <- TD | SD -> {td_import_shm_name}\033[0m")` (direction arrows make the new naming readable).
- **L462**: `self.osc_reporter.send_output_name(self.manager.td_import_shm_name)` (the OSC reporter announces the SHM name where TD should read).

The timestamp uniquifier is dropped because `Streamoutname` (TD-side, e.g. `StreamDiffusionTD_512-512`) is already per-COMP unique. If a user has two COMPs at the same resolution they need different `Streamoutname` values — that's a TD-side configuration concern, not a Python-side uniqueness trick.

### Step 4 — Wire SD→TD CUDA IPC output direction

#### 4a — `src/streamdiffusion/wrapper.py`

Add ctor kwargs (near existing ones around L82-160):

```python
use_cuda_ipc: bool = False,
cuda_ipc_shm_name: str | None = None,
cuda_ipc_num_slots: int = 2,
```

Add instance slots in `__init__` (alongside existing output-type init around L312):

```python
self.use_cuda_ipc = use_cuda_ipc
self._cuda_ipc_shm_name = cuda_ipc_shm_name
self._cuda_ipc_num_slots = cuda_ipc_num_slots
self._cuda_ipc_exporter = None  # lazy-init on first frame
```

Add a fast-path inside `postprocess_image` (currently at `wrapper.py:894-948`), early-exit when IPC is active. The function signature stays the same; the new branch happens before the existing `output_type == "pil"|"pt"|"np"|"latent"` dispatch:

```python
def postprocess_image(self, image_tensor, output_type="pil"):
    # CUDA IPC fast-path: zero-copy BGRA export to TD via _compat.cuda_ipc.
    # Skips D2H, CPU repack, and CPU SHM write. Returns None to signal "frame
    # consumed by IPC" so the caller's CPU SHM write path is skipped.
    if self.use_cuda_ipc and self._cuda_ipc_shm_name:
        bgra = self._ipc_pack_rgba(image_tensor)  # HWC uint8 BGRA on GPU
        exporter = self._lazy_init_ipc_exporter(bgra.shape[0], bgra.shape[1])
        exporter.export_frame(bgra.data_ptr(), bgra.numel())
        return None

    # ... existing dispatch unchanged ...
```

Add helpers:

```python
def _ipc_pack_rgba(self, image_tensor):
    # Convert pipeline output to HWC uint8 BGRA on GPU. cuda-link expects BGRA
    # per CUDAIPCExporter docstring (cuda_ipc_exporter.py:11-22).
    # image_tensor is NCHW float [0,1] for SDXL pipelines — see uses at
    # wrapper.py:787, 812, 853.
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]  # CHW
    x = (image_tensor.clamp(0, 1) * 255).to(torch.uint8)  # CHW uint8
    rgb = x.permute(1, 2, 0).contiguous()  # HWC RGB
    # BGRA = swap R↔B, append alpha=255
    bgra = torch.cat([
        rgb[..., 2:3], rgb[..., 1:2], rgb[..., 0:1],
        torch.full(rgb.shape[:-1] + (1,), 255, dtype=torch.uint8, device=rgb.device),
    ], dim=-1).contiguous()
    return bgra

def _lazy_init_ipc_exporter(self, height, width):
    if self._cuda_ipc_exporter is not None:
        return self._cuda_ipc_exporter
    from streamdiffusion._compat.cuda_ipc import CUDAIPCExporter
    exporter = CUDAIPCExporter(
        shm_name=self._cuda_ipc_shm_name,
        height=height, width=width,
        channels=4, dtype="uint8",
        num_slots=self._cuda_ipc_num_slots,
        debug=False,
    )
    exporter.initialize()  # required per cuda_ipc_exporter.py:338
    self._cuda_ipc_exporter = exporter
    return exporter
```

Add cleanup in the existing `cleanup`/`__del__` path (wrapper has a teardown method around L2658 in the abandoned-plan diff — confirm exact location during implementation):

```python
if self._cuda_ipc_exporter is not None:
    self._cuda_ipc_exporter.cleanup()
    self._cuda_ipc_exporter = None
```

#### 4b — `src/streamdiffusion/config.py`

Add three new `param_map` entries so YAML keys reach the wrapper ctor:

```python
"use_cuda_ipc": "use_cuda_ipc",
"cuda_ipc_shm_name": "cuda_ipc_shm_name",
"cuda_ipc_num_slots": "cuda_ipc_num_slots",
```

(Exact dict location TBD during implementation — search `param_map` in `config.py`.)

#### 4c — `StreamDiffusionTD/td_manager.py` — pass YAML through

In the `create_wrapper_from_config` call site (L72) the config dict already flows to the wrapper through `config.py`'s param_map. The only TD-side concern is:

- Read `self.td_settings.get('use_cuda_ipc', False)` once and stash on `self.use_cuda_ipc`.
- When `use_cuda_ipc` is true, inject `cuda_ipc_shm_name = self.td_import_shm_name` into the config dict before passing to `create_wrapper_from_config`. This keeps the wrapper agnostic of the TD-perspective YAML naming.
- Skip the CPU SHM connect/create for the **output direction** (L335-346) when IPC is on — the SHM doesn't need to exist on the Python side. Input-side CPU SHM (L330) still opens.

#### 4d — Output frame send path

In `_send_output_frame` (L643-740), early-return when `self.use_cuda_ipc` is true:

```python
def _send_output_frame(self, output_image):
    if self.use_cuda_ipc:
        # Frame was already exported via wrapper.postprocess_image IPC fast-path;
        # output_image is None.
        return
    # ... existing CPU SHM write path unchanged ...
```

This requires `postprocess_image` returning `None` to propagate up to `_send_output_frame`'s caller (currently the streaming loop at L534) without breaking. Check `wrapper.__call__` or whichever method calls `postprocess_image` for the chain.

### Step 5 — StreamDiffusionExt YAML emitter (manual TD edit)

Out of repo. Inside `StreamDiffusionTD_dev.toe`, the StreamDiffusionExt extension's YAML-emitter code (previously at extracted L3754-3768 in the old Scripts/-synced copy) must be updated to emit:

```python
yaml_lines.append(f"  td_export_shm_name: '{parent.SDTD.par.Streamoutname}_output_ipc'")
yaml_lines.append(f"  td_import_shm_name: '{parent.SDTD.par.Streamoutname}_input_ipc'")
yaml_lines.append(f"  use_cuda_ipc: {bool(parent.SDTD.par.Usecudaipc)}")  # new toggle param on SDTD COMP
```

The user has to make this edit directly inside TouchDesigner. The CUDAIPCExtension COMPs inside the .toe also need their `Ipcmemname` params bound to the same `Streamoutname + '_input_ipc'` (Receiver-mode COMP) so SD and TD agree on the SHM identity.

### Step 6 — Verification

```bash
# Static rename completeness
grep -rn "input_mem_name\|output_mem_name" StreamDiffusion/StreamDiffusionTD/ src/streamdiffusion/ configs/ 2>&1
# → zero hits

# Import smoke (pre-existing controlnet_aux missing is unrelated; wrapper module itself loads)
cd StreamDiffusion && python -c "from streamdiffusion.wrapper import StreamDiffusionWrapper; print('import OK')"

# IPC class loadability
python -c "from streamdiffusion._compat.cuda_ipc import CUDAIPCExporter; print(CUDAIPCExporter.__init__.__doc__[:200])"
```

End-to-end TD test:
1. Open `StreamDiffusionTD_dev.toe`. Set `parent.SDTD.par.Streamoutname = "StreamDiffusionTD_512-512"` and `Usecudaipc = True`.
2. Update StreamDiffusionExt YAML emitter per Step 5; re-emit `td_config.yaml`.
3. Restart Python pipeline. Verify console shows `Memory: <export_name> <- TD | SD -> <import_name>_input_ipc`.
4. Trigger a frame from TD's input. Confirm SD's output appears in TD's Receiver COMP at ≥16 FPS (matching abandoned Phase 2.1's 16-25 FPS).
5. Hot-disable IPC: set `use_cuda_ipc: false` in YAML, restart, confirm legacy CPU SHM path still works (regression check).

## Critical files & key references

**Modified (this plan)**:
- `StreamDiffusionTD/td_config.yaml` L73-74 — YAML keys rename + IPC toggle add
- `configs/td_config.yaml.example` L95-96 — same
- `StreamDiffusionTD/td_manager.py` L40, L41-42, L98-100, L314-322, L330-346, L358, L643-740 — ctor + internal refs + output send early-return
- `StreamDiffusionTD/td_main.py` L399-418, L450, L462 — drop timestamp uniquifier, update kwargs + log
- `src/streamdiffusion/wrapper.py` L82-160 (ctor kwargs), L312 (instance slots), L894 (`postprocess_image` IPC fast-path), L2658-area (cleanup) — IPC wiring
- `src/streamdiffusion/config.py` — three new `param_map` entries
- StreamDiffusionExt YAML emitter (in `.toe` binary, out of repo) — user edits in TD

**Reused verbatim (no edits)**:
- `src/streamdiffusion/_compat/cuda_ipc/` — pristine cuda-link `92989fc`. Integration surface: `CUDAIPCExporter` at `cuda_ipc_exporter.py:191` (ctor L208-218, `initialize()` L338, `export_frame(gpu_ptr, size)` L705, `cleanup()` L957). BGRA HWC uint8 wire format per docstring L11-22.
- `src/streamdiffusion/_compat/td_exporter/` — pristine cuda-link TD-side. `TDSender.py:70` (`_EXPORT_BUFFER_NAME = "ExportBuffer"`), `CUDAIPCExtension.py:98` (`Ipcmemname` default `cudalink_output_ipc`), `TDReceiver.py:171-190` (`RetryState` — relevant for Step 5 user-side TD config).

**Out of scope (next round)**:
- Input direction (TD→SD): `CUDAIPCImporter` wiring inside `_get_input_frame`. Requires solving the first-connect `traceback.print_exc()` noise at `cuda_ipc_importer.py:808` via SHM-existence probe before construction (the abandoned-Phase-2.4 approach). The YAML key `td_export_shm_name` is reserved for this.
- Migration of deployed user `td_config.yaml` files. Per user choice, manual update.
- `controlnet_aux` import error (pre-existing missing dependency unrelated to this work).

## Memory note

Per [[feedback_save_plans_as_project_files]], after ExitPlanMode and on user approval, copy this plan to `D:/dev/SD_3_0_1/test_Install_dev/StreamDiffusion/StreamDiffusion/_plans/2026-05-17_redo-cuda-ipc-output-direction.md` as the project-tracked copy.
