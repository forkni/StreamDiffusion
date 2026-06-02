# cuda-link v1.7.2 Integration Update — StreamDiffusion

> **SUPERSEDED** — Upgraded to v1.8.1 with full vendoring removal (refactor: depend solely on
> pip cuda-link v1.8.1, retire `_compat` mirrors). See `feat/cuda-ipc-output` commit history.

## Context

The user upgraded the `cuda-link` library to **v1.7.2** (pip-installed in the project venv) and
manually overlaid `src/streamdiffusion/_compat/td_exporter/` with the canonical v1.7.2 source from
`F:\RD_PROJECTS\COMPONENTS\cuda-link\td_exporter\`. The task is to finish bringing the
StreamDiffusion integration of cuda-link fully in line with v1.7.2.

**Key finding from exploration:** the *live runtime path* already works on 1.7.2. `wrapper.py`
imports `Exporter / FrameSpec / GpuFrame / FrameOutcome` directly from the installed pip package,
and that API has been stable since v1.5.0 (the v1.7.2 additions — `FrameSpec.extra_flags`,
`FLAGS_MONO_ALPHA` — are additive with safe defaults and unused by the wrapper's uint8-BGRA path).
Nothing in the project imports the two `_compat/` mirrors at runtime (verified by grep) — they are
offline reference/fallback copies that project convention keeps in lockstep with the library.

So this update is **consistency + cleanup work on the vendored mirrors**, not a behavioral change:

| Surface | Current | Target | Action |
|---|---|---|---|
| venv pip `cuda_link` (live) | 1.7.2 ✓ | 1.7.2 | none (already current) |
| `_compat/td_exporter/` | content ~1.7.2, but overlay left 2 stragglers + stale stamp | 1.7.2 clean | clean re-copy + regenerate version file |
| `_compat/cuda_ipc/` | 1.5.1 (lags 4 releases) | 1.7.2 | re-vendor + re-apply relative-import patches |
| `td_manager.py` (deprecated `CUDAIPCImporter`) | works in 1.7.2 | unchanged | DEFER migration |

**Scope decisions:**
- Re-vendor `_compat/cuda_ipc/` to 1.7.2 — YES.
- Migrate `td_manager.py` off deprecated `CUDAIPCImporter` → `Importer.open(ImportSpec)` — DEFERRED
  (the class still ships in 1.7.2; removed only in v1.8.0; migration is a behavioral change to the
  live TD input/control path requiring TD-hardware testing — do it later as a dedicated change).

Repo root: `D:\dev\SD_3_0_1\test_Install_dev\StreamDiffusion\StreamDiffusion\`

---

## Changes made (2026-05-31)

### Step 1 — `_compat/td_exporter/` re-synced to 1.7.2

- Deleted all old `.py` + `.md` files from `src/streamdiffusion/_compat/td_exporter/`
- Re-copied all 30 `.py` files + `HELP_DOC.md` from canonical
  `F:\RD_PROJECTS\COMPONENTS\cuda-link\td_exporter\`
- Explicitly deleted straggler `CudaAdapters.py` (old v1.6.x name) and `warning_emitter_callbacks.py`
  (folded into `script_top_callbacks.py` in v1.7.0, absent from canonical v1.7.2)
- Re-copied canonical `CUDAAdapters.py` (correct v1.7.0+ name)
- Regenerated `VENDORED_VERSION.txt` → version 1.7.2, 2026-05-31

**Note:** `td_exporter/` files use flat-namespace imports (`from SHMProtocol import ...`) — no
relative-import patches applied (correct for TD Text-DAT context).

### Step 2 — `_compat/cuda_ipc/` re-vendored 1.5.1 → 1.7.2

- Copied all 17 modules + `py.typed` from canonical `F:\RD_PROJECTS\COMPONENTS\cuda-link\src\cuda_link\`
  over `src/streamdiffusion/_compat/cuda_ipc/` (head commit `9be8212d`, 2026-05-31)
- Adds `_console.py` (new in v1.6.0; a Windows console-handler helper)
- Applied **5 relative-import patches** across **3 files** to avoid the cross-package class-identity
  bug (upstream dual-namespace try/except blocks that otherwise resolve to the pip package):
  - `cuda_ipc_wrapper.py` ×3: env_bool, cuda_runtime_types import, cuda_graphs import
  - `cuda_graphs.py` ×1: cuda_runtime_types import
  - `nvml_observer.py` ×1: env_bool
- Updated `VENDORED_VERSION.txt` → version 1.7.2, head_commit, updated TRAP section

### Step 3 — `wrapper.py` comment refresh

- `src/streamdiffusion/wrapper.py:918`: stale `v1.7.1+` comment updated to `v1.5.0+ API` (the Exporter API is stable since v1.5.0; no code change).

### Step 4 — `td_manager.py` deprecation TODO comments

- Added `# TODO(cuda-link v1.8.0): CUDAIPCImporter deprecated — migrate to Importer.open(ImportSpec(...))`
  at both `CUDAIPCImporter` import sites (input channel and ControlNet channel) in:
  - `StreamDiffusionTD/td_manager.py` (lines ~345, ~373)
  - `Scripts/streamdiffusionTD__Text__td_manager__td.py` (the live .tox-synced twin)

---

## RE-VENDORING TRAP (for future updates)

When re-vendoring `_compat/cuda_ipc/` in the future, ALWAYS re-apply the 5 relative-import patches
after copying from canonical (see `VENDORED_VERSION.txt` for the exact blocks). The `td_exporter/`
mirror does NOT need patching.

---

## Deferred: `td_manager.py` Importer.open() migration

The `CUDAIPCImporter` class (used in `td_manager.py`) is deprecated and will be removed in
cuda-link v1.8.0. Migration path:

```python
# BEFORE (current — deprecated CUDAIPCImporter):
imp = CUDAIPCImporter(shm_name=..., device=..., timeout_ms=...)
imp.connect()
if not imp.is_ready(): ...
# get_frame() returns a raw GPU tensor

# AFTER (new API):
from cuda_link import Importer, ImportSpec, ImportOutcome
imp = Importer.open(ImportSpec(shm_name=..., device=..., timeout_ms=...))
# Note: get_frame() now returns ImportResult(outcome, frame)
result = imp.get_frame()
if result.outcome == ImportOutcome.NEW_FRAME:
    gpu_tensor = result.frame  # the GPU tensor
```

The `_get_input_frame()` and `_process_controlnet_frame()` methods in `td_manager.py` will need
updating to unwrap `ImportResult`. Requires TD-hardware testing on RTX 4090 with live IPC stream.
