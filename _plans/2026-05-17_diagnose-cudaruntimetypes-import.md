# Diagnose & fix `_get_input_frame: No module named 'CUDARuntimeTypes'`

> **Hand-off from `cozy-snacking-wilkinson.md` Round 2.** Both Round 1 (SD-side) and Round 2 (TD emitter) landed. The `FileNotFoundError` is gone. A new failure replaced it: the SD-side Importer construction throws `ModuleNotFoundError: No module named 'CUDARuntimeTypes'` on every frame, so no input ever reaches the wrapper.

## Context

**What the user observed** (SD cmd log + TD textport, 2026-05-17 21:13):

- ✅ Round 1 guard worked: `CUDA IPC input active; legacy SharedMemory skipped (will read StreamDiffusionTD_512-512_input_ipc)` — no `FileNotFoundError`.
- ❌ Round 2 success criterion missing: no `CUDA IPC input ready: shm=...` log line — Importer was never successfully constructed.
- ❌ 13 × `TouchDesignerManager - DEBUG - _get_input_frame: No module named 'CUDARuntimeTypes'` at startup, all at 21:13:03 (~77ms/attempt), then SD log goes silent while TD Sender continues writing frames at full rate (TD log shows Frame 97 → Frame 2716 timing).

The error message format `_get_input_frame: <msg>` matches the OUTER catch at `td_manager.py:_get_input_frame` (debug-only):

```python
except Exception as e:
    if self.debug_mode:
        logger.debug(f"_get_input_frame: {e}")
    return None
```

That catch sees the exception raised by the `from streamdiffusion._compat.cuda_ipc import CUDAIPCImporter` statement inside `_get_input_frame_cuda_ipc` (Step 4 of cozy-snacking) — it sits OUTSIDE the inner try/except, so it propagates up.

## Root cause (verified)

The vendored `_compat/cuda_ipc/` package has **two files with broken absolute imports** that worked in the upstream `cuda_link` package context (where `cuda_link` is pip-installed) AND in TouchDesigner's flat namespace (where `CUDARuntimeTypes` is a top-level module), but fail in SD's venv where **neither** is available:

### File 1 — `src/streamdiffusion/_compat/cuda_ipc/cuda_ipc_wrapper.py:24-54`

```python
try:
    from cuda_link.cuda_runtime_types import (   # ← NOT installed in SD venv
        CUDAError, CUDAEvent_t, CUDAGraph_t, CUDAGraphExec_t, CUDAGraphNode_t,
        CUDAStream_t, cudaIpcEventHandle_t, cudaIpcMemHandle_t,
        cudaMemcpy3DParms, cudaPointerAttributes,
    )
except ImportError:
    from CUDARuntimeTypes import (               # ← TD-only top-level module — also missing
        CUDAError, CUDAEvent_t, ...
    )

try:
    from cuda_link.cuda_graphs import CUDAGraphsMixin   # ← same problem
except ImportError:
    from CUDAGraphs import CUDAGraphsMixin             # ← same problem
```

### File 2 — `src/streamdiffusion/_compat/cuda_ipc/cuda_graphs.py:18-41`

Same try/except pattern: tries `cuda_link.cuda_runtime_types`, falls back to top-level `CUDARuntimeTypes`. Both fail in SD.

### Import chain that triggers the failure

1. `_get_input_frame_cuda_ipc` runs `from streamdiffusion._compat.cuda_ipc import CUDAIPCImporter`.
2. `_compat/cuda_ipc/__init__.py:11` does `from .cuda_ipc_wrapper import CUDARuntimeAPI, get_cuda_runtime`.
3. `cuda_ipc_wrapper.py:24-49` hits the broken pair → raises `ModuleNotFoundError: No module named 'CUDARuntimeTypes'`.
4. (Even if the wrapper import were skipped, `cuda_ipc_importer.py:108` does `from .cuda_ipc_wrapper import CUDARuntimeAPI, get_cuda_runtime` directly — same failure.)
5. ImportError propagates → outer catch in `_get_input_frame` → DEBUG log → returns None → loop retries forever.

### Why the sibling vendored files already exist

`_compat/cuda_ipc/cuda_runtime_types.py` is present and exports every symbol the wrapper/graphs files need — verified via grep:

- Classes: `cudaPos`, `cudaMemcpy3DParms`, `cudaIpcMemHandle_t`, `cudaIpcEventHandle_t`, `cudaPointerAttributes`, `CUDAError`, `cudaPitchedPtr`, `cudaExtent`
- Aliases: `CUDAEvent_t`, `CUDAStream_t`, `CUDAGraph_t`, `CUDAGraphExec_t`, `CUDAGraphNode_t`, `CUDART_GRAPHS_MIN_VERSION`

And `_compat/cuda_ipc/cuda_graphs.py` is present as a sibling for `CUDAGraphsMixin`.

The sibling-relative import pattern is **already established in this package**:

- `cuda_ipc_importer.py:108-109` — `from .cuda_ipc_wrapper import ...` + `from .cuda_runtime_types import ...`
- `cuda_ipc_exporter.py:61-62` — same pattern

So `cuda_ipc_wrapper.py` and `cuda_graphs.py` are the **only two stragglers** still using the broken `cuda_link.X` / `CUDARuntimeTypes` pattern.

### Why cozy-snacking missed this

That plan's "MCP verification" section read file contents (ctor signatures, line ranges). It did not execute the import chain. `from streamdiffusion._compat.cuda_ipc import CUDAIPCImporter` was never actually attempted in SD's venv before the commit landed — the static-content check looked clean.

> Process refinement: future plans relying on a `from <package> import <symbol>` line should verify the chain by running `python -c "from streamdiffusion._compat.cuda_ipc import CUDAIPCImporter"` in SD's venv during Phase 3 review — not just by reading line ranges. (Memory update candidate for `[[feedback_verify_plan_with_mcp]]`.)

## Fix — replace broken imports with relative ones

Match the convention already used by `cuda_ipc_importer.py:108-109` and `cuda_ipc_exporter.py:61-62`.

### Patch 1 — `src/streamdiffusion/_compat/cuda_ipc/cuda_ipc_wrapper.py`

Replace **lines 24-54** with:

```python
from .cuda_runtime_types import (  # noqa: E402
    CUDAError,
    CUDAEvent_t,
    CUDAGraph_t,
    CUDAGraphExec_t,
    CUDAGraphNode_t,
    CUDAStream_t,
    cudaIpcEventHandle_t,
    cudaIpcMemHandle_t,
    cudaMemcpy3DParms,
    cudaPointerAttributes,
)
from .cuda_graphs import CUDAGraphsMixin  # noqa: E402
```

### Patch 2 — `src/streamdiffusion/_compat/cuda_ipc/cuda_graphs.py`

Replace **lines 18-41** with:

```python
from .cuda_runtime_types import (  # noqa: E402
    CUDAEvent_t,
    CUDAGraph_t,
    CUDAGraphExec_t,
    CUDAGraphNode_t,
    CUDAStream_t,
    cudaExtent,
    cudaMemcpy3DParms,
    cudaPitchedPtr,
    cudaPos,
)
```

**No symbol changes.** All names already exist in `_compat/cuda_ipc/cuda_runtime_types.py` (verified above). The diff is `-26 +13` lines total.

### Why drop the try/except entirely (Option A) vs. add a third fallback (Option B)

- **Option A (recommended)**: hard-replace with relative imports. The `_compat/cuda_ipc/` directory is a sealed in-tree copy used only from SD's venv — its sibling files are the canonical source of these symbols here. Matches what `cuda_ipc_importer.py` / `cuda_ipc_exporter.py` already do.
- **Option B**: keep the try/except chain and add a third `from .cuda_runtime_types import ...` fallback. Preserves byte-similarity to upstream `cuda_link`, useful only if someone ever drops the upstream pip package into SD's venv. Adds 8 lines of dead defensive code for a path nobody uses today.

Recommendation: **Option A**. If the user later wants to install `cuda_link` as a real pip dep, the `_compat` copy can be deleted entirely at that point — the relative-import version doesn't need to coexist with the pip version.

## Verification

After applying both patches in the running SD venv (no rebuild needed — pure Python):

1. **Smoke-test the import chain** (Bash/PowerShell):

   ```powershell
   python -c "from streamdiffusion._compat.cuda_ipc import CUDAIPCImporter; print('OK', CUDAIPCImporter)"
   ```

   Must print `OK <class 'streamdiffusion._compat.cuda_ipc.cuda_ipc_importer.CUDAIPCImporter'>`. If it prints any traceback, the patch is wrong — stop and re-read the broken file.

2. **Relaunch the .toe.** Inspect SD cmd log for the Round 2 success criteria (from cozy-snacking lines 326-330):
   - ✅ `td_config.yaml` printout contains `use_cuda_ipc_input: true`.
   - ✅ NO `FileNotFoundError`.
   - ✅ NO `_get_input_frame: No module named 'CUDARuntimeTypes'`.
   - ✅ One-shot `CUDA IPC input ready: shm=StreamDiffusionTD_512-512_input_ipc` on the first frame.
   - ✅ TD textport `[CUDAIPCExtension:Receiver]` lines continue to confirm the output direction round-trips (this was already working in Round 1 commit `4c2a742`).

3. **Round-trip visual check**: TD's Receiver COMP should show the SD-processed frames moving (not a frozen first frame). If the BGRA→RGB conversion at `td_manager.py:687-688` is off, colors will be swapped — that's a separate Round-3 fix, not part of this diagnosis.

## Commit

Both files are tracked (`src/streamdiffusion/_compat/cuda_ipc/`), so this lands as a normal commit on `feat/cuda-ipc-output`, on top of `72dc7cc`.

```powershell
./scripts/git/commit_enhanced.sh --no-venv `
  "fix: use relative imports in vendored _compat/cuda_ipc (CUDARuntimeTypes missing in SD venv)"
```

(Per `[[feedback_pr_branch_convention]]`, branch stays at `feat/cuda-ipc-output`; PR target is `SDTD_031_dev`.)

## Critical files (this diagnosis only)

| File | Lines | Change |
|---|---|---|
| `StreamDiffusion/src/streamdiffusion/_compat/cuda_ipc/cuda_ipc_wrapper.py` | 24-54 | replace 2× try/except with 2× relative import (`-26 +13`) |
| `StreamDiffusion/src/streamdiffusion/_compat/cuda_ipc/cuda_graphs.py` | 18-41 | replace 1× try/except with 1× relative import (`-23 +10`) |

Reused verbatim (no edits):

- `StreamDiffusion/src/streamdiffusion/_compat/cuda_ipc/cuda_runtime_types.py` — already exports all required symbols
- `StreamDiffusion/src/streamdiffusion/_compat/cuda_ipc/__init__.py` — import order is fine
- `StreamDiffusion/src/streamdiffusion/_compat/cuda_ipc/cuda_ipc_importer.py` — already uses `.cuda_ipc_wrapper` / `.cuda_runtime_types` correctly
- `StreamDiffusion/src/streamdiffusion/_compat/cuda_ipc/cuda_ipc_exporter.py` — same

## Out of scope

- BGRA→RGB conversion correctness (`td_manager.py:687-688` `clamp(0,1)*255 → uint8`). Round-3 if user reports off colors.
- True zero-copy GPU input (skip the `.cpu().numpy()` D2H). Deferred per cozy-snacking Round 1 "Out of scope".
- Installing `cuda_link` as a real pip package and removing the vendored copy. Larger refactor; not needed for the fix.
- Saving this plan into `StreamDiffusion/_plans/2026-05-17_diagnose-cudaruntimetypes-import.md` per `[[feedback_save_plans_as_project_files]]` — will copy on exit from plan mode (plan-mode editor only permits the assigned file in `~/.claude/plans/`).
