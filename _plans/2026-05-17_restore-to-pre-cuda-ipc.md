# Restore branch to "Quantization landed, CUDA IPC just started"

## Context

The current branch `feat/quantization-robustness` (HEAD `1a8065f`, 9 ahead of `origin/SDTD_031_dev`) carries an in-progress CUDA IPC integration in the working tree. The previous attempt (`_plans/drifting-twirling-tulip.md`) adopted a **StreamDiffusion-perspective** naming convention: cuda-link's `Sender`/`Receiver` modes were aliased to SD-side `input`/`output` directions, and YAML keys like `use_cuda_ipc_output` / `cuda_ipc_input_shm_name` were introduced. Phases 2.1–2.3 landed (Python→TD output working at 16–25 FPS), but Phase 2.4 (input direction first-connect retry) ran into a `traceback.print_exc()` noise issue in upstream `cuda_ipc_importer.py:809` that the manager-side probe was working around.

User has decided to **abandon this perspective** and redo the integration with cuda-link's vocabulary kept verbatim (canonical names per cuda-link 1.4.1 @ `92989fc`: `ExportBuffer`/`ImportBuffer` TOPs, `Sender`/`Receiver` modes, `CUDAIPCExporter`/`CUDAIPCImporter` classes, `export_frame`/`import_frame` methods, `Ipcmemname` param). The rename direction flips: instead of mapping cuda-link → SD-side names, rename SD-side names → cuda-link.

This plan covers **only the restore**. The naming-flip redo is a separate task.

## Target end state

- HEAD unchanged at `1a8065f` (post-Quantization, robustness fixes preserved).
- `git status` shows **clean working tree** for tracked files. Three diffs reverted: `configs/td_config.yaml.example`, `src/streamdiffusion/config.py`, `src/streamdiffusion/wrapper.py`.
- Untracked vendored sources **preserved as-is** (per user choice "Keep vendored, drop only glue"):
  - `src/streamdiffusion/_compat/cuda_ipc/` (10 files, cuda-link @ `92989fc`, VENDORED_VERSION.txt intact)
  - `src/streamdiffusion/_compat/td_exporter/` (TD-side vendored, no `__init__.py` by design)
- Scripts/ and StreamDiffusionTD/ files cleaned of all IPC integration glue (TouchDesignerManager.ipc_input_importer, `_try_construct_ipc_importer`, cuda_ipc_input_shm_name handling, ExtensionExt YAML emit lines).
- Safety net in place: `git stash` entry + archive branch `archive/cuda-ipc-input-perspective` pointing at current HEAD with all WIP tree captured.
- Abandoned plan archived under `_plans/archive/` with header note.

## Execution (in order)

All paths relative to `D:/dev/SD_3_0_1/test_Install_dev/StreamDiffusion/StreamDiffusion/` unless absolute.

### Step 1 — Safety net first

```bash
# (1a) Create archive branch tag at current HEAD (no checkout — stays on feat/quantization-robustness)
git branch archive/cuda-ipc-input-perspective HEAD

# (1b) Stash everything including untracked, with a descriptive message
git stash push -u -m "cuda-ipc-input-perspective WIP 2026-05-17 — before naming-flip restore"

# (1c) Verify stash captured all of: 3 tracked diffs + _compat/cuda_ipc/ + _compat/td_exporter/ + _plans/
git stash show -u stash@{0} --stat
```

Result: stash entry exists, archive branch points at `1a8065f`, working tree is **already clean** for tracked files post-stash. The untracked dirs are also in the stash and will be gone from the working tree.

### Step 2 — Restore vendored sources from the stash (cherry-pick the keepers)

`git stash apply` would re-apply everything (including the glue we want gone). Instead, restore only the vendored sources directly from the stash:

```bash
# Restore the two vendored dirs from stash@{0} (untracked portion)
git checkout stash@{0}^3 -- src/streamdiffusion/_compat/cuda_ipc/
git checkout stash@{0}^3 -- src/streamdiffusion/_compat/td_exporter/
```

Note: `stash@{0}^3` is the untracked-files commit inside a `git stash -u` stash. Verify with `git log --oneline stash@{0}^3` before running — if the stash layout differs (some git versions), use `git stash show -u --name-only stash@{0}` and then `git restore --source=stash@{0} --staged --worktree -- <path>` instead.

Verify VENDORED_VERSION.txt files still show `head_commit: 92989fc` and `vendored: 2026-05-17`.

### Step 3 — Surgical removal of TD-side IPC glue

Per user choice. Edit both `Scripts/` (source-of-truth, feeds .tox at runtime per [[project_scripts_dir_purpose]]) **and** `StreamDiffusionTD/` (gitignored mirror — same bytes, fed back into the .tox export).

**Critical**: `diff -q` confirmed Scripts/ and StreamDiffusionTD/ copies are byte-identical (48713 bytes). Edit Scripts/ first, then copy to StreamDiffusionTD/.

#### Edit set A — `Scripts/streamdiffusionTD__Text__td_manager__td.py` (48713 bytes)

Remove these exact regions (line numbers as of current working tree):
- **L84, L88**: instance-attr declarations `self.ipc_input_importer = None`, `self._ipc_importer_cls = None`, and any sibling pending-name attr (`_pending_ipc_input_name`, `_ipc_input_connected_logged`) — sweep the `__init__` body for any `ipc_`/`_ipc_` attr and remove.
- **L336–359**: the entire `if ipc_input_name := self.td_settings.get('cuda_ipc_input_shm_name'):` block inside `_initialize_memory_interfaces` (the probe-construct-fallback logic). Restore the pre-existing CPU SHM fallback to be the only input path.
- **L424–429**: the `if self.ipc_input_importer is not None: ... cleanup()` block in the cleanup path.
- **L672–683**: the IPC retry probe + `get_frame_numpy()` path inside `_get_input_frame`. Restore plain CPU SHM read.
- **L703–740**: the `_try_construct_ipc_importer` method definition + any `_probe_ipc_shm_exists` helper. Delete wholesale.

Verification grep after edits: `grep -nE "(ipc_input|cuda_ipc_input|_try_construct_ipc|_probe_ipc_shm|CUDAIPCImporter)" Scripts/streamdiffusionTD__Text__td_manager__td.py` must return zero hits.

#### Edit set B — `Scripts/StreamDiffusionTD__Text__StreamDiffusionExt__td.py`

Per Explore findings:
- **L3754–L3756**: the three writes for `use_cuda_ipc_output`, `cuda_ipc_shm_name`, `cuda_ipc_num_slots` in the YAML emitter. Remove all three.
- **L3768**: the `cuda_ipc_input_shm_name` write. Remove. (L3767 `input_mem_name` and L3769 `output_mem_name` are pre-existing — keep them.)

Verification grep: `grep -nE "(use_cuda_ipc_output|cuda_ipc_shm_name|cuda_ipc_num_slots|cuda_ipc_input_shm_name)" Scripts/StreamDiffusionTD__Text__StreamDiffusionExt__td.py` must return zero hits.

#### Edit set C — Sync to StreamDiffusionTD/

```bash
# After Scripts/ edits are complete:
cp Scripts/streamdiffusionTD__Text__td_manager__td.py StreamDiffusion/StreamDiffusionTD/td_manager.py
cp Scripts/StreamDiffusionTD__Text__StreamDiffusionExt__td.py StreamDiffusion/StreamDiffusionTD/StreamDiffusionExt.py  # verify exact filename
# Path-from-parent for the cp source: D:/dev/SD_3_0_1/test_Install_dev/StreamDiffusion/Scripts/
# Path-from-parent for the cp dest:   D:/dev/SD_3_0_1/test_Install_dev/StreamDiffusion/StreamDiffusion/StreamDiffusionTD/
```

Note: the StreamDiffusionTD/ filenames may differ from Scripts/ names (Scripts/ uses TD's `<comp>__<optype>__<dat>__td.py` convention; StreamDiffusionTD/ uses bare module names). Confirm exact mapping via `ls StreamDiffusion/StreamDiffusionTD/*.py` before cp.

### Step 4 — Archive the abandoned plan

```bash
mkdir -p _plans/archive
git mv _plans/drifting-twirling-tulip.md _plans/archive/2026-05-17_cuda-ipc-input-perspective_ABANDONED.md 2>/dev/null || mv _plans/drifting-twirling-tulip.md _plans/archive/2026-05-17_cuda-ipc-input-perspective_ABANDONED.md
```

Prepend this header to the archived file:

```markdown
> **ABANDONED 2026-05-17.** This plan used SD-perspective naming (cuda-link Sender→SD input, Receiver→SD output, YAML `use_cuda_ipc_output` etc.). The integration is being redone with cuda-link's vocabulary kept verbatim and SD-side names renamed instead. Kept as historical reference for the working Phase 2.1 BGRA repack approach and the Phase 2.4 SHM-probe-before-construct trick.

```

### Step 5 — Final verification

```bash
git status                                     # → only _plans/archive/ should appear (untracked)
git diff --stat                                # → empty (no tracked-file diffs)
ls src/streamdiffusion/_compat/cuda_ipc/       # → 10 files + VENDORED_VERSION.txt
ls src/streamdiffusion/_compat/td_exporter/    # → vendored TD scripts still present
grep -rn "cuda_ipc_input_shm_name\|ipc_input_importer\|_try_construct_ipc_importer\|use_cuda_ipc_output" Scripts/ StreamDiffusion/StreamDiffusionTD/ 2>&1
# → zero hits

# Confirm restored state is loadable
cd StreamDiffusion && python -c "from streamdiffusion.wrapper import StreamDiffusionWrapper; print('import OK')"
# → "import OK" (the wrapper.py revert dropped the IPC code paths cleanly)
```

Then a TD smoke test: open `StreamDiffusionTD_dev.toe` and confirm the legacy CPU SHM path (`input_mem_name`/`output_mem_name`) still drives a frame end-to-end. No CUDA IPC paths active.

## Critical files & key references

**Modified (revert via stash mechanism):**
- `src/streamdiffusion/wrapper.py` — L134-136 ctor kwargs, L320-321 + L336-338 instance slots, L931-947 IPC fast-path inside `postprocess_image`, L995-1009 `_ipc_pack_rgba`, L1011-1040 `_lazy_init_ipc_exporter`, L2658-2666 cleanup. All from the WIP diff.
- `src/streamdiffusion/config.py` — L160-162 three new `param_map` entries.
- `configs/td_config.yaml.example` — L85-92 new YAML keys.

**Kept as-is (vendored, preserved):**
- `src/streamdiffusion/_compat/cuda_ipc/` — cuda-link Python sources, untouched upstream copy. The `CUDAIPCExporter` (`cuda_ipc_exporter.py:191-1163`) and `CUDAIPCImporter` (`cuda_ipc_importer.py:479-`) classes will be the integration surface in the redo.
- `src/streamdiffusion/_compat/td_exporter/` — cuda-link TD-side vendored scripts. `TDSender.py:70` defines `_EXPORT_BUFFER_NAME = "ExportBuffer"`, `CUDAIPCExtension.py:169` defines `def import_frame(self, import_buffer: TOP)`. These names are canonical and will be adopted on the SD side in the redo.

**Edited surgically (per Step 3):**
- `Scripts/streamdiffusionTD__Text__td_manager__td.py` (== `StreamDiffusionTD/td_manager.py`, gitignored mirror)
- `Scripts/StreamDiffusionTD__Text__StreamDiffusionExt__td.py` (== gitignored mirror)

**Archived:**
- `_plans/drifting-twirling-tulip.md` → `_plans/archive/2026-05-17_cuda-ipc-input-perspective_ABANDONED.md`

## Next-task sketch (NOT executed in this plan — for continuity only)

The redo will rename SD-side concepts to cuda-link vocabulary. Concrete mapping derived from MCP search of canonical names (cuda-link `92989fc`, README L61, `CUDAIPCExtension.py` L98, `TDSender.py:70`, `TDReceiver.py:313`):

| Current SD-side name | Cuda-link canonical | Blast radius |
|---|---|---|
| `td_settings.input_mem_name` (YAML) | `td_settings.export_buffer_shm_name` (or just `Ipcmemname` per TD COMP convention) | HIGH — in every user's local `td_config.yaml` |
| `td_settings.output_mem_name` | `td_settings.import_buffer_shm_name` | HIGH — same |
| `TouchDesignerManager.__init__(input_mem_name, output_mem_name, ...)` | `TouchDesignerManager.__init__(export_shm_name, import_shm_name, ...)` | MED — only `td_main.py:399-402` calls positionally |
| `image` param on `wrapper.__call__/img2img` | **KEEP** — `image` is a general PyTorch convention, not a TD-IPC concept | n/a |
| `image_tensor` locals | **KEEP** — same reasoning | n/a |
| `_process_skip_diffusion` `preprocessor_input/output` locals | **KEEP** — these are preprocessor I/O, not TD-IPC | n/a |

The interesting rename surface is the **TD↔SD transport boundary** (memory names, mode enum, helper class names) — not the internal PyTorch tensor flow. Most of the wrapper.py rename pressure dissolves once we see that `image`/`image_tensor` are PyTorch idiom, orthogonal to cuda-link.

Memory note per [[feedback_save_plans_as_project_files]]: after ExitPlanMode and on user approval, copy this plan to `D:/dev/SD_3_0_1/test_Install_dev/StreamDiffusion/StreamDiffusion/_plans/2026-05-17_restore-to-pre-cuda-ipc.md` as the project-tracked copy.
