# Wire CUDA IPC input direction (TD → SD), fix FileNotFoundError on legacy CPU SHM

## Context

After committing the output-direction IPC (commit `4c2a742` on `feat/cuda-ipc-output`), launching the .toe now crashes SD with:

```
FileNotFoundError: [WinError 2] The system cannot find the file specified: 'StreamDiffusionTD_512-512'
  at td_manager.py:331  →  self.input_memory = shared_memory.SharedMemory(name=self.input_mem_name)
```

The TD textport log shows TD has switched its **input** to a cuda-link **Sender**:

- `[CUDAIPCExtension:Sender] Created new SharedMemory: StreamDiffusionTD_512-512_input_ipc (433 bytes)`
- TD now writes input frames as zero-copy GPU IPC (3 slots, 4 MB each, 512×512 **float32 4ch**, ~313–624 µs/frame)
- TD no longer creates the legacy CPU SharedMemory `StreamDiffusionTD_512-512`, so SD's open call fails

The deferred input direction is now required. SD must read input via `CUDAIPCImporter` (vendored at `_compat/cuda_ipc/cuda_ipc_importer.py`) when the toggle is on, and skip the legacy CPU SHM open.

User's authoritative YAML already has the SHM-name reserved:

```yaml
td_settings:
  input_mem_name: 'StreamDiffusionTD_512-512'                  # legacy (unused when IPC input on)
  cuda_ipc_input_shm_name: 'StreamDiffusionTD_512-512_input_ipc'   # TD Sender writes here
```

Missing: a top-level `use_cuda_ipc_input: true` toggle and the SD-side Importer wiring.

## Target end state

- New top-level YAML toggle `use_cuda_ipc_input: true|false` (parallel to `use_cuda_ipc_output`).
- When `use_cuda_ipc_input: true`, SD skips the legacy CPU SHM input open and reads frames via a lazy-initialized `CUDAIPCImporter` bound to `td_settings.cuda_ipc_input_shm_name`.
- First-connect noise (the `traceback.print_exc()` at `cuda_ipc_importer.py:810` when SHM is missing) is sidestepped by a cheap pre-probe via `multiprocessing.shared_memory.SharedMemory(name=...)` — Importer is only constructed once the probe confirms TD's Sender has created the SHM header.
- `_get_input_frame` returns a numpy HWC uint8 RGB array compatible with the existing streaming-loop contract (L513–514 then does `astype(float32) / 255.0`). TD's wire is HWC float32 BGRA → convert on GPU (drop alpha, swap B↔R, scale to [0,255] uint8) → `.cpu().numpy()`. Keeps the streaming-loop contract unchanged; defers full zero-copy GPU pipeline to a future round.
- Crash is gone. With `use_cuda_ipc_input: true`, SD reads TD's GPU IPC frames and feeds them through the existing img2img path.
- A second commit lands on `feat/cuda-ipc-output` adding only the SD-side input wiring (TD-side files remain gitignored).

## Execution (in order)

All paths relative to `D:/dev/SD_3_0_1/test_Install_dev/StreamDiffusion/StreamDiffusion/`.

### Step 1 — Add `use_cuda_ipc_input` toggle to YAML

**`StreamDiffusionTD/td_config.yaml`** (gitignored — user must edit, or we set it):
Add at top level, next to `use_cuda_ipc_output`:

```yaml
use_cuda_ipc_input: true
```

**`configs/td_config.yaml.example`** (tracked):
Add the same key with `false` default, next to `use_cuda_ipc_output: false`:

```yaml
# CUDA IPC zero-copy GPU-to-GPU output (SD→TD via cuda-link)
use_cuda_ipc_output: false
cuda_ipc_shm_name: 'StreamDiffusionTD_512-512_output_ipc'
cuda_ipc_num_slots: 3
output_type: 'np'

# CUDA IPC zero-copy GPU-to-GPU input (TD→SD via cuda-link)
# When true, SD reads input frames from td_settings.cuda_ipc_input_shm_name
# instead of the legacy CPU SharedMemory at td_settings.input_mem_name.
use_cuda_ipc_input: false
```

### Step 2 — Wire toggle + Importer state in `td_manager.py.__init__`

`StreamDiffusionTD/td_manager.py` (gitignored, runtime fix). Near the existing `self.use_cuda_ipc_output` at L62:

```python
self.use_cuda_ipc_output = self.config.get('use_cuda_ipc_output', False)
self.use_cuda_ipc_input = self.config.get('use_cuda_ipc_input', False)
self.cuda_ipc_input_shm_name = self.td_settings.get('cuda_ipc_input_shm_name')
self._cuda_ipc_importer = None  # lazy-init on first frame
```

### Step 3 — Skip legacy CPU SHM open when IPC input is on

`_initialize_memory_interfaces` around L331 — wrap the input SHM open in the same guard pattern used for the output side:

```python
# Input memory (from TouchDesigner) — skip when CUDA IPC input is active
if not self.use_cuda_ipc_input:
    self.input_memory = shared_memory.SharedMemory(name=self.input_mem_name)
    logger.debug(f"Connected to input SharedMemory: {self.input_mem_name}")
else:
    self.input_memory = None
    logger.debug(f"CUDA IPC input active; legacy SharedMemory skipped (will read {self.cuda_ipc_input_shm_name})")
```

This single guard fixes the `FileNotFoundError` crash.

### Step 4 — Add IPC-aware fast-path in `_get_input_frame`

Replace the body of `_get_input_frame` (L628–644) with a branch-on-toggle:

```python
def _get_input_frame(self) -> Optional[np.ndarray]:
    """Get input frame from TouchDesigner (platform-specific)"""
    try:
        if self.use_cuda_ipc_input:
            return self._get_input_frame_cuda_ipc()
        if self.is_macos and self.syphon_handler:
            return self.syphon_handler.capture_input_frame()
        if self.input_memory:
            width = self.config['width']
            height = self.config['height']
            frame = np.ndarray((height, width, 3), dtype=np.uint8, buffer=self.input_memory.buf)
            return frame.copy()
        return None
    except Exception as e:
        if self.debug_mode:
            logger.debug(f"_get_input_frame: {e}")
        return None
```

Add the new IPC helper alongside (right after `_get_input_frame`):

```python
def _get_input_frame_cuda_ipc(self) -> Optional[np.ndarray]:
    """Read one frame from TD's CUDAIPCExporter (Sender). Returns HWC uint8 RGB,
    matching the legacy CPU SHM contract so the streaming loop is unchanged."""
    # Lazy-construct the Importer once TD's Sender SHM exists.
    if self._cuda_ipc_importer is None:
        if not self._probe_ipc_input_shm():
            return None  # TD Sender not active yet — retry next tick
        from streamdiffusion._compat.cuda_ipc import CUDAIPCImporter
        try:
            self._cuda_ipc_importer = CUDAIPCImporter(
                shm_name=self.cuda_ipc_input_shm_name,
                debug=False,
            )
        except Exception as e:
            logger.warning(f"CUDAIPCImporter init failed: {e}")
            self._cuda_ipc_importer = None
            return None
        if not self._cuda_ipc_importer.is_ready():
            # init silently failed (e.g. magic mismatch); drop and retry next tick
            self._cuda_ipc_importer = None
            return None
        logger.info(f"CUDA IPC input ready: shm={self.cuda_ipc_input_shm_name}")

    # TD wire: HWC float32 BGRA on GPU. Convert to HWC uint8 RGB to match
    # streaming-loop contract (L513–514 expects uint8 → float32 / 255.0).
    gpu_frame = self._cuda_ipc_importer.get_frame()  # zero-copy torch.Tensor on GPU
    if gpu_frame is None:
        return None
    rgb = gpu_frame[..., [2, 1, 0]].contiguous()      # BGRA → RGB (drop alpha)
    rgb_u8 = (rgb.clamp(0, 1) * 255).to(torch.uint8)  # float [0,1] → uint8 [0,255]
    return rgb_u8.cpu().numpy()                       # D2H to match existing contract
```

### Step 5 — Add cheap SHM-existence probe

Add as a sibling method (suppresses the noisy `traceback.print_exc()` at `cuda_ipc_importer.py:810` by only constructing the Importer when the SHM segment exists):

```python
def _probe_ipc_input_shm(self) -> bool:
    """Return True iff TD has created the input IPC SharedMemory."""
    if not self.cuda_ipc_input_shm_name:
        return False
    try:
        from multiprocessing.shared_memory import SharedMemory
        shm = SharedMemory(name=self.cuda_ipc_input_shm_name)
        shm.close()
        return True
    except FileNotFoundError:
        return False
    except Exception:
        return False
```

This is the proven solution to the first-connect noise that broke the abandoned input-perspective plan.

### Step 6 — Importer cleanup

In `_cleanup_memory_interfaces` (around L387, after the existing `cleanup_cuda_ipc` for the Exporter):

```python
if self._cuda_ipc_importer is not None:
    try:
        self._cuda_ipc_importer.cleanup()
    except Exception:
        pass
    self._cuda_ipc_importer = None
```

### Step 7 — Runtime verification

1. Set `use_cuda_ipc_input: true` (and `use_cuda_ipc_output: true`) in `StreamDiffusionTD/td_config.yaml`.
2. Launch .toe. Confirm no `FileNotFoundError` on startup.
3. Watch for `CUDA IPC input ready: shm=StreamDiffusionTD_512-512_input_ipc` log line on first frame.
4. Confirm SD output appears in TD's Receiver COMP (round-trip works: TD Sender → SD Importer → wrapper → SD Exporter → TD Receiver).
5. Toggle `use_cuda_ipc_input: false`, restart, confirm legacy CPU SHM path still works (regression — assumes TD COMP is switched back to non-Sender mode). If TD is still in Sender mode the legacy path will fail to open SHM — that's expected; document as "TD-side mode must match SD-side toggle."

### Step 8 — Commit on `feat/cuda-ipc-output`

```bash
git add configs/td_config.yaml.example
./scripts/git/commit_enhanced.sh --no-venv --skip-lint \
  "feat: add CUDA IPC input direction via cuda-link (TD->SD zero-copy GPU transport)"
```

Only `configs/td_config.yaml.example` is tracked; the runtime files (`td_manager.py`, `td_config.yaml`) are gitignored. The commit is small by design — the IPC import wiring lives in the .tox binary (synced into the gitignored `StreamDiffusionTD/` dir per [[project_scripts_dir_purpose]]).

## Critical files & key references

**Tracked (committed)**:

- `configs/td_config.yaml.example` — add `use_cuda_ipc_input: false` toggle alongside existing `use_cuda_ipc_output`

**Gitignored (runtime fix only)**:

- `StreamDiffusionTD/td_config.yaml` — add `use_cuda_ipc_input: true` for the user's session
- `StreamDiffusionTD/td_manager.py`:
  - L62-area: add `use_cuda_ipc_input`, `cuda_ipc_input_shm_name`, `_cuda_ipc_importer` slots
  - L331: guard legacy CPU SHM input open with `if not self.use_cuda_ipc_input:`
  - L387-area (`_cleanup_memory_interfaces`): tear down Importer
  - L628-644 (`_get_input_frame`): IPC fast-path branch
  - Add new methods: `_get_input_frame_cuda_ipc`, `_probe_ipc_input_shm`

**Reused verbatim (no edits)**:

- `src/streamdiffusion/_compat/cuda_ipc/cuda_ipc_importer.py`:
  - ctor L495-515 (`shm_name`, `shape=None`, `dtype=None`, `debug`, `timeout_ms=5000`, `device=0` — auto-detects from SHM metadata)
  - `_initialize()` L783-811 (non-blocking; noisy `traceback.print_exc()` at L810 — sidestepped by Step 5 probe)
  - `get_frame()` L903 (returns zero-copy torch.Tensor on GPU, HWC, dtype from metadata)
  - `cleanup()` L1237-1266 (idempotent)
  - `is_ready()` L1272 (post-init state check)
- TD's wire format (from textport log): **HWC float32 4ch BGRA** at 512×512, 3 slots

## MCP verification (against current index)

Verified via `mcp__code-search__search_code` before finalizing this plan (per [[feedback_verify_plan_with_mcp]]):

- `CUDAIPCImporter.__init__` — confirmed at `cuda_ipc_importer.py:495-563`. Signature: `(shm_name="cudalink_output_ipc", shape=None, dtype=None, debug=False, timeout_ms=5000.0, device=0)`. Auto-detects shape/dtype from SHM metadata, so Step 4 passing only `shm_name` + `debug` is sufficient.
- `CUDAIPCImporter._initialize` — confirmed at `cuda_ipc_importer.py:783-811`. Non-blocking, single-shot. `traceback.print_exc()` at L810 is the noise the Step 5 probe sidesteps.
- `CUDAIPCImporter._open_and_validate_shm` — confirmed at `cuda_ipc_importer.py:628-685`. Catches `FileNotFoundError` and re-raises after logging; the probe avoids triggering this path entirely.
- `CUDAIPCImporter.get_frame` — confirmed at `cuda_ipc_importer.py:903-988`. Returns zero-copy `torch.Tensor` on GPU with shape/dtype from SHM metadata (matches Step 4's GPU-side BGRA→RGB→uint8 conversion).
- `CUDAIPCImporter.cleanup` — confirmed at `cuda_ipc_importer.py:1237-1266`. Idempotent (Step 6 cleanup is safe to call unconditionally).
- `CUDAIPCImporter.is_ready` — confirmed at `cuda_ipc_importer.py:1272`. Post-init state check (used in Step 4 to drop a half-initialized importer).
- `_get_input_frame` — confirmed at `StreamDiffusionTD/td_manager.py:628-644` (authoritative target). Returns HWC uint8 RGB numpy from `self.input_memory.buf` — Step 4's replacement preserves this contract.

**Prior art note** — `Scripts/streamdiffusionTD__Text__td_manager__td.py:717-738` contains an existing `_try_construct_ipc_importer` from earlier abandoned input-direction work, using an `_ipc_importer_cls` indirection + `is_ready()` guard. The Step 4/5 implementation borrows the same probe-then-construct + `is_ready` pattern but does not depend on Scripts/ (which may be stale per [[project_scripts_dir_purpose]]).

## Round 2 — Emitter fix (TD COMP overwrites yaml on launch)

### Symptom

After the SD-side wiring committed at `72dc7cc`, launching the .toe still crashes with the same `FileNotFoundError: ... 'StreamDiffusionTD_512-512'` at `td_manager.py:336` (the guarded SHM open). The guard is in place — but `self.use_cuda_ipc_input` evaluates to `False`.

### Root cause

The TD .tox COMP **regenerates `StreamDiffusionTD/td_config.yaml` on every launch** via its YAML emitter at:

```
D:/dev/SD_3_0_1/test_Install_dev/StreamDiffusion/Scripts/StreamDiffusionTD__Text__StreamDiffusionExt__td.py:3740-3774
```

(Scripts/ is at the parent dir of the repo, per [[project_scripts_dir_purpose]] — edits sync into the running .tox immediately.) The emitter currently writes these IPC keys:

- L3754 `use_cuda_ipc_output: {use_ipc}` ← driven by `Usecudaipcoutput` par, default `True`
- L3755 `cuda_ipc_shm_name: '{stream_name}_output_ipc'`
- L3756 `cuda_ipc_num_slots: 3`
- L3768 `cuda_ipc_input_shm_name: '{stream_name}_input_ipc'`

Missing: **no `use_cuda_ipc_input` write**. The yaml I edited manually had the line stripped on .toe launch. `self.config.get('use_cuda_ipc_input', False)` → `False` → guard inactive → crash.

The TD COMP is **hardwired to Sender mode** for the input direction (textport log confirms `[CUDAIPCExtension:Sender] Created new SharedMemory: StreamDiffusionTD_512-512_input_ipc`). So the emitter should mirror that and write `use_cuda_ipc_input: true`.

### Fix

One symmetric block added to the emitter, mirroring the existing output-direction pattern at L3747-3757. Insert immediately after L3757 (`output_type: 'np'`), before L3759 (`# TouchDesigner specific settings`):

```python
# Emit CUDA IPC INPUT setting — enable by default; override via Usecudaipcinput par if present
use_ipc_input = True
try:
    use_ipc_input = bool(self.ownerComp.par.Usecudaipcinput.eval())
except AttributeError:
    pass
yaml_content += '\n# CUDA IPC zero-copy GPU-to-GPU input (TD→SD via cuda-link)\n'
yaml_content += f'use_cuda_ipc_input: {str(use_ipc_input).lower()}\n'
```

Defaults to `True` (matches the .tox's hardwired Sender). If the user later adds a `Usecudaipcinput` parameter to the COMP, it'll be respected — symmetric with `Usecudaipcoutput`.

### Verification

1. Launch .toe. Confirm `td_config.yaml` is rewritten and now contains `use_cuda_ipc_input: true` near the existing `use_cuda_ipc_output: true`.
2. Confirm no `FileNotFoundError` on startup.
3. Watch for `CUDA IPC input ready: shm=StreamDiffusionTD_512-512_input_ipc` log on first frame.
4. Confirm round-trip: TD Sender → SD Importer → wrapper → SD Exporter → TD Receiver.

### Out of scope (Round 2)

- Independent TD parameter `Usecudaipcinput` — defer until user wants to toggle input direction separately from the hardwired Sender. Current default-`True` falls through if the parameter is absent, so adding it later is non-breaking.
- Committing the emitter change — Scripts/ is outside the git repo (parent dir per [[project_scripts_dir_purpose]]), so this is a runtime sync only. No git artifact.

### Critical files (Round 2)

- `D:/dev/SD_3_0_1/test_Install_dev/StreamDiffusion/Scripts/StreamDiffusionTD__Text__StreamDiffusionExt__td.py:3757-3759` — insert the 6-line emit block

## Out of scope (Round 1)

- True zero-copy GPU input path (skip the `.cpu().numpy()` D2H by feeding `wrapper.img2img` a GPU tensor directly). Requires touching the streaming loop's uint8→float conversion at L513-514 and the wrapper's img2img preprocessing — defer to a follow-up.
- Symphonous handling when TD toggles Sender mode at runtime (current behavior: SD detects via probe on next tick, lazily reinitializes; teardown on TD side currently triggers `SlotState.SHUTDOWN` in `_try_acquire`, importer auto-cleans).
- Pushing the branch / opening a PR — user-driven per [[feedback_pr_branch_convention]].

---

## Next session — log review handoff

**Status going in**: Both code changes are applied (SD-side committed at `72dc7cc` on branch `feat/cuda-ipc-output`; TD-side emitter edit at `Scripts/StreamDiffusionTD__Text__StreamDiffusionExt__td.py` post-L3757 is **uncommitted** since Scripts/ lives outside the git repo). The user is expected to relaunch the .toe between sessions, then return with two fresh logs.

### What to ask the user for

1. **SD cmd log** — full stdout from launching the .toe (the same channel that previously showed the `FileNotFoundError: ... 'StreamDiffusionTD_512-512'` crash).
2. **TD textport log** — full TD console output (the `[CUDAIPCExtension:Sender]` lines and any new `[CUDAIPCExtension:Receiver]` lines for the output direction).

### Success criteria — what the logs MUST show

**SD cmd log (must see)**:

- ✅ `td_config.yaml` printout near startup includes `use_cuda_ipc_input: true` (between `output_type: 'np'` and `# TouchDesigner specific settings`). If this line is missing, the emitter edit didn't sync into the .tox — check whether TD was restarted (the COMP re-emits on init, not on file change).
- ✅ NO `FileNotFoundError: ... 'StreamDiffusionTD_512-512'` traceback.
- ✅ One-shot `CUDA IPC input ready: shm=StreamDiffusionTD_512-512_input_ipc` log line on first frame received.
- ✅ Streaming loop continues — frame timing logs, no repeated `_get_input_frame` exceptions.

**TD textport log (must see)**:

- ✅ Sender side already known-good: `Created new SharedMemory: StreamDiffusionTD_512-512_input_ipc` + `FIRST FRAME: ...` + steady-state `Frame N: slot X, ... GPU memcpy=...us`.
- ✅ Receiver side activates: look for `[CUDAIPCExtension:Receiver] ...` lines confirming the output direction round-trips. (Round 1 already wired SD→TD output, so this should match commit `4c2a742`'s behavior.)

### Common failure modes to triage (in order of likelihood)

1. **Emitter didn't sync** — yaml printout still missing `use_cuda_ipc_input`. Confirm Scripts/ edit landed (`grep -n use_cuda_ipc_input D:/dev/SD_3_0_1/test_Install_dev/StreamDiffusion/Scripts/StreamDiffusionTD__Text__StreamDiffusionExt__td.py`). If present in Scripts/ but not in regenerated yaml, the .toe may need a full close+reopen (not just re-init) to pick up Scripts/ changes — verify per [[project_scripts_dir_purpose]].
2. **Importer init fails** — see `CUDAIPCImporter init failed: ...` warn in SD log. Likely cause: SHM magic mismatch or shape mismatch between vendored `cuda_ipc_importer.py` (`PROTOCOL_MAGIC = 0x43495044` "CIPD") and the TD-side cuda-link v1.4.1 emitter. Verify the magic with `grep -n PROTOCOL_MAGIC D:/dev/SD_3_0_1/test_Install_dev/StreamDiffusion/StreamDiffusion/src/streamdiffusion/_compat/cuda_ipc/cuda_ipc_importer.py` and confirm TD Sender wrote the same.
3. **Probe never succeeds** — SD silently returns None every frame, never logs "CUDA IPC input ready". Means `_probe_ipc_input_shm` keeps hitting `FileNotFoundError`. Could be a Windows SHM name-collision quirk (the legacy CPU SHM `StreamDiffusionTD_512-512` and IPC SHM `StreamDiffusionTD_512-512_input_ipc` are different names — should not collide, but worth verifying TD Sender textport log shows the `_input_ipc` SHM actually created).
4. **BGRA→RGB conversion artifacts** — frames flow but output looks wrong (color-swapped or alpha leak). The Round 1 conversion at `td_manager.py:687-688` (`gpu_frame[..., [2, 1, 0]].contiguous()` + `clamp(0,1) * 255 → uint8`) assumes TD writes float32 BGRA in [0,1]. Textport log confirms `512x512 float32 4ch` — but if values are out of [0,1] range (TD's sRGB pipeline can produce >1 in HDR), the `clamp(0, 1)` would crush highlights. Worth visual-comparison test if user reports off colors.

### Next agent's first move

```
1. Read the SD log + TD log the user pastes
2. Grep yaml printout for `use_cuda_ipc_input` to confirm emitter fix landed
3. Grep for `CUDA IPC input ready` + `FileNotFoundError`
4. If happy path → commit the emitter change is NOT possible (Scripts/ is outside the repo); instead acknowledge round-trip success and ask user whether to merge `feat/cuda-ipc-output` → `SDTD_031_dev` per [[feedback_pr_branch_convention]]
5. If failure → triage per "Common failure modes" above before touching code
```

### Reference: what's committed vs uncommitted

| File | Tracked? | Committed? | Notes |
|---|---|---|---|
| `configs/td_config.yaml.example` | yes | `72dc7cc` | adds `use_cuda_ipc_input: false` default |
| `StreamDiffusionTD/td_config.yaml` | gitignored | no | regenerated by TD on launch; manual edits clobbered |
| `StreamDiffusionTD/td_manager.py` | gitignored | no | runtime fix: guard, helpers, cleanup, probe |
| `Scripts/StreamDiffusionTD__Text__StreamDiffusionExt__td.py` | **outside repo** | n/a | Round 2 emitter fix (1 block post-L3757); no git artifact possible |
