> **ABANDONED 2026-05-17.** This plan used SD-perspective naming (cuda-link Sender→SD input, Receiver→SD output, YAML `use_cuda_ipc_output` etc.). The integration is being redone with cuda-link's vocabulary kept verbatim and SD-side names renamed instead. Kept as historical reference for the working Phase 2.1 BGRA repack approach and the Phase 2.4 SHM-probe-before-construct trick.

# Replace SDTD shmem* COMPs with upstream cuda-link `CUDAIPCLink_v1.4.1.tox`

## Status (2026-05-17 16:22, Phase 2.4 round 2 — probe-before-construct)

**Phase 2.1 OK** — Python→TD output transport (cuda-link Receiver) confirmed working at 16-25 FPS.
**Phase 2.2 OK** — BGRA byte-swap in `wrapper.py:_ipc_pack_rgba` landed; colors correct.
**Phase 2.3 OK** — `consume_pending_resolution()` added to `Scripts/shmem__Text__output_callbacks__td.py`.
**Phase 2.4 round 1** (logger silencing + retry) — landed but **insufficient**. The error log lines from `cuda_ipc_importer` are silenced correctly, BUT `cuda_ipc_importer.py:809` calls `traceback.print_exc()` which writes to `sys.stderr` directly, bypassing the `logging` module entirely. `logger.setLevel(CRITICAL)` cannot suppress raw stderr writes.

Moving to **Phase 2.4 round 2: probe `SharedMemory` existence ourselves BEFORE invoking the importer's `_initialize()`** — that keeps the failing code path unreached during the normal startup race so `traceback.print_exc()` never fires.

---

## Phase 2.4 round 2 — Probe SHM existence before invoking importer

### What round 1 achieved + what it missed

Round 1 (logger silencing + 1/sec retry of `_initialize()`) landed and is **partially working**:

- ✅ `cuda_ipc_importer - ERROR - SharedMemory ... not found` lines: **suppressed** (logger filter at CRITICAL works).
- ✅ `CUDA IPC input configured (waiting for TD Sender): ...` log appears as planned.
- ❌ Raw Python traceback still printed on cold-start + on every retry tick.

### Root cause of the leftover noise

`StreamDiffusion/src/streamdiffusion/_compat/cuda_ipc/cuda_ipc_importer.py:807-810`:

```python
except (OSError, RuntimeError, ValueError, struct.error, IndexError) as e:
    logger.error("Initialization failed: %s", e)
    traceback.print_exc()   # ← writes directly to sys.stderr, bypassing logging
    return False
```

`traceback.print_exc()` is **not a logging call** — it writes formatted frames straight to `sys.stderr`. `logger.setLevel(CRITICAL)` filters records inside the `logging` module; it has zero effect on direct stderr writes. There is no public knob in `cuda_ipc_importer` that disables this behavior, and we don't modify `_compat/cuda_ipc/` (preserves upstream sync — same rationale documented earlier in this plan).

### Asymmetric upstream — verified

Searched `cuda_ipc_importer.py` for retry primitives:
- `_wait_for_slot()` (line 851) — for waiting on frame slots **after** connect (CPU poll on `query_event`)
- `_reinitialize()` (line 1163) — re-opens IPC handles, **requires `self._conn` non-None** (line 1166), unusable pre-connect
- No `reconnect`, `request_immediate_reconnect`, `retry`, `wait_for_*` first-connect primitive.

Upstream cuda-link's Python side appears designed primarily for Python-as-Sender; Python-as-Receiver (our case for TD→Python input) lacks first-connect retry. **The manager must drive retry itself.**

### Design — probe-before-construct

The importer's `__init__` eagerly calls `_initialize()` and the failing branch unconditionally calls `traceback.print_exc()`. So **we must avoid invoking the constructor (and `_initialize()`) until the SHM actually exists.**

Probe cost: a single `multiprocessing.shared_memory.SharedMemory(name=...).close()` round-trip. On a missing SHM that raises `FileNotFoundError` immediately (`OpenFileMapping → ERROR_FILE_NOT_FOUND`, microseconds). Since the probe happens in our manager code, **we own the try/except** — no `traceback.print_exc()` ever runs.

Lifecycle:

```
manager __init__:           _pending_ipc_input_name = None
                            ipc_input_importer = None

_initialize_memory_interfaces:
    if td_settings has cuda_ipc_input_shm_name:
        _pending_ipc_input_name = name
        if _probe_ipc_shm_exists(name):  ← silent SHM open/close probe
            _try_construct_ipc_importer()   ← only if probe says yes
        else:
            log "configured (waiting for TD Sender)"
    if _pending_ipc_input_name is None:
        open CPU SHM input fallback           ← gate flipped from `if importer is None`

_get_input_frame:
    if _pending_ipc_input_name and ipc_input_importer is None:
        every ≥1s:
            if _probe_ipc_shm_exists(name):
                _try_construct_ipc_importer()
                if importer ready: one-shot "connected" log
    if ipc_input_importer:  → get_frame_numpy() + alpha strip
    elif input_memory:      → CPU SHM read (legacy fallback path; only fires when IPC not configured)
    return None
```

### Edit set — `Scripts/streamdiffusionTD__Text__td_manager__td.py`

**Note:** Round 1 already applied three edits to this file. Round 2 **supersedes Edits 2 and 3** with the probe-first versions below. Edit 1 (slot declarations) stays; one slot is added.

**Edit 1 — extend slot declarations (~line 83-86):**

```python
self.syphon_handler = None
self.ipc_input_importer = None  # CUDAIPCImporter when cuda_ipc_input_shm_name is in td_settings
self._ipc_input_last_retry = 0.0  # monotonic timestamp of last reconnect attempt
self._ipc_input_connected_logged = False  # one-shot log when sender first comes online
self._pending_ipc_input_name: Optional[str] = None  # set when IPC configured but not yet connected
self._ipc_importer_cls = None  # cached CUDAIPCImporter class, populated lazily
```

**Edit 2 — replace round-1 IPC construction block in `_initialize_memory_interfaces()`:**

Replace the current `if ipc_input_name:` block AND change the CPU SHM fallback gate two lines below from `if self.ipc_input_importer is None:` to `if self._pending_ipc_input_name is None:`.

New block:

```python
ipc_input_name = self.td_settings.get('cuda_ipc_input_shm_name')
if ipc_input_name:
    try:
        from streamdiffusion._compat.cuda_ipc import CUDAIPCImporter
        self._ipc_importer_cls = CUDAIPCImporter
        self._pending_ipc_input_name = ipc_input_name
        # Probe SHM existence ourselves before letting the importer call _initialize().
        # _initialize()'s OSError catch unconditionally runs traceback.print_exc() (writes
        # to sys.stderr, bypassing the logging module) — so we must not invoke it on a
        # missing SHM. _probe_ipc_shm_exists() catches FileNotFoundError silently in OUR code.
        if self._probe_ipc_shm_exists(ipc_input_name):
            self._try_construct_ipc_importer()
            if self.ipc_input_importer is not None and self.ipc_input_importer.is_ready():
                logger.info(f"CUDA IPC input connected: {ipc_input_name}")
                self._ipc_input_connected_logged = True
        if self.ipc_input_importer is None:
            logger.info(
                f"CUDA IPC input configured (waiting for TD Sender): {ipc_input_name}"
            )
    except Exception as e:
        logger.warning(f"CUDA IPC input setup failed, falling back to CPU SHM: {e}")
        self._pending_ipc_input_name = None
        self.ipc_input_importer = None

if self._pending_ipc_input_name is None:
    # CPU SHM fallback — only when IPC is genuinely not configured
    self.input_memory = shared_memory.SharedMemory(name=self.input_mem_name)
    logger.debug(f"Connected to input SharedMemory: {self.input_mem_name}")
```

**Edit 3 — replace round-1 retry logic in `_get_input_frame()`:**

```python
# Lazy/throttled importer construction — fires when TD Sender comes online after Python startup
if self._pending_ipc_input_name is not None and self.ipc_input_importer is None:
    now = time.monotonic()
    if now - self._ipc_input_last_retry >= 1.0:
        self._ipc_input_last_retry = now
        if self._probe_ipc_shm_exists(self._pending_ipc_input_name):
            self._try_construct_ipc_importer()
            if self.ipc_input_importer is not None and not self._ipc_input_connected_logged:
                logger.info("CUDA IPC input connected (TD Sender came online)")
                self._ipc_input_connected_logged = True

if self.ipc_input_importer is not None:
    frame = self.ipc_input_importer.get_frame_numpy()
    if frame is not None and frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame[:, :, :3]  # strip alpha; downstream pipeline expects RGB
    return frame
```

If IPC is configured but not yet connected, this branch falls through; the existing `if self.input_memory:` check fires only when CPU SHM was opened (i.e., IPC was never configured). When IPC is configured but waiting for Sender, all three conditions are false → method returns `None`, pipeline pauses for that frame.

**Edit 4 — add two helper methods on the manager (place near other internal helpers):**

```python
def _probe_ipc_shm_exists(self, name: str) -> bool:
    """Return True iff the named Windows SharedMemory currently exists.

    Cheap probe (~µs on Windows). Used to gate CUDAIPCImporter construction:
    its __init__ eagerly calls _initialize(), whose except-OSError branch unconditionally
    invokes traceback.print_exc() — which writes to sys.stderr and cannot be silenced by
    logger configuration. Probing here keeps that code path unreached during the normal
    startup race with TD's Sender activation.
    """
    try:
        shm = shared_memory.SharedMemory(name=name)
        shm.close()
        return True
    except FileNotFoundError:
        return False
    except Exception:
        # Any other failure (permission, etc.) — treat as not-yet-available; retry will pick up.
        return False

def _try_construct_ipc_importer(self) -> None:
    """Construct CUDAIPCImporter (which calls _initialize() eagerly).

    Caller must have verified SHM existence via _probe_ipc_shm_exists first to avoid the
    traceback-printing failure path. On unexpected init failure (e.g., bad magic bytes,
    version mismatch), null out the importer so retry can probe again.
    """
    try:
        self.ipc_input_importer = self._ipc_importer_cls(
            shm_name=self._pending_ipc_input_name,
            device=torch.cuda.current_device(),
            timeout_ms=500.0,
        )
        if not self.ipc_input_importer.is_ready():
            logger.warning(
                f"CUDA IPC input '{self._pending_ipc_input_name}' opened but importer "
                f"not ready (protocol mismatch?). Will retry."
            )
            self.ipc_input_importer = None
    except Exception as e:
        logger.warning(f"CUDA IPC input construction failed: {e}")
        self.ipc_input_importer = None
```

### Why not modify `_compat/cuda_ipc/cuda_ipc_importer.py`?

Same rationale as round 1 — preserve upstream-sync compatibility. The cleanest upstream fix would be either (a) removing the `traceback.print_exc()` on line 809 (it's redundant with `logger.error` plus `logger.exception()` would do the right thing) or (b) gating it behind a debug flag. Both belong in an upstream PR, not as local fork divergence.

### Why throttle to 1/sec?

`shared_memory.SharedMemory(name=...)` on Windows is a single `OpenFileMapping` syscall — microseconds whether it succeeds or fails. 1/sec keeps the syscall rate trivially low while giving sub-second user-perceived activation latency (toggle Active=On in TD → frames flow within ≤1s).

### Verification

1. **Cold start** (Python before TD Sender activated):
   - Save edits in `Scripts/streamdiffusionTD__Text__td_manager__td.py`.
   - Trigger TD **Writeconfigs** (propagates edits into runtime `StreamDiffusionTD/td_manager.py`).
   - Restart Python with TD's `shmem_out` Sender **deactivated** (`Active = Off`).
   - **Expected logs:** `CUDA IPC input configured (waiting for TD Sender): StreamDiffusionTD_512-512_input_ipc`. **No** `Traceback`, no `cuda_ipc_importer - ERROR/WARNING/INFO` lines on startup. Streaming loop proceeds; `_get_input_frame` returns None per frame.

2. **Late-activation** (TD Sender comes online after Python startup):
   - Toggle `shmem_out.par.Active = On` in TD.
   - **Within ≤1s expect:** `CUDA IPC input connected (TD Sender came online)` (single line). Frames flow.

3. **Warm restart** (TD Sender already online when Python starts):
   - With Active=On in TD, restart Python.
   - **Expected:** `CUDA IPC input connected: StreamDiffusionTD_512-512_input_ipc` immediately at init. No "waiting" message. No traceback.

4. **Bounce test** (TD Sender Off→On→Off→On while Python connected): Out of scope — Phase 2.5. The probe is only consulted when importer is None; once connected, no probe runs.

### Rollback

Revert Edits 2, 3, and 4 in `Scripts/streamdiffusionTD__Text__td_manager__td.py` (Edit 1's slot declarations are harmless if left). To restore round-1 behavior instead, re-apply round-1 Edit 2/3 from this plan's git history.

### Open questions (defer to Phase 2.5)

1. **Reconnect after mid-stream disconnect.** If TD's Sender deactivates while Python is connected, `is_ready()` may still report True until something in the IPC layer notices. Need to detect stale handles and trigger `cleanup()` + null-out so the retry loop re-probes.
2. **Upstream contribution.** Removing `traceback.print_exc()` from `cuda_ipc_importer.py:809` (or gating it on a debug flag) would let downstreams use simple `_initialize()` retry without manager-side probing. File against upstream cuda-link.

---

## Phase 2.3 — Frozen frame on new shmem Receiver (current blocker)

### Symptom

After Phase 2.2 wrapper.py edit + Python restart: the new `shmem` Receiver's `output` Script TOP no longer animates. User saw the receiver path firing in Phase 2.1 (FPS, copyCUDAMemory logs); now it appears static.

User hypothesis was "Execute DAT uses old Mode names 'Sender'/'Receiver'" — **rejected**: `'Sender'`/`'Receiver'` (capitalized) ARE the canonical cuda-link names (`CUDAIPCExtension._mode`, set by `_normalize_mode` in `CUDAIPCExtension.py`). Both `Scripts/shmem__Execute__execute__td.py:29,31` and `Scripts/shmem_out__Execute__execute__td.py:29,31` use these correctly.

### Root cause: Scripts/ filename-collision overwrote new shmem's DATs

When the user duplicated `shmem_out` → renamed to `shmem`, TD's Scripts/ filename-convention auto-sync (where DAT contents are paired with `Scripts/<COMP-name>__<DAT-type>__<DAT-name>__td.py`) loaded the pre-existing OLD fork files for the `shmem__*` namespace into the new COMP's DATs. The new COMP did NOT keep the working content from `shmem_out__*`.

Evidence:
- `Scripts/shmem__Execute__execute__td.py` (4548 B, mtime May 17 09:34) — OLD fork content with DIAG-EXEC instrumentation, references `parent().par.Play.eval()` (line 89) and `parent().par.Mode.eval() == 'receive'` (line 67).
- `Scripts/shmem_out__Execute__execute__td.py` (2163 B, mtime May 17 00:30) — Clean cuda-link variant, no DIAG, smaller.
- `Scripts/shmem__Text__output_callbacks__td.py` (1142 B, mtime May 17 **12:41**) — modified TODAY, most recent. OLD fork content with DIAG-COOK instrumentation. **Missing `consume_pending_resolution()` call.**
- `Scripts/shmem_out__Text__output_callbacks__td.py` (1142 B, mtime Apr 22) — unrelated/legacy CPU-SHM output callbacks, untouched.

### Why "frozen frame" — the resolution-update bug

The OLD `shmem__Text__output_callbacks__td.py:12-19` onCook:

```python
def onCook(scriptOp):
    try:
        cuda_ext = parent().ext.CUDAIPCExtension
        if cuda_ext is not None and cuda_ext.mode == 'Receiver' and cuda_ext.is_active():
            # NOTE: resolution updates moved to Execute DAT to avoid re-cook race.
            cuda_ext.import_frame(scriptOp)
            return
    except Exception as e:
        ...  # dedup-log DIAG
```

**No `consume_pending_resolution()` call.** The comment "resolution updates moved to Execute DAT" assumes `modoutsidecook` is enabled on the Script TOP (TD 2025+). But the new `shmem`'s `output` Script TOP probably has `modoutsidecook` OFF (TD 2023 default).

Cross-reference Execute DAT path (`Scripts/shmem__Execute__execute__td.py:34-38`):

```python
if hasattr(import_buffer.par, 'modoutsidecook') and import_buffer.par.modoutsidecook.eval():
    cuda_ext.import_frame(import_buffer)
    cuda_ext.update_receiver_resolution(import_buffer)  # only fires when modoutsidecook=True
else:
    import_buffer.cook(force=True)  # triggers onCook — but onCook NEVER updates resolution
```

When `modoutsidecook` is OFF:
1. Execute DAT calls `import_buffer.cook(force=True)` → triggers `output.onCook`.
2. `onCook` calls `import_frame(scriptOp)` directly. Skips resolution update.
3. `TDReceiver.update_receiver_resolution` (`TDReceiver.py:523-548`) is **never** called.
4. The `output` Script TOP stays at default resolution (typically 1×1 or whatever it was duplicated as).
5. `import_frame` copies 512×512×4 GPU bytes into a TOP buffer that is **not** 512×512 → silent buffer-size mismatch, TD shows a stale frame.

Canonical `script_top_callbacks.py:31-43` (in `_compat/td_exporter/`) handles this path correctly — it calls `consume_pending_resolution()` and writes `outputresolution=9, resolutionw/h` to the scriptTop BEFORE `import_frame`.

### Edit

**Single edit:** patch `Scripts/shmem__Text__output_callbacks__td.py` to call `consume_pending_resolution()` inside the Receiver branch BEFORE `import_frame`. Preserve the existing DIAG-COOK instrumentation.

Replace the body of `onCook(scriptOp)` (lines 12-35) with:

```python
def onCook(scriptOp):
    # CUDA IPC Receiver path: GPU-to-GPU frame import via copyCUDAMemory
    try:
        cuda_ext = parent().ext.CUDAIPCExtension
        if cuda_ext is not None and cuda_ext.mode == 'Receiver' and cuda_ext.is_active():
            # Apply pending resolution update (TD 2023 path — fires when modoutsidecook is OFF).
            # When modoutsidecook is ON, Execute DAT already called update_receiver_resolution
            # and this returns None — harmless.
            pending = cuda_ext.consume_pending_resolution()
            if pending is not None:
                width, height = pending
                try:
                    scriptOp.par.outputresolution = 9  # Custom Resolution
                    scriptOp.par.resolutionw = width
                    scriptOp.par.resolutionh = height
                    cuda_ext._log(
                        f"[shmem onCook] Set output resolution to {width}x{height}",
                        force=True,
                    )
                except (AttributeError, RuntimeError) as e:
                    cuda_ext._log(f"[shmem onCook] Could not set resolution: {e}", force=True)
            cuda_ext.import_frame(scriptOp)
            return
    except Exception as e:
        # DIAG (Round-6, temporary): dedup-log exceptions from import_frame.
        key = type(e).__name__ + ':' + str(e)
        if not hasattr(onCook, '_diag_seen_errors'):
            onCook._diag_seen_errors = {}
        seen = onCook._diag_seen_errors.get(key, 0)
        onCook._diag_seen_errors[key] = seen + 1
        if seen == 0 or seen == 60 or seen == 600:
            try:
                parent().ext.CUDAIPCExtension._log(
                    f"[DIAG-COOK] onCook raised ({seen + 1}x): {key}",
                    force=True,
                )
            except Exception:
                print(f"[DIAG-COOK] onCook raised ({seen + 1}x): {key} (ext unavailable)")
```

API references verified in `CUDAIPCExtension.py:284-291` — `consume_pending_resolution()` returns `(width, height)` tuple if pending update is set, `None` otherwise. Safe to call repeatedly.

### Alternative considered (rejected)

**Enable `modoutsidecook` on the new shmem's `output` Script TOP** — would let the Execute DAT's `update_receiver_resolution` path drive resolution. Rejected because:
- TD 2023 doesn't have `modoutsidecook`. Forces a TD-version requirement.
- Existing `shmem_in_cn_processed` and other Receiver siblings in Phase 2.4+ may share the same TOP topology and would each need the same param flip. Fixing onCook covers all of them via the Scripts/ shared file.
- The canonical `_compat/td_exporter/script_top_callbacks.py` template uses `consume_pending_resolution()` in onCook — aligning with upstream is preferable.

### Why not also rewrite Execute DAT?

The current `Scripts/shmem__Execute__execute__td.py` has dead-code SHM references (`parent().par.Play.eval()`, `parent().par.Mode.eval()`). These would throw `AttributeError` on the new cuda-link COMP if `Play`/`Mode` parameters don't exist — BUT they're inside a `try/except: pass` block (lines 51-87), AND the user's Phase 2.1 logs confirmed the Receiver branch DID fire (`copyCUDAMemory=0.95ms`), so the Execute DAT IS reaching its Receiver path.

Deferred to Phase 2.5 (cleanup) — the dead-code SHM references are noise but not load-bearing for current symptoms. Fixing them is part of `Phase 3 (cleanup)`.

### Verification (Phase 2.3)

1. Save edit. Edit is to `Scripts/shmem__Text__output_callbacks__td.py` only.
2. TD picks up the change automatically (Scripts/ sync). No Writeconfigs needed (output_callbacks is a Text DAT, not the runtime `td_manager.py`).
3. Toggle the new `shmem.par.Active = Off → On` (forces Receiver re-init → `needs_resolution_update = True`).
4. **Textport log to look for:** `[shmem onCook] Set output resolution to 512x512` (single fire, first frame after Activate).
5. Visual check: frame updates (no longer static); colors look correct (red is red, blue is blue) — this validates Phase 2.2 BGRA fix simultaneously.
6. `Receiver's output Script TOP info` shows resolution `512×512` (was likely 1×1 or duplicated-source size before).

### Rollback

Revert `Scripts/shmem__Text__output_callbacks__td.py` to the prior 1142-byte version (current mtime May 17 12:41). No state to clean — `consume_pending_resolution` is idempotent.

### Open question (defer)

Whether the Scripts/ filename-collision also overrode `Scripts/shmem__Execute__execute__td.py` with OLD content vs the cuda-link variant. mtimes suggest yes (May 17 09:34 — recent). But the functional cuda-link branch in lines 51-72 of that file is intact, so it's currently working. Leave the file as-is for Phase 2.3 — revisit during Phase 3 cleanup.

---

## Phase 2.2 — Fix output channel order (RGBA → BGRA)

### Root cause

Three pieces of evidence converge on R↔B swap, not dtype:

1. `wrapper.py:995-1002` — `_ipc_pack_rgba` copies `rgb_nhwc` (R,G,B from `_denormalize_on_gpu` → uint8) verbatim into channels `[0,1,2]` of the IPC buffer. Byte order on the wire: `R,G,B,A`.
2. `cuda_ipc_exporter.py:19` (docstring) — "output_tensor: (H, W, 4) uint8 **BGRA** on GPU". The exporter's contract is BGRA.
3. `TDReceiver.py:800-801, 483-487` — receiver builds a uint8 RGBA8 TOP via `copyCUDAMemory(addr, size, CUDAMemoryShape(dtype=uint8, ...))`. TD interprets the bytes positionally: byte[0]→R-display, byte[1]→G-display, byte[2]→B-display. Wrapper writes B-source data into byte[2], so blue from Python is shown as red, and vice versa.

The legacy fork SHM path went through `copyNumpyArray()` (CPU SHM read by `script_top_callbacks.py`), which TD's TOP also displays as RGBA — but the legacy producer was sending the SAME RGB byte order. So why did the OLD path look right? **Because the legacy producer was `Scripts/shmem__Text__SharedMemEXT__td.py:280-297`'s `sendData()`, which TD called from the TD-side; TD's input TOP gave numpyArray in TD's native RGBA byte order to begin with.** The new path bypasses that — it sends from the Python wrapper, which assumes RGB throughout.

### Edit

Single edit to `StreamDiffusion/src/streamdiffusion/wrapper.py:995-1002` (`_ipc_pack_rgba`):

**Before:**
```python
def _ipc_pack_rgba(self, rgb_nhwc: torch.Tensor) -> torch.Tensor:
    """Pad (B,H,W,3) uint8 → (B,H,W,4) uint8 with opaque alpha on GPU, reusing a persistent buffer."""
    B, H, W, _ = rgb_nhwc.shape
    if self._ipc_rgba_buf is None or self._ipc_rgba_buf.shape != (B, H, W, 4):
        self._ipc_rgba_buf = torch.empty((B, H, W, 4), dtype=torch.uint8, device=rgb_nhwc.device)
        self._ipc_rgba_buf[..., 3] = 255  # alpha channel set once; reused across frames
    self._ipc_rgba_buf[..., :3].copy_(rgb_nhwc)
    return self._ipc_rgba_buf
```

**After:**
```python
def _ipc_pack_rgba(self, rgb_nhwc: torch.Tensor) -> torch.Tensor:
    """Pad (B,H,W,3) uint8 RGB → (B,H,W,4) uint8 BGRA on GPU for cuda-link IPC transport.

    cuda-link's wire contract is BGRA (cuda_ipc_exporter.py:19). TD interprets the raw GPU
    bytes positionally as RGBA8 in the Script TOP, so we must swap R↔B at pack time to
    keep colors correct in TD.
    """
    B, H, W, _ = rgb_nhwc.shape
    if self._ipc_rgba_buf is None or self._ipc_rgba_buf.shape != (B, H, W, 4):
        self._ipc_rgba_buf = torch.empty((B, H, W, 4), dtype=torch.uint8, device=rgb_nhwc.device)
        self._ipc_rgba_buf[..., 3] = 255  # alpha channel set once; reused across frames
    # BGRA byte order: byte[0]=B, byte[1]=G, byte[2]=R, byte[3]=A
    self._ipc_rgba_buf[..., 0].copy_(rgb_nhwc[..., 2])  # B ← source channel 2 (B)
    self._ipc_rgba_buf[..., 1].copy_(rgb_nhwc[..., 1])  # G ← source channel 1 (G)
    self._ipc_rgba_buf[..., 2].copy_(rgb_nhwc[..., 0])  # R ← source channel 0 (R)
    return self._ipc_rgba_buf
```

Three separate `.copy_()` calls instead of a single `[..., :3].copy_()` — each is a contiguous channel-wise GPU memcpy and adds negligible overhead (single-digit µs at 512×512). No new allocations, persistent buffer preserved.

### Why not swap on the TD side (Receiver) instead?

The wrapper's output is the producer's responsibility — the IPC contract (per `cuda_ipc_exporter.py` docstring) is BGRA. Fixing the producer aligns the code with the documented contract and avoids touching the vendored upstream `td_exporter/` code. Receiver-side fix would mean editing TouchDesigner Script TOP callbacks per-component.

### Input-direction symmetry concern (Phase 1 — note, don't fix here)

Phase 1's input direction (TD→Python via `shmem_out` Sender) also goes through a uint8 RGBA byte stream, but the alpha-strip path in `Scripts/streamdiffusionTD__Text__td_manager__td.py:_get_input_frame` does `frame[:, :, :3]` — it takes channels `[0,1,2]` as RGB. If TD's cuda-link Sender writes those bytes as BGRA, the SD pipeline receives BGR-as-RGB → input is silently color-swapped too.

Two possibilities:
1. The OLD fork SHM input path was equally swapped, and SDXL-Turbo's prompt conditioning is tolerant enough that nobody noticed; the regression doesn't manifest visually.
2. The TD-side Sender's TOP→GPU export does an implicit swizzle, so the byte stream reaching Python is already RGB.

Defer investigation until after Phase 2.2 output fix lands. If input also looks color-shifted after the fix, mirror the swap in `_get_input_frame` (swap channels 0↔2 before returning).

### Validation (Phase 2.2)

1. Save edit in `wrapper.py`. No TD-side change needed (Receiver stays as-is).
2. Restart Python process. Confirm exporter still logs `dtype=uint8 (kind=1 bits=8 flags=0x0000)` — metadata unchanged.
3. Visual check: blue objects in the SD output now appear blue in the TD viewer (not orange/red). Skin tones look correct.
4. Sanity-check perf: `Frame N: avg cudaMemory` line shows same ~120 µs as before — three split copies should not slow the per-frame budget.
5. Off→On→Off→On cycle parity preserved.

### Rollback

Revert the 3-line copy back to `self._ipc_rgba_buf[..., :3].copy_(rgb_nhwc)`. No state to clean up; persistent buffer is dtype/shape-compatible.

---

## Phase 2.1 — Swap `shmem` (Python→TD output direction)

### Approach: duplicate the working `shmem_out`

Cleanest path — preserves the extension wiring, internal Text DAT sync to `_compat/td_exporter/`, and all parameter overrides.

**Procedure (in TD):**

1. **Archive the old fork `shmem`**: rename existing `shmem` → `shmem_old` (don't delete yet — keep as rollback).
2. **Duplicate `shmem_out`**: copy-paste → rename copy to `shmem`.
3. **Flip parameters** on the new `shmem`:
   - `Mode`: `Sender` → `Receiver`
   - `Ipcmemname`: `parent.SDTD.par.Streamoutname + '_input_ipc'` → `parent.SDTD.par.Streamoutname + '_output_ipc'`
     - Resolves to `StreamDiffusionTD_512-512_output_ipc` — symmetric with `_input_ipc` and matches the wrapper's canonical default `"streamdiffusion_output_ipc"` (`wrapper.py:135`).
     - **Replaces the old fork expression** `op('stream_osc_data')['output-name',1]` (which resolved to a randomized per-process name like `StreamDiffusionTD_512-512_out_1779045216`). That randomized scheme is dead with the cuda-link swap — the new IPC name is deterministic.
   - `Numslots`: keep 3.
4. **Re-wire I/O**:
   - Sender's input wire is gone (no source TOP feeds a Receiver).
   - New Receiver exposes its output via internal Script TOP `output`. Re-wire whatever currently consumes the old `shmem` output to consume the new `shmem`'s output instead.
5. **Activate**: toggle new `shmem.par.Active = On`. Watch Textport for `[CUDAIPCExtension:Receiver]` init log, then `import_frame` lines.

**One Python-side edit** — rename the exporter SHM to `_output_ipc` for symmetry with `_input_ipc`:

Edit `Scripts/StreamDiffusionTD__Text__StreamDiffusionExt__td.py:3755`:
```python
# Before:
yaml_content += f"cuda_ipc_shm_name: '{stream_name}_ipc'\n"
# After:
yaml_content += f"cuda_ipc_shm_name: '{stream_name}_output_ipc'\n"
```

Propagation:
1. TD Writeconfigs writes new YAML value `cuda_ipc_shm_name: 'StreamDiffusionTD_512-512_output_ipc'`
2. Python loads it via `config.py:161` and `wrapper.py:135,337`
3. `CUDAIPCExporter` at `wrapper.py:1011` now writes to `..._output_ipc`
4. New `shmem` Receiver reads from `..._output_ipc` — names match.

No edit to `wrapper.py` or `config.py` needed — they already use `cuda_ipc_shm_name` as a config-driven string. The wrapper's hardcoded default `"streamdiffusion_output_ipc"` (`wrapper.py:135`) is overridden by the YAML-driven value.

### `ImageChanged()` — pre-existing regression (note, don't fix here)

Research finding (Phase 1 explore agent): `ImageChanged()` is only fired by the dead-code SHM path in `Scripts/shmem__Text__SharedMemEXT__td.py:412-420` (`_trigger_change_callback`). The active CUDA IPC path **never** fires it. This means feedback-safe mode (`StreamDiffusionExt:372-374`, `is_feedback_safe and is_stream_active`) has been silently broken since the Round-3 SHM-path removal.

**Phase 2.1 does NOT make this worse** — it just continues not firing it. If feedback-safe is in use, the fix is to add a 5-line onCook hook to the new `shmem`'s `output` Script TOP that calls `parent.SDTD.ImageChanged()` (optionally with a hash-detect guard to match the original change-detected firing model, not per-frame). Deferred to a separate ticket.

### Validation (Step 2.1.5)

1. **YAML emit propagated**: after Writeconfigs, `StreamDiffusion/StreamDiffusionTD/td_config.yaml:72` shows `cuda_ipc_shm_name: 'StreamDiffusionTD_512-512_output_ipc'` (line number may shift; key value is what matters).
2. **Python exporter uses new name**: on `start_streaming()`, Textport / Python logs show `CUDAIPCExporter` initialized with `StreamDiffusionTD_512-512_output_ipc`.
3. Toggle `shmem.par.Active = On` after Python is streaming. Textport shows `[CUDAIPCExtension:Receiver] Receiver initialized` and `import_frame` lines with non-zero `copyCUDAMemory` time (already observed in Phase 1 logs from the fork Receiver — confirm same with the upstream version).
4. Visual check: the downstream node consuming `shmem`'s output displays the StreamDiffusion frames in real time.
5. Run 10+ min — verify no growth in receiver slot count, no re-init storms, no error 201 on the Receiver side.
6. Off→On→Off→On cycle parity with `shmem_out` — clean teardown.

### Rollback

If validation fails: disable new `shmem.par.Active`, rename it `shmem_new`, rename `shmem_old` → `shmem`, re-wire downstream to the old COMP's output. Python side is unaffected — it'll keep writing to the same `_ipc` SHM and the fork Receiver picks it up.

---

## Phase 1 — historical record

**TD side working.** Latest run confirms the new `shmem_out` (cuda-link v1.4.1 Sender) initializes cleanly:

```
[CUDAIPCExtension:Sender] Created new SharedMemory: StreamDiffusionTD_512-512_input_ipc (433 bytes)
[CUDAIPCExtension:Sender] Wrote all IPC handles v1 to SharedMemory (433 bytes total)
[CUDAIPCExtension:Sender] Wrote metadata: 512x512x4, kind=1 bits=8 flags=0x0000, size=1048576B
[CUDAIPCExtension:Sender] Initialization complete - ready for zero-copy GPU transfer
```

- **Phase 1 Steps 1.1, 1.2, 1.3 — DONE.** COMP swapped, source TOP wired, `Ipcmemname = parent.SDTD.par.Streamoutname + '_input_ipc'` → `StreamDiffusionTD_512-512_input_ipc`.
- **YAML emit DONE.** `Scripts/StreamDiffusionTD__Text__StreamDiffusionExt__td.py:3768` already emits `cuda_ipc_input_shm_name: '{stream_name}_input_ipc'`. Verified live in `StreamDiffusionTD/td_config.yaml:83`.
- **Vendored cuda-link td_exporter/ in place** — `_compat/td_exporter/` synced to v1.4.1 (commit `92989fc`).

**Python side still on CPU SHM — current blocker.** Run fails with:

```
TouchDesignerManager - ERROR - Failed to initialize SharedMemory:
  [WinError 2] The system cannot find the file specified: 'StreamDiffusionTD_512-512'
```

The Python process tries to open the **old** CPU SHM name (`StreamDiffusionTD_512-512`), which the new cuda-link COMP no longer creates. Only `StreamDiffusionTD_512-512_input_ipc` exists now (the IPC control-packet SHM, 433 bytes — far smaller than the expected RGB frame buffer).

**Root cause of earlier wasted edit:** I previously edited `StreamDiffusion/StreamDiffusionTD/td_manager.py` directly. That file is a **build target** — TD's Writeconfigs (`StreamDiffusionExt:2602-2632`) overwrites it from the Text DAT `streamdiffusionTD/td_manager`. Canonical source is `Scripts/streamdiffusionTD__Text__td_manager__td.py`.

---

## Context

**Why this change.** Previous debugging (Round 1-9 in `structured-purring-kurzweil.md`) chased a sticky `CUDA error 201 INVALID_CONTEXT` in the SDTD-bundled CUDA IPC sender at `TDSender.py:843`. The SDTD code is a fork of upstream `cuda-link`. Upstream ships a packaged `CUDAIPCLink_v1.4.1.tox` that speaks the same wire protocol (`PROTOCOL_MAGIC = 0x43495044`). Swapping the TD-side .tox lets us stop debugging the fork and inherit upstream fixes.

**Scope:** Proof-of-concept first — swap `shmem_out` only (TD→Python direction). Keep Python-side `_compat/cuda_ipc/` fork (wire-protocol compatible). Eventually swap remaining 4 `shmem*` siblings, each in its own phase.

---

## Phase 1 remaining work — finish Python integration

### Edit location convention (MUST FOLLOW)

| File modified by user | Runtime file (auto-generated) | Direction |
|---|---|---|
| `Scripts/streamdiffusionTD__Text__td_manager__td.py` | `StreamDiffusion/StreamDiffusionTD/td_manager.py` | Edit Scripts/, TD's Writeconfigs writes runtime file |
| `Scripts/StreamDiffusionTD__Text__StreamDiffusionExt__td.py` | (lives inside .toe — synced to/from Text DAT) | Edit Scripts/, instantly synced |

**Never edit `StreamDiffusion/StreamDiffusionTD/td_manager.py` directly** — it will be clobbered on next Writeconfigs.

### Edit set — `Scripts/streamdiffusionTD__Text__td_manager__td.py`

Apply four small edits, each minimal:

**Edit 1 — declare the importer slot (after line 83):**

```python
self.syphon_handler = None
self.ipc_input_importer = None  # CUDAIPCImporter when cuda_ipc_input_shm_name is in td_settings
```

**Edit 2 — `_initialize_memory_interfaces()` (line 327-332):** wrap CPU SHM open behind an IPC-importer-first check.

Replace:

```python
else:
    # Initialize SharedMemory (same pattern as your current version)
    try:
        # Input memory (from TouchDesigner)
        self.input_memory = shared_memory.SharedMemory(name=self.input_mem_name)
        logger.debug(f"Connected to input SharedMemory: {self.input_mem_name}")
```

With:

```python
else:
    # Initialize SharedMemory (same pattern as your current version)
    try:
        # Input: prefer CUDA IPC (cuda-link shmem_out COMP) over CPU SHM
        ipc_input_name = self.td_settings.get('cuda_ipc_input_shm_name')
        if ipc_input_name:
            try:
                from streamdiffusion._compat.cuda_ipc import CUDAIPCImporter
                self.ipc_input_importer = CUDAIPCImporter(
                    shm_name=ipc_input_name,
                    device=torch.cuda.current_device(),
                    timeout_ms=500.0,
                )
                logger.info(f"CUDA IPC input configured: {ipc_input_name}")
            except Exception as e:
                logger.warning(f"CUDA IPC input init failed, falling back to CPU SHM: {e}")

        if self.ipc_input_importer is None:
            # Input memory (from TouchDesigner)
            self.input_memory = shared_memory.SharedMemory(name=self.input_mem_name)
            logger.debug(f"Connected to input SharedMemory: {self.input_mem_name}")
```

Rest of the function (output_memory, control_memory, ipadapter_memory) is unchanged — they remain on CPU SHM in Phase 1.

**Edit 3 — `_get_input_frame()` (line 627-653):** try the IPC importer first.

Replace whole body:

```python
def _get_input_frame(self) -> Optional[np.ndarray]:
    """Get input frame from TouchDesigner (platform-specific)"""
    try:
        if self.is_macos and self.syphon_handler:
            return self.syphon_handler.capture_input_frame()

        if self.ipc_input_importer is not None:
            frame = self.ipc_input_importer.get_frame_numpy()
            if frame is not None and frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]  # strip alpha; downstream pipeline expects RGB
            return frame

        if self.input_memory:
            width = self.config['width']
            height = self.config['height']
            frame = np.ndarray((height, width, 3), dtype=np.uint8, buffer=self.input_memory.buf)
            return frame.copy()

        return None
    except Exception as e:
        logger.error(f"Error getting input frame: {e}")
        return None
```

**Edit 4 — `_cleanup_memory_interfaces()` (after the syphon_handler block, before `if self.input_memory:`):**

```python
if self.ipc_input_importer is not None:
    try:
        self.ipc_input_importer.cleanup()
    except Exception:
        pass
    self.ipc_input_importer = None
```

### Reused infrastructure (no new code needed)

- `streamdiffusion._compat.cuda_ipc.CUDAIPCImporter` — already exported from `__init__.py:11` and `:27`. Constructor signature `(shm_name, shape=None, dtype=None, debug=False, timeout_ms=5000.0, device=0)`. Lazy init on first `get_frame_numpy()` call. Methods: `get_frame_numpy() -> np.ndarray | None`, `cleanup() -> None`.
- `torch.cuda.current_device()` for the device arg — `torch` is already imported at `td_manager.py:17`.
- `self.td_settings` — already populated at `td_manager.py:61` from YAML's `td_settings` block.

### After edits — propagation flow

1. Save `Scripts/streamdiffusionTD__Text__td_manager__td.py`. TD picks up the change in its Text DAT immediately.
2. Trigger **Writeconfigs** in TD (the action that runs `StreamDiffusionExt:2602-2632`). This rewrites `StreamDiffusion/StreamDiffusionTD/td_manager.py` from the Text DAT.
3. Restart the StreamDiffusion Python process. On `start_streaming()`, `_initialize_memory_interfaces()` reads `cuda_ipc_input_shm_name` from `td_settings`, opens the IPC SHM `StreamDiffusionTD_512-512_input_ipc`, and the per-frame loop calls `ipc_input_importer.get_frame_numpy()`.

---

## Validation (Step 1.5)

After Writeconfigs + Python restart, verify in this order:

1. **Init log line present:** `TouchDesignerManager - INFO - CUDA IPC input configured: StreamDiffusionTD_512-512_input_ipc`. No `[WinError 2]` for the old name.
2. **First-frame open:** `CUDAIPCImporter` log line shows it opened the IPC handles for slots 0/1/2 (matches TD's `Wrote slot 0/1/2 handles` lines).
3. **Frame flow:** StreamDiffusion produces output frames (Textport shows non-zero output FPS; not stuck at "waiting for sender").
4. **Shape sanity:** TD writes 512×512×4 (RGBA uint8, 1,048,576 B) per metadata line — `get_frame_numpy()` returns `(512, 512, 4)`, alpha is stripped to `(512, 512, 3)` by the Edit 3 guard, matches the pipeline's expected RGB shape.
5. **Stress test:** Run 10+ min at 512×512. Watch GPU mem (should stay flat — ring buffer is 3 × 1 MB), no re-init storms, no error 201.
6. **Rollback path:** If anything fails, set `shmem_out`'s `Active = Off` in TD. The Python side will log the IPC init failure and fall through to CPU SHM (which will still fail with WinError 2 because nothing creates it now — so the practical rollback is restoring the previous `.toe` from the pre-swap backup).

---

## Phase 2 (deferred) — swap remaining shmem* siblings

| Phase | COMP | Mode | Notes |
|---|---|---|---|
| 2.1 | `shmem` | Receiver | Python→TD output; `Ipcmemname = {Streamoutname}_output_ipc` — symmetric with `_input_ipc`, requires one-line YAML emit edit |
| 2.2 | `shmem_out_cn` | Sender | TD→Python ControlNet input; `Ipcmemname = {Streamoutname}_cn_ipc` |
| 2.3 | `shmem_out_out_ip` | Sender | TD→Python IPAdapter input; `Ipcmemname = {Streamoutname}_ip_ipc` |
| 2.4 | `shmem_in_cn_processed` | Receiver | Python→TD ControlNet processed output |

Each requires a parallel slot/init/cleanup edit in `td_manager.py` for its Python-side counterpart, plus a YAML emit key.

---

## Phase 3 (deferred) — cleanup

After all 5 siblings swapped:

- Delete `Scripts/shmem*__Text__SharedMemEXT__td.py` (5 files)
- Delete `Scripts/shmem*__Execute__execute__td.py` (5 files)
- Delete `Scripts/shmem*__Text__dot_lop_utils__td.py`, `__Text__dot_chat_util__td.py`, `__Text__output_callbacks__td.py`, `__ParExecute__*__td.py`
- Leave `_compat/cuda_ipc/` and `_compat/td_exporter/` untouched

---

## Critical files

**Edit (Scripts/ — canonical source):**
- `D:/dev/SD_3_0_1/test_Install_dev/StreamDiffusion/Scripts/streamdiffusionTD__Text__td_manager__td.py` — 4 edits listed above

**Already edited / verified correct:**
- `D:/dev/SD_3_0_1/test_Install_dev/StreamDiffusion/Scripts/StreamDiffusionTD__Text__StreamDiffusionExt__td.py:3768` — emits `cuda_ipc_input_shm_name`
- TD COMP `shmem_out` — `Ipcmemname = parent.SDTD.par.Streamoutname + '_input_ipc'`

**Read-only references:**
- `D:/dev/SD_3_0_1/test_Install_dev/StreamDiffusion/StreamDiffusion/src/streamdiffusion/_compat/cuda_ipc/cuda_ipc_importer.py:494` — `CUDAIPCImporter.__init__`
- `D:/dev/SD_3_0_1/test_Install_dev/StreamDiffusion/StreamDiffusion/src/streamdiffusion/_compat/cuda_ipc/cuda_ipc_importer.py:989` — `get_frame_numpy()`
- `D:/dev/SD_3_0_1/test_Install_dev/StreamDiffusion/StreamDiffusion/src/streamdiffusion/_compat/cuda_ipc/cuda_ipc_importer.py:1236` — `cleanup()`
- `D:/dev/SD_3_0_1/test_Install_dev/StreamDiffusion/StreamDiffusion/src/streamdiffusion/_compat/cuda_ipc/__init__.py:11,27` — `CUDAIPCImporter` re-export

**Do NOT edit (runtime targets — clobbered by TD on Writeconfigs):**
- `D:/dev/SD_3_0_1/test_Install_dev/StreamDiffusion/StreamDiffusion/StreamDiffusionTD/td_manager.py`
- `D:/dev/SD_3_0_1/test_Install_dev/StreamDiffusion/StreamDiffusion/StreamDiffusionTD/td_config.yaml`

---

## Save plan to project repo

Per memory `feedback_save_plans_as_project_files`, copy this plan to:
`D:/dev/SD_3_0_1/test_Install_dev/StreamDiffusion/StreamDiffusion/_plans/drifting-twirling-tulip.md`
(after exiting plan mode — copy not possible in plan mode).
