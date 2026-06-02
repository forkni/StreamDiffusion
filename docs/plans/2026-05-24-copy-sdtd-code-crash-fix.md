# Fix: stream crash — empty on-disk StreamDiffusionTD/*.py ("Press any key to continue" with no output)

## Context

After pressing **Start Stream** in TouchDesigner, the spawned CMD window shows only the
batch `pause` prompt — `Press any key to continue . . .` — with **nothing above it**. Zero
output means the Python process produced no stdout/stderr and exited cleanly (exit 0), then
`Start_StreamDiffusion.bat` hit its `pause`.

This was initially suspected to be fallout from this session's CUDA perf edits
(P4/P5/P6 in `td_manager`), but that was **ruled out**: those edits live inside
log-and-continue `try/except` blocks and would print a traceback, not produce zero output.

## Root cause (confirmed)

`Start_StreamDiffusion.bat` runs `venv\Scripts\python.exe streamdiffusionTD\td_main.py`.
The on-disk `StreamDiffusionTD\td_main.py` is **0 bytes** → Python runs an empty file →
exits 0 → batch `pause` with no output. Same for three sibling files.

`StreamDiffusionExt.copy_sdtd_code()` (in
`Scripts/StreamDiffusionTD__Text__StreamDiffusionExt__td.py`, loop at lines **2603–2635**)
runs at every stream start (called twice in `Startstream`, lines 698 and 720). For each
mapped file it does `open(file_path, 'w')` (truncate) then `file.write(text_dat.text)`.
For 4 code DATs the `.text` is **empty in the running .tox**, so it wrote 0-byte files,
clobbering the previously-working content. The `td_config` branch (lines 2607–2620)
generates YAML independently, which is why `td_config.yaml` survived.

Evidence — on-disk `StreamDiffusion\StreamDiffusion\StreamDiffusionTD\` vs intact
`StreamDiffusion\Scripts\` mirrors:

| File | on-disk | Scripts/ mirror (source) | mirror date |
|---|---|---|---|
| `td_main.py` | **0** | `streamdiffusionTD__Text__td_main__td.py` 21434 | Apr 26 |
| `td_manager.py` | **0** | `streamdiffusionTD__Text__td_manager__td.py` 51755 | **May 24 (current — P4/P5/P6)** |
| `td_osc_handler.py` | **0** | `streamdiffusionTD__Text__td_osc_handler__td.py` 23068 | Apr 22 |
| `install_tensorrt.py` | **0** | `streamdiffusionTD__Text__install_tensorrt__td.py` 6706 | Apr 26 |
| `td_config.yaml` | 2400 (ok) | generated dynamically | — |
| `syphon_utils.py` | 17251 (ok) | DAT has content | — |

`StreamDiffusionTD/` is gitignored (`.gitignore:243`) — no git history, no remote to restore
from. The `Scripts/` externalized-DAT mirrors are the only on-disk source. `Backup/` holds
only `.toe` binaries (DAT text not extractable as plain text).

## Fix (approved approach)

**1 — Disable the `.py` copy in `copy_sdtd_code()`, keep `td_config` regen.**
File: `Scripts/StreamDiffusionTD__Text__StreamDiffusionExt__td.py`, `copy_sdtd_code()`.

In the loop body, skip the 4 code DATs so empty/unloaded DATs can never again clobber the
on-disk files — the on-disk `.py` (maintained via the `Scripts/` mirrors) is canonical.
Keep the existing `td_config` branch (2607–2620) and the `syphon_utils` / `requirements_mac`
writes untouched, so dynamic `td_config.yaml` regeneration at both call sites (incl. the
post-`round_width_height` regen at line 720) still works.

```python
# Code files are maintained on-disk via externalized Scripts/ mirrors; never overwrite
# them from DATs at stream start — empty/unloaded DATs would clobber working files.
_skip_code_copy = {'td_main', 'td_manager', 'td_osc_handler', 'install_tensorrt'}
...
for dat_name, filename in text_dat_paths.items():
    file_path = os.path.join(streamdiffusionTD_folder, filename)
    if dat_name == "td_config":
        ... # unchanged
        continue
    if dat_name in _skip_code_copy and os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        self.logger.log(f'Skipping copy of {dat_name} — on-disk file is canonical', level='DEBUG')
        continue
    ... # unchanged standard-copy for syphon_utils, requirements_mac (+ any missing code file)
```

**Note on the `os.path.exists(...) and getsize > 0` guard:** this is a deliberate refinement
of "skip the 4 entirely." A bare skip would break a **fresh binary install** (distributed
`.tox` where the on-disk `.py` don't exist yet and must be extracted from the bundled DATs).
Guarding on "file already present and non-empty" keeps the dev behavior the user wants
(never clobber working files) while preserving first-run extraction for the shipped product.
If the user prefers a literal unconditional skip, drop the `os.path.*` clause.

**2 — Restore the 4 cleared files from the `Scripts/` mirrors (verbatim copy).**
No `x.x.x123454321` placeholder exists in the current `td_main` mirror, so the version
substitution `copy_sdtd_code` normally applies is a no-op — copy each mirror byte-for-byte:

- `StreamDiffusionTD\td_main.py`        ← `Scripts\streamdiffusionTD__Text__td_main__td.py`
- `StreamDiffusionTD\td_manager.py`     ← `Scripts\streamdiffusionTD__Text__td_manager__td.py`
- `StreamDiffusionTD\td_osc_handler.py` ← `Scripts\streamdiffusionTD__Text__td_osc_handler__td.py`
- `StreamDiffusionTD\install_tensorrt.py` ← `Scripts\streamdiffusionTD__Text__install_tensorrt__td.py`

`td_manager` mirror is current (has this session's P4/P5/P6). `td_main` / `td_osc_handler` /
`install_tensorrt` mirrors are Apr 22–26; accepted as the best available source per user.

## Latent bug noticed (separate from the crash — fix only if user wants it now)

Restoring `td_manager.py` reintroduces this session's **P6** code in `_get_input_frame`,
which **assumes uint8 [0,255] IPC input** and normalizes via `mul_(1/127.5).add_(-1)`. The TD
textport shows `input_ipc` is **float32** (kind=2, bits=32). If TD sends float32 already in
`[0,1]`, the correct map to `[-1,1]` is `mul_(2).add_(-1)`, and the dtype must be detected,
not assumed. This does not crash; it would produce wrong pixel values once the stream runs.
Recommend a follow-up edit to `_get_input_frame` (and the symmetric P5 path) to branch on
`gpu_frame.dtype` / observed range. Out of scope for the crash fix unless approved.

## Verification

1. After the edit + restore, confirm the 4 on-disk files are non-zero
   (`td_main.py` ≈ 21 KB, `td_manager.py` ≈ 52 KB, etc.).
2. Press **Start Stream** in TD with **Debugcmd** on. The CMD window must now show the
   td_main loading animation + ASCII logo **above** any prompt — not a bare `pause`.
3. Confirm `td_config.yaml` still regenerates each start (check its mtime / the
   `Generated td_config.yaml: fp8=..., static_shapes=...` log line) — proves the
   `td_config` branch survived the edit.
4. Smoke per model targets: SD-Turbo / SDXL-Turbo, 512×512, 2-step, t_index=[32,45], seed=2.
   Watch the TD textport / `logs/` for tracebacks.
5. If the P6 float32 fix is applied: visually confirm output isn't washed-out/clipped
   (wrong-range symptom).

## Notes / constraints

- `Scripts/StreamDiffusionTD__Text__StreamDiffusionExt__td.py` and the `Scripts/` +
  `StreamDiffusionTD/` dirs are at the **parent** `...\StreamDiffusion\` level, outside the
  cwd git repo; edits there sync live to the running `.tox` (no TOX rebuild needed).
- Do **not** hand-edit `td_config.yaml` — it is regenerated by the emitter we are preserving.
- After approval, copy this plan into `StreamDiffusion\docs\plans\` per the
  save-plans-as-project-files convention.
