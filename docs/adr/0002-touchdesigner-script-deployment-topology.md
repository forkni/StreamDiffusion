# TouchDesigner script deployment stays 3 repos / 4 layers, decentralized by design

Status: accepted

## Context & Decision

The StreamDiffusion TouchDesigner component is not one repo — it's three coordinated
repos, each with a distinct role:

| Repo | Role | Working copy |
|---|---|---|
| `dotsimulate/StreamDiffusion` | Python **library** (pip package) | fork `forkni/StreamDiffusion`, editable-installed from `src\` |
| `dotsimulate/StreamDiffusionTD` | **TD component**: `operator/streamdiffusionTD/*.py` (`td_main` / `td_manager` / `td_osc_handler` / `install_tensorrt`) + `StreamDiffusion` / `dotloader` / `tox_updater` / `sd_installer` submodules | local clone (`dev` branch); no fork yet |
| `dotsimulate/StreamDiffusion-installer` | Installer, pinned via `dat_version_manifest` (= the `sd_installer` submodule target) | checked out inside the working copy |

Within a single TouchDesigner working copy, the TD component's scripts additionally exist
in **four runtime layers**, source-of-record at the top:

1. **`.tox` internal Text DATs** — the component's live, in-project copy of each script.
2. **`Scripts\…__Text__<dat>__td.py`** — each DAT's external **authoring surface**. A Text
   DAT's `File` parameter (`par.file`) points here, and **`par.loadonstart` pulls file→DAT
   at TD project startup** — confirmed against TouchDesigner's own docs (Text DAT / DAT
   Class). Editing `Scripts/` and restarting is therefore the supported way to change the
   `.tox`; DAT ⇄ `Scripts/` ⇄ `.tox` are effectively one synced unit.
3. **`streamdiffusionTD\<name>.py`** — the **deployed** copy, written by `copy_sdtd_code()`
   FROM the DATs. Its `_skip_code_copy` guard freezes the four core files
   (`td_main`/`td_manager`/`td_osc_handler`/`install_tensorrt`) once they exist non-empty, so
   a hand-tuned deployed file is never clobbered by a re-sync. `StreamDiffusionExt.
   Startstream()` runs **this** copy as a CMD subprocess at Start Stream.
4. **`src\streamdiffusion\`** — the Python library half; editable-installed, independent of
   layers 1-3, always current.

**Decision: keep this topology as-is.** The `_skip_code_copy` guard, and the resulting
ability for the deployed layer to drift from the synced DAT/`Scripts`/`.tox` unit, is
**intentional design, confirmed by the maintainer** — it exists to let one script element be
changed independently without forcing a global re-sync (the guard's own history says it was
added to stop "confusing constant re-sync"). Any future automation must *reconcile
deliberately*, not remove this independent-edit capability.

## Considered Options

- **Collapse to a single source of truth (e.g. always deploy straight from `Scripts/`,
  drop the guard)** — rejected: this is exactly the "confusing constant re-sync" behavior
  the guard was added to prevent, and it removes the ability to hand-tune one deployed
  script (e.g. hotfixing `td_manager.py` in a running install) without touching the DAT.
- **Treat `Scripts/` as a passive export/backup only** — rejected: confirmed incorrect. Per
  TouchDesigner's Text DAT `File`/`Load on Start` behavior, `Scripts/` is an **authoring
  surface** the DAT reads from at startup, not a one-way dump.
- **Keep the 3-repo / 4-layer decentralized topology, document it, and define
  reconciliation primitives on top (chosen)** — preserves the independent-edit capability;
  makes the drift surface and its risk explicit; gives a future coordinated routine a
  concrete, minimal set of operations to drive.

## Consequences

- **Drift is expected and bounded to one layer.** Measured 2026-07-13: `td_osc_handler.py`
  and `install_tensorrt.py` (deployed vs. `Scripts/` mirror) are byte-identical; `td_main.py`
  differs only in a cosmetic banner/tagline string; `td_manager.py` differs only by the
  presence of the inference error-report hook (added directly to the deployed copy). No
  other drift exists across the four layers.
- **A deployed-only hand-edit is invisible to the DAT, `.tox`, and git until promoted.**
  Example: the `td_manager.py` error-report hook (imports of `write_error_report` /
  `report_error`, the `_last_error_report_sig` debounce field, and the debounced except-block
  in `_streaming_loop`) exists **only** in the gitignored deployed file — confirmed via
  call-graph analysis (both functions have exactly one TD-side caller, and it resolves to
  the deployed file, not the `Scripts/` mirror). If that deployed file is ever deleted or
  regenerated, the hook is lost. This is an accepted tradeoff of the independent-edit
  capability, not a bug — capturing the hook durably (deployed → `Scripts/` → reload DAT →
  save `.tox` → eventually upstream `StreamDiffusionTD`) is deferred to the merge phase of
  the wider PR-stack work, not fixed here.
- **A future coordinated routine has a concrete, minimal shape:** to promote a deployed-only
  change durably, write it to the matching `Scripts\…__td.py` file, pulse the DAT's
  `par.loadonstartpulse` (or restart TD) to pull it back into the live DAT, save the `.toe`
  to persist the `.tox`, then (eventually) commit it to the upstream `StreamDiffusionTD`
  repo. The routine should reconcile **on demand**, never on every run — running it
  unconditionally on every start would reintroduce the "confusing constant re-sync" problem
  the guard exists to prevent.
- **Exact Text DAT sync parameters a routine would drive** (per DAT Class /
  Text DAT docs):

  | Param (display) | `par` name | Direction / when |
  |---|---|---|
  | File | `par.file` | external path (`Scripts\…__td.py`) |
  | Load on Start | `par.loadonstart` | file→DAT at project start (the startup-pull mechanism) |
  | Load File | `par.loadonstartpulse` | file→DAT, instant reload without restart |
  | Write on Toe Save | `par.write` | DAT→file on `.toe` save |
  | Write File | `par.writepulse` | DAT→file, instant |
  | Sync to File | `par.syncfile` | bidirectional (load at start + write-through on change) |

  Which of `loadonstart` / `write` / `syncfile` are actually enabled per-DAT determines
  whether a DAT-side edit survives the next TD startup — verify per-DAT before a routine
  relies on it.
- **The library layer (4) is out of scope for this topology** — it's the ordinary editable
  Python install covered by the ordinary PR-stack workflow (see the fork-review PR plan);
  this ADR only concerns the three TD-script layers (1-3) and their cross-repo coordination.
- Complements ADR-0001: that ADR covers the cuda-link **library** dependency deployment;
  this one covers the TD **script** deployment. Both describe deliberately-accepted,
  non-single-source-of-truth designs rather than problems to eliminate.
- Do **not** re-propose collapsing this topology to one layer in future architecture
  reviews without first checking whether the independent-edit capability is still in active
  use — this ADR exists precisely because that capability is a confirmed, intentional
  feature, not an oversight.

## References

- Text DAT (params): <https://derivative.ca/UserGuide/Text_DAT>
- DAT Class (`par.file`, `.save()`, `.write()`): <https://docs.derivative.ca/DAT_Class>
