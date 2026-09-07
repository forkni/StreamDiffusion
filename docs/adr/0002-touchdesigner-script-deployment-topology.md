# TouchDesigner script deployment stays 3 repos / 4 layers, decentralized by design

Status: accepted

## Context & Decision

The StreamDiffusion TouchDesigner component is not one repo — it's three coordinated
repos, each with a distinct role:

| Repo | Role | Working copy |
|---|---|---|
| `dotsimulate/StreamDiffusion` | Python **library** (pip package) | fork `forkni/StreamDiffusion`, editable-installed from `src\` |
| `dotsimulate/StreamDiffusionTD` | **TD component**: `operator/streamdiffusionTD/*.py` (`td_main` / `td_manager` / `td_osc_handler` / `install_tensorrt`) + `StreamDiffusion` / `dotloader` / `tox_updater` / `sd_installer` submodules | local clone (`dev` branch); fork `forkni/StreamDiffusionTD` |
| `dotsimulate/StreamDiffusion-installer` | Installer, pinned via `dat_version_manifest` (= the `sd_installer` submodule target) | checked out inside the working copy; fork `forkni/StreamDiffusion-installer` |

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

**Decision: keep this topology as-is, on fork-local evidence only.** The `_skip_code_copy`
guard, and the resulting ability for the deployed layer to drift from the synced
DAT/`Scripts`/`.tox` unit, is grounded in the guard's own commit history, which says it was
added to stop "confusing constant re-sync" — that part of the record is solid. Any future
automation must *reconcile deliberately*, not remove this independent-edit capability.

**Correction (2026-07-29): the "confirmed by the maintainer" claim below was wrong and has
been withdrawn.** This ADR originally asserted the guard was "intentional design, confirmed
by the maintainer." Studying upstream history for the fork/PR plan turned up
`dotsimulate/StreamDiffusion@8f6d6396` (2026-07-18), which deletes this exact document with
the message: *"remove ADR-0002: TouchDesigner operator script topology doc, belongs in
private operator repo, contains incorrect maintainer-confirmation claims."* That is the
maintainer's actual, on-the-record position — the opposite of what this ADR claimed. The
guard's *behavior* (freezing the four core deployed files once non-empty) is still accurately
described below from direct code inspection; only the maintainer-confirmation framing is
withdrawn. This document is retained fork-side because the topology facts remain useful for
fork-internal tooling decisions — see the status note at the end.

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

- **Drift is now actively guarded, not just measured.** As of 2026-09-06,
  `scripts\sync_td_mirror.py --check` enforces byte-identity between deployed and `Scripts/`
  mirror for all four core files (`td_main.py`, `syphon_utils.py`, `td_osc_handler.py`,
  `td_manager.py`); all four currently pass. This supersedes the 2026-07-13 snapshot below,
  which recorded ad hoc drift before the guard existed: `td_main.py` differed only in a
  cosmetic banner/tagline string, and `td_manager.py` differed only by the presence of the
  inference error-report hook (added directly to the deployed copy) — both gaps have since
  been closed by promoting the deployed-only changes into `Scripts/` and re-verifying via
  `diff`.
- **A deployed-only hand-edit is invisible to the DAT, `.tox`, and git until promoted.**
  This was true of the `td_manager.py` error-report hook (imports of `write_error_report` /
  `report_error`, the `ErrorReporter` dedup state, and the debounced except-block in
  `_streaming_loop`), which at one point existed **only** in the gitignored deployed file.
  It has since been promoted into the `Scripts/` mirror (see above) and both layers are
  byte-identical today. The risk described here is still real for *future* deployed-only
  edits to any of the four guarded files — `sync_td_mirror.py --check` is what surfaces that
  drift going forward, rather than requiring a manual `diff` — and remains unmitigated for
  the two Text-DAT authoring surfaces not yet in `PAIRS`
  (`StreamDiffusionExt__td.py`, `AsyncIOManager__td.py`), which have no deployed/canonical
  counterpart to drift from in the first place.
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
  use — this ADR exists precisely because the guard's own history shows it was a deliberate
  choice at the time it was added, not an oversight. (It is no longer accurate to call that
  choice "confirmed by the maintainer" going forward — see the correction above.)

## Status note — fork-internal only

This document does not exist upstream (`dotsimulate/StreamDiffusion` deleted it in
`8f6d6396`, see References) and must **never** be included in an upstream PR. It stays in
`forkni/StreamDiffusion` for fork-internal tooling/deployment decisions only.

## References

- Text DAT (params): <https://derivative.ca/UserGuide/Text_DAT>
- DAT Class (`par.file`, `.save()`, `.write()`): <https://docs.derivative.ca/DAT_Class>
- Upstream deletion of this document: `dotsimulate/StreamDiffusion@8f6d6396`
  (2026-07-18) — "remove ADR-0002: TouchDesigner operator script topology doc, belongs in
  private operator repo, contains incorrect maintainer-confirmation claims"
