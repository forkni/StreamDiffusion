# Project-anchored install resolution, not a shared global config slot

Status: accepted

## Context & Decision

Two StreamDiffusion installs now coexist on this machine — `D:\dev\SDTD_040_Beta\StreamDiffusion`
and `D:\dev\SDTD_041\StreamDiffusion` — each under its own TouchDesigner project. They shared
**one global config slot**, `%APPDATA%\dotsimulate_TD_Tools\streamdiffusion_config.json`, holding
a single `base_folder` value. `StreamDiffusionExt.Basefolder()` runs at every extension init
(`__init__`, `force=True`) and unconditionally rewrote that slot whenever the `Basefolder` par
was non-empty, and adopted the slot's value into a blank par otherwise. That made the slot
last-writer-wins: opening either project overwrote the other project's registration, observed
thrashing across a single session —

```text
12:50  D:/dev/SDTD_041/StreamDiffusion        (041 install ran)
13:12  D:/dev/SDTD_040_Beta/StreamDiffusion   (opened 040_Beta)
13:19  rewritten again
```

— and a cold `.tox` drop or fresh TD project with a blank `Basefolder` par bound to whichever
project had last written the slot, not necessarily its own install. That produced exactly the
rival-install collision the resolver is designed to refuse rather than paper over:

```text
CUDA-Link is already loaded from D:\dev/SDTD_040_Beta/...\cuda_link;
the selected installation is D:/dev/SDTD_041/...\cuda_link.
Restart TouchDesigner to switch installations.
```

Both roots in that message carry forward slashes — i.e. both came from the config slot, not the
filesystem — confirming the slot had changed underneath a live TD session, not that either
project's own state was wrong.

**Decision: each project resolves its own install from its own location before the shared slot
is ever consulted.** `<project.folder>\StreamDiffusion` is probed first, falling back to
`<project.folder>` itself, in both `StreamDiffusionExt.Basefolder()` (primary — `par.Basefolder`
has 45 consumers, including the stream subprocess's `cwd`, that a bootstrap-only fix would leave
on the other install) and `cuda_link_bootstrap.py`'s `_layers()` (covers the window where a DAT
compiles before `Basefolder()` has run). A candidate root is validated by reusing
`find_base_folder`'s own test — `setup.py` and `src/` present
(`StreamDiffusion-installer/sd_installer/cli.py:23-59`) — extended with a `src/streamdiffusion/`
check for specificity, since this probe runs against arbitrary project folders rather than
`find_base_folder`'s known-good cwd, where `setup.py` + `src/` alone would match any src-layout
Python repo sitting at `<project.folder>`. The shared slot becomes **seed-only**:
`set_streamdiffusion_install_path` now skips the write whenever the stored value already
resolves on disk, so it only seeds a project that can't self-anchor rather than recording
whichever project loaded most recently.

## Considered Options

- **Store multiple paths in the global slot (a registry keyed by project)** — rejected: the
  registry still needs a key to pick the right entry on load, and the only reliable key is the
  project itself — at which point the project can derive its own path directly from
  `project.folder`, with no file-format migration for a config file dotsimulate ships to end
  users.
- **Leave the slot last-writer-wins, but suppress the rival-install collision when it fires** —
  rejected: the collision is the resolver correctly refusing to silently reuse an unrelated
  install (`_RivalInstallError`, `cuda_link_bootstrap.py:44-51`), not a bug to hide. The original
  `SHMProtocol` ImportError this whole fix series addresses was itself the product of a
  version-skewed install being silently accepted; swallowing the guard here would reopen that
  exact failure mode.
- **Project-anchored resolution with a seed-only shared slot (chosen)** — each project resolves
  its own install from its own location; the shared slot degrades to a first-run hint for
  installs that can't self-anchor, instead of a mutable record of "whichever project ran last."

## Consequences

- **Two projects open in one TD process now fail loudly, by design.** Each project binding to
  its own install means the rival-install guard fires where the shared slot previously produced
  accidental (and unsound) agreement. This is an accepted trade-off, not a regression —
  cuda-link's own `docs/adr/0003:111-113` already flags cross-version install skew as unsolved,
  and `_abort` (`cuda_link_bootstrap.py:211`) is unchanged by this design.
- **`Basefolder()`'s project branch does not call `Loadconfig()`.** `Loadconfig` restores a
  global snapshot of every custom par written by whichever project last pulsed `Writeconfig`;
  anchoring to your own install and then importing another project's settings would be
  incoherent, and skipping the call avoids re-entering the `Basefolder` → `Loadconfig` →
  `Basefolder(force=True)` bounce.
- **`Basefolder` was added to both `skip_params` sets**, so the global operator-config snapshot
  (`streamdiffusion_operator_config.json`, written on `Writeconfig`) stops carrying an absolute
  install path between projects. Latent on this machine today — that file does not exist until
  someone pulses `Writeconfig` — but would otherwise have reopened the same cross-project leak
  this design closes.
- **`SDTD_BASE_FOLDER_PATH`, a second machine-wide `setx` slot with the identical
  last-writer-wins defect, was retired separately** (`sd_installer/installer.py`,
  `sd_installer/verifier.py` — folded into `dotsimulate/StreamDiffusion-installer#6`) rather than
  fixed by this design, since `diagnostics.py` already falls back to a module-relative repo root
  that is inherently anchored to whichever install actually crashed.
- Validated only against `find_base_folder`'s own definition of an install root. A future
  install layout that doesn't match `setup.py` + `src/streamdiffusion/` (e.g. a packaged wheel
  with no `src/` tree) would silently fall through to the shared-slot layer instead of
  self-anchoring — acceptable today since every real install on this machine is a source
  checkout.

## References

- `StreamDiffusionTD-fork/operator/cuda_link_bootstrap.py` — `_project_roots()`, spliced into
  `_layers()` between the `Basefolder` parameter layer and the dotsimulate config layer.
- `StreamDiffusionTD-fork/operator/StreamDiffusionExt.py` — `_project_install_path()`, the new
  `Basefolder()` branch, and the seed-only guard in `set_streamdiffusion_install_path`.
- `StreamDiffusion-installer/sd_installer/cli.py:23-59` — `find_base_folder`, reused as the
  root-validation base case.
- `dotsimulate/StreamDiffusion-installer#6` — retirement of the `SDTD_BASE_FOLDER_PATH` slot.
- `dotsimulate/StreamDiffusionTD#20` — the modal-storm / layered-resolver fix this design stacks
  on top of.
