# cuda-link is an external pip dependency, not vendored

Status: accepted

## Context & Decision

StreamDiffusion previously vendored the `cuda-link` CUDA-IPC library into
`src/streamdiffusion/_compat/` as two mirror trees (`cuda_ipc/` Python runtime +
`td_exporter/` TouchDesigner DAT source). Keeping the mirrors in lockstep was a recurring
maintenance trap (the "RE-VENDORING TRAP" documented in `VENDORED_VERSION.txt`: 5
relative-import patches re-applied on every re-vendor) and the dependency seam was
dishonest — `wrapper.py` hard-imported `cuda_link` while `setup.py` never declared it.

We now **depend solely on the pip-installed `cuda-link`** (declared in `setup.py` as
`cuda-link @ git+https://github.com/forkni/cuda-link@v1.8.1`, exposed via the `cuda_ipc`
optional extra). The TouchDesigner side consumes the same installed package through
`CUDALinkBootstrap`'s **library mode** (`CUDALINK_LIB_PATH` injects the venv onto TD's
`sys.path` and aliases the 14 bare module names used by TD DATs), so the TD DAT mirror
inside the repo is no longer needed either.

## Considered Options

- **Re-vendor on each release** — rejected: the 5-patch RE-VENDORING TRAP makes every
  upstream update a multi-step, error-prone manual process with no test coverage.
- **Keep mirrors as read-only reference copies** — rejected: they silently diverge from the
  installed version, giving a false sense of documentation accuracy.
- **Depend on pip + library mode (chosen)** — single source of truth; honest dependency
  seam; TD and Python both consume the same binary; upgrade is `pip install -U cuda-link`.

## Consequences

- The load-bearing diffusers monkey-patch (`diffusers_kvo_patch.py`) was relocated from
  `_compat/` to `src/streamdiffusion/_patches/` — it was misfiled there (nothing to do
  with cuda-link). See `_patches/__init__.py` for the import-time side effect.
- IPC is an **optional feature**: `pip install -e .[cuda_ipc]`. The `wrapper.py` imports
  are lazy/in-method so the core package installs and runs without cuda-link.
- The `github.com/forkni/cuda-link` remote **must carry the referenced tag** (e.g. `v1.8.1`)
  or clean installs fail. Tag `v1.8.1` is pushed as of 2026-06-01.
- The `CUDALINK_LIB_PATH` env var must be set in the TD launch environment to point at the
  venv site-packages path; without it, `CUDALinkBootstrap` falls back to classic Text-DAT
  module discovery (still works, but requires the mirror DATs to be present in the COMP).
- Do **not** re-suggest re-vendoring in future architecture reviews — this is a deliberate
  reversal of the previous approach, made after confirming that the mirror trees had zero
  runtime consumers in the Python import graph.
