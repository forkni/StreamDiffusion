# Static typing: pyrefly with tensor shapes

The project is type-checked with [pyrefly](https://pyrefly.org) 1.4 plus its experimental
tensor-shape stubs, so a `torch.Tensor` can carry its shape in the type
(`Tensor[[B, 4, Hl, Wl]]`) and pyrefly reports dimension mismatches statically. Configuration
lives in `pyproject.toml` under `[tool.pyrefly]`; pre-existing errors are frozen in
`pyrefly-baseline.json` and only new errors fail the check.

## Install (once per venv)

```powershell
venv\Scripts\python.exe -m pip install -r requirements-dev.txt
venv\Scripts\python.exe -m pip install --no-deps pyrefly-torch-stubs==1.4.0
```

`pyrefly`, `pyrefly-shape-extensions` and `pyrefly-torch-stubs` must stay on the same version.
The stubs package declares `torch` as a dependency, so it is installed with `--no-deps` to keep
pip from pulling a second, CPU-only torch into the CUDA venv.

Always run the venv's `venv\Scripts\pyrefly.exe`. A global `pyrefly` on `PATH` (1.3.0 on the
development machine) predates the stubs and disagrees with the committed baseline.

## Check

```powershell
venv\Scripts\pyrefly.exe check
```

Expected: `0 errors`. `hello_shapes.py` in this directory is a smoke file that only type-checks
(it is not collected by pytest): if it reports errors, the stubs are not resolving. Confirm with
`venv\Scripts\pyrefly.exe dump-config src\streamdiffusion\pipeline.py` that the interpreter is
the venv's and that `venv\Lib\site-packages` is on the site-packages path.

## Baseline

After intentionally fixing or adding to the backlog, regenerate the baseline and record the
before/after entry counts in the commit message:

```powershell
venv\Scripts\pyrefly.exe check --update-baseline
```

History: 806 entries under pyrefly 1.3.0 without stubs; 1663 after switching to 1.4.0 with the
shape stubs (2026-10-24), which are stricter than torch's own type hints.

## Writing shape annotations

Shape symbols (`B`, `C`, `H`, `W`, `Hl`, `Wl`, `N`) come from `src/streamdiffusion/_shapes.py`
and exist only under `TYPE_CHECKING`; at runtime they are plain `int`. Rules:

- `from __future__ import annotations` in every module that uses them.
- Shape symbols appear only in annotations and in `assert_type` / `reveal_type`, never in a
  runtime expression such as `torch.zeros(B, C)`.
- `shape_extensions` is a development dependency only: never import it outside an
  `if TYPE_CHECKING:` block, and never add it to `requirements.txt` or `pyproject` dependencies.
- Calls into untracked code (diffusers VAE/UNet, TensorRT engines) return a bare `Tensor`; pin
  the result with an annotated assignment (`x: Tensor[[B, 4, Hl, Wl]] = ...`) rather than a cast.

The porting procedure is the `add-shape-types-to-torch-model` skill (vendored from
`facebook/pyrefly` in the `claude-dotfiles` repository).

## Git hooks

`.cgw.conf` (machine-local, untracked) points the CGW typecheck step at the venv binary:

```bash
CGW_TYPECHECK_CMD="venv/Scripts/pyrefly.exe"
CGW_TYPECHECK_CHECK_ARGS="check"
CGW_TYPECHECK_EXCLUDES=""
```

The baseline path comes from `pyproject.toml`, so no `--baseline` flag is needed.

## pre-commit and CI

Two portable gates run the same `pyrefly check --baseline pyrefly-baseline.json` as the venv:

- `.pre-commit-config.yaml`: ruff (lint + format check) and pyrefly. Run it by hand with
  `venv\Scripts\pre-commit.exe run --all-files`. Do **not** run `pre-commit install` on a checkout
  that uses the CGW wrappers: it would replace the CGW `.git/hooks/pre-commit` described above.
  The pyrefly hook installs `pyrefly==1.4.0` into pre-commit's own environment (so a different
  global `pyrefly` on `PATH` is never used) and resolves the torch stubs through
  `python-interpreter-path` in `pyproject.toml`, i.e. the project venv.
- `.github/workflows/typecheck.yml`: job `pyrefly` on Ubuntu with Python 3.11. It installs CPU
  torch/torchvision from the PyTorch index, the pins in `tests/typing/requirements-ci.txt`
  (the exact versions in the development venv, including the diffusers fork commit) and
  `pyrefly-torch-stubs` with `--no-deps`, then runs the check with
  `--python-interpreter-path "$(which python)"` because the venv path in `pyproject.toml` does
  not exist on the runner. Pull requests into `SDTD_040_beta_release` / `SDTD_032_dev` and
  pushes to `SDTD_040_beta_release` trigger it.

How the runner reproduces the local result:

- `python-platform = "windows"` in `pyproject.toml` makes `sys.platform` branches resolve the
  same way on Linux (without it, `os.add_dll_directory` reads as a missing attribute there).
- Packages the runner does not install (TensorRT, polygraphy, ONNX, OpenCV, mediapipe,
  insightface, xformers, modelopt, ...) are listed in `ignore-missing-imports` and type as `Any`.
  A venv that has them is checked against the real package, so CI errors are a subset of local
  errors, never a superset.
- `pyrefly-baseline.json` stores no line numbers and uses repository-relative forward-slash
  paths, so it is valid on both operating systems and survives unrelated line shifts.

When the venv moves (torch, diffusers, transformers, pyrefly), update `requirements-ci.txt`
and `requirements-dev.txt` together, re-run `pyrefly check --update-baseline`, and commit the
pins and the baseline in the same change.
