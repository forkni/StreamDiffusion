---
applyTo: "tests/**"
---

# Test review guidance

## Flag these

- **Tests that require a real GPU to run in CI.** Unit tests under `tests/unit/` must stay CPU-only / CI-safe
  — monkeypatch `torch.cuda` and any TensorRT/CUDA-only calls rather than skip-marking them or assuming a
  GPU runner. `tests/unit/test_diagnostics.py` is the reference pattern for this (monkeypatched torch, no
  real CUDA calls).
- **New behavior shipped without a matching test.** A new public function/method in `src/streamdiffusion/`
  (especially anything in `utils/`) or a new CLI/wrapper entry point should come with at least one unit test
  covering its main path and one error/edge case.
- **Assertions that only check "it didn't crash."** Prefer asserting on the actual returned/written content
  (e.g. report text contains expected fields) over a bare "no exception raised" check.

## Do NOT flag

- `E402` (module-level import not at top of file) in `tests/**` — this is repo-wide per-file-ignored in
  `pyproject.toml` because tests use a `sys.path` hack before importing the package under test; don't
  suggest reordering these imports.
- Tests living outside `tests/unit/` (e.g. `tests/manual/`, `tests/quality/`) requiring a GPU — those are
  excluded from the CPU-only expectation and from the pyrefly project scope.
