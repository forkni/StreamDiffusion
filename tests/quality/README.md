# Quality Regression Harness

Compares FP8-TRT output against FP16-TRT golden reference images using SSIM + LPIPS.

## First-time setup

```powershell
# Install dev deps (lpips, scikit-image)
pip install -r requirements-dev.txt

# Build FP16-TRT engines for both fixture models (if not already cached)
python -m streamdiffusion.acceleration.tensorrt.build --model stabilityai/sd-turbo
python -m streamdiffusion.acceleration.tensorrt.build --model stabilityai/sdxl-turbo

# Generate goldens + update manifest
python tests/quality/regenerate_golden.py --update-manifest

# Build FP8-TRT engines
python -m streamdiffusion.acceleration.tensorrt.build --fp8 --model stabilityai/sd-turbo
python -m streamdiffusion.acceleration.tensorrt.build --fp8 --model stabilityai/sdxl-turbo

# Seed thresholds from FP8 baseline (run once, check in thresholds.yaml)
python tests/quality/run_compare.py --baseline
```

## Running the harness

```powershell
# Full comparison (both fixtures)
python tests/quality/run_compare.py

# Single fixture
python tests/quality/run_compare.py --fixture sdxl_turbo_img2img_plain
```

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | All fixtures pass SSIM/LPIPS thresholds |
| 1 | One or more fixtures fail thresholds |
| 2 | Manifest version mismatch — results would be meaningless |
| 3 | Golden PNG missing — run `regenerate_golden.py --update-manifest` first |

## Refreshing goldens after a dep bump

```powershell
pip install -r requirements.txt -r requirements-dev.txt  # upgrade deps
python tests/quality/regenerate_golden.py --update-manifest  # regenerate + re-hash
python tests/quality/run_compare.py --baseline              # re-seed thresholds
```

## File layout

```
tests/quality/
├── run_compare.py           # orchestrator — runs FP8, computes metrics, checks thresholds
├── regenerate_golden.py     # generates FP16-TRT goldens and updates manifest hashes
├── manifest.json            # pinned dep versions + golden sha256 (abort on mismatch)
├── thresholds.yaml          # SSIM/LPIPS floors per fixture (seed with --baseline)
├── fixtures/                # fixture parameter files
│   ├── sd_turbo_img2img_plain.json
│   └── sdxl_turbo_img2img_plain.json
├── goldens/                 # FP16-TRT reference PNGs (generated, then checked in)
│   ├── sd_turbo_img2img_plain.png
│   └── sdxl_turbo_img2img_plain.png
└── outputs/                 # generated on run — gitignored
    ├── *_fp8.png            # FP8 output for each fixture
    ├── *_comparison.png     # side-by-side comparison
    └── report.json          # SSIM/LPIPS results
```

## Phase 3 gate

This harness is the prerequisite for Phase 3 (feature-aware calibration + Q/DQ exclusion
narrowing). Do not narrow exclusions until both fixtures pass at stable thresholds across
at least two independent runs.
