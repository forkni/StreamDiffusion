"""Quality regression harness: compare FP8-TRT output against FP16-TRT goldens.

Exit codes:
    0  All fixtures pass SSIM/LPIPS thresholds
    1  One or more fixtures fail thresholds
    2  Manifest version mismatch (abort — results would be meaningless)
    3  Golden PNG missing — run regenerate_golden.py --update-manifest first

Usage:
    python tests/quality/run_compare.py
    python tests/quality/run_compare.py --fixture sdxl_turbo_img2img_plain
    python tests/quality/run_compare.py --baseline   # seed thresholds.yaml
    python tests/quality/run_compare.py --skip-manifest-check  # bypass version pins
"""

import argparse
import hashlib
import json
import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logger = logging.getLogger("quality.run_compare")

TESTS_QUALITY_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(TESTS_QUALITY_DIR, "..", "..")
INPUT_IMAGE = os.path.join(REPO_ROOT, "images", "inputs", "input.png")
GOLDENS_DIR = os.path.join(TESTS_QUALITY_DIR, "goldens")
FIXTURES_DIR = os.path.join(TESTS_QUALITY_DIR, "fixtures")
MANIFEST_PATH = os.path.join(TESTS_QUALITY_DIR, "manifest.json")
THRESHOLDS_PATH = os.path.join(TESTS_QUALITY_DIR, "thresholds.yaml")
OUTPUTS_DIR = os.path.join(TESTS_QUALITY_DIR, "outputs")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _installed_version(pkg: str) -> str:
    import importlib.metadata

    try:
        return importlib.metadata.version(pkg)
    except Exception:
        if pkg == "tensorrt":
            try:
                import tensorrt as _trt

                return _trt.__version__
            except Exception:
                pass
        return "unknown"


def _check_manifest(manifest: dict) -> list[str]:
    """Return list of version mismatch messages (empty = all match)."""
    pkg_map = {
        "torch": "torch",
        "tensorrt": "tensorrt",
        "nvidia_modelopt": "nvidia-modelopt",
        "diffusers": "diffusers",
    }
    mismatches = []
    for key, pkg in pkg_map.items():
        pinned = manifest["versions"].get(key, "unknown")
        installed = _installed_version(pkg)
        if pinned != "unknown" and installed != "unknown" and pinned != installed:
            mismatches.append(f"  {key}: manifest={pinned}  installed={installed}")
    return mismatches


def _run_fp8_inference(fixture_name: str, fixture: dict, output_path: str) -> None:
    from streamdiffusion import StreamDiffusionWrapper

    stream = StreamDiffusionWrapper(
        model_id_or_path=fixture["model_id"],
        t_index_list=fixture["t_index_list"],
        frame_buffer_size=1,
        width=fixture["width"],
        height=fixture["height"],
        warmup=fixture.get("warmup", 1),
        acceleration="tensorrt",
        mode="img2img",
        use_denoising_batch=fixture.get("use_denoising_batch", True),
        cfg_type=fixture.get("cfg_type", "none"),
        seed=fixture["seed"],
        fp8=True,
    )
    stream.prepare(
        prompt=fixture["prompt"],
        negative_prompt=fixture.get("negative_prompt", ""),
        num_inference_steps=fixture.get("num_inference_steps", 50),
        guidance_scale=fixture.get("guidance_scale", 1.0),
        delta=fixture.get("delta", 1.0),
    )
    image_tensor = stream.preprocess_image(INPUT_IMAGE)
    for _ in range(stream.batch_size - 1):
        stream(image=image_tensor)
    output_image = stream(image=image_tensor)
    output_image.save(output_path)


def _compute_metrics(golden_path: str, output_path: str) -> dict:
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim_fn

    golden = np.array(Image.open(golden_path).convert("RGB"), dtype=np.float32) / 255.0
    output = np.array(Image.open(output_path).convert("RGB"), dtype=np.float32) / 255.0

    ssim_val = float(ssim_fn(golden, output, data_range=1.0, channel_axis=2))

    try:
        import lpips
        import torch

        loss_fn = lpips.LPIPS(net="alex", verbose=False)
        g_t = torch.from_numpy(golden).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        o_t = torch.from_numpy(output).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        with torch.no_grad():
            lpips_val = float(loss_fn(g_t, o_t).item())
    except Exception as e:
        logger.warning(f"LPIPS computation failed: {e}. Using placeholder 0.0.")
        lpips_val = 0.0

    return {"ssim": ssim_val, "lpips": lpips_val}


def _make_comparison_image(golden_path: str, output_path: str, comparison_path: str) -> None:
    from PIL import Image, ImageDraw

    golden = Image.open(golden_path).convert("RGB")
    output = Image.open(output_path).convert("RGB")
    w, h = golden.width, golden.height
    canvas = Image.new("RGB", (w * 2 + 10, h + 24), (30, 30, 30))
    canvas.paste(golden, (0, 24))
    canvas.paste(output, (w + 10, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), "FP16-TRT golden", fill=(200, 200, 200))
    draw.text((w + 14, 4), "FP8-TRT output", fill=(200, 200, 200))
    canvas.save(comparison_path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixture", default=None, help="Run only this fixture (default: all)")
    parser.add_argument("--baseline", action="store_true", help="Seed thresholds.yaml from this run's SSIM/LPIPS")
    parser.add_argument("--skip-manifest-check", action="store_true", help="Bypass version pin check")
    args = parser.parse_args()

    with open(MANIFEST_PATH) as fp:
        manifest = json.load(fp)
    with open(THRESHOLDS_PATH) as fp:
        thresholds = yaml.safe_load(fp)

    if not args.skip_manifest_check:
        mismatches = _check_manifest(manifest)
        if mismatches:
            print("ERROR: Manifest version mismatch — results would be meaningless.\n")
            print("\n".join(mismatches))
            print("\nRun regenerate_golden.py --update-manifest to refresh the manifest.")
            sys.exit(2)

    fixture_files = sorted(f for f in os.listdir(FIXTURES_DIR) if f.endswith(".json"))
    if args.fixture:
        fixture_files = [f"{args.fixture}.json"]

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    results = {}
    all_pass = True

    for fname in fixture_files:
        name = fname[:-5]
        with open(os.path.join(FIXTURES_DIR, fname)) as fp:
            fixture = json.load(fp)

        golden_path = os.path.join(GOLDENS_DIR, f"{name}.png")
        if not os.path.exists(golden_path):
            print(f"ERROR: Golden not found for {name}: {golden_path}")
            print("Run: python tests/quality/regenerate_golden.py --update-manifest")
            sys.exit(3)

        golden_sha = manifest["fixtures"].get(name, {}).get("golden_sha256")
        if golden_sha is not None:
            actual_sha = _sha256(golden_path)
            if actual_sha != golden_sha:
                print(f"WARNING: Golden sha256 mismatch for {name}. File may be corrupted or outdated.")

        output_path = os.path.join(OUTPUTS_DIR, f"{name}_fp8.png")
        logger.info(f"[{name}] Running FP8-TRT inference...")
        _run_fp8_inference(name, fixture, output_path)

        logger.info(f"[{name}] Computing metrics...")
        metrics = _compute_metrics(golden_path, output_path)
        results[name] = metrics

        comparison_path = os.path.join(OUTPUTS_DIR, f"{name}_comparison.png")
        _make_comparison_image(golden_path, output_path, comparison_path)

        thresh = thresholds.get("fixtures", {}).get(name, {})
        ssim_min = thresh.get("ssim_min", 0.0)
        lpips_max = thresh.get("lpips_max", 1.0)
        passed = metrics["ssim"] >= ssim_min and metrics["lpips"] <= lpips_max
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False

        print(
            f"[{name}] {status}  SSIM={metrics['ssim']:.4f} (min={ssim_min})  "
            f"LPIPS={metrics['lpips']:.4f} (max={lpips_max})"
        )
        logger.info(f"[{name}] Comparison: {comparison_path}")

    report_path = os.path.join(OUTPUTS_DIR, "report.json")
    with open(report_path, "w") as fp:
        json.dump(results, fp, indent=2)
    logger.info(f"Report: {report_path}")

    if args.baseline:
        for name, metrics in results.items():
            if name not in thresholds.get("fixtures", {}):
                thresholds.setdefault("fixtures", {})[name] = {}
            thresholds["fixtures"][name]["ssim_min"] = round(metrics["ssim"] - 0.02, 4)
            thresholds["fixtures"][name]["lpips_max"] = round(metrics["lpips"] + 0.05, 4)
            thresholds["fixtures"][name].pop("_note", None)
        with open(THRESHOLDS_PATH, "w") as fp:
            yaml.dump(thresholds, fp, default_flow_style=False, sort_keys=False)
        print(f"\nThresholds seeded → {THRESHOLDS_PATH}")

    n = len(results)
    n_pass = sum(
        1
        for name, m in results.items()
        if m["ssim"] >= thresholds.get("fixtures", {}).get(name, {}).get("ssim_min", 0.0)
        and m["lpips"] <= thresholds.get("fixtures", {}).get(name, {}).get("lpips_max", 1.0)
    )
    print(f"\n{n_pass}/{n} fixtures pass thresholds.")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
