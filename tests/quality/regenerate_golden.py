"""Generate FP16-TRT golden reference images for the quality regression harness.

Usage:
    python tests/quality/regenerate_golden.py
    python tests/quality/regenerate_golden.py --update-manifest
    python tests/quality/regenerate_golden.py --fixture sdxl_turbo_img2img_plain

Goldens are FP16-TRT engine outputs (not bare PyTorch FP16) so they catch
TRT-specific regressions that matter in production.

--update-manifest  Also updates tests/quality/manifest.json with sha256 hashes
                   of the generated goldens and current installed dep versions.
"""

import argparse
import hashlib
import json
import logging
import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from streamdiffusion import StreamDiffusionWrapper


logger = logging.getLogger("quality.regenerate")

TESTS_QUALITY_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(TESTS_QUALITY_DIR, "..", "..")
INPUT_IMAGE = os.path.join(REPO_ROOT, "images", "inputs", "input.png")
GOLDENS_DIR = os.path.join(TESTS_QUALITY_DIR, "goldens")
FIXTURES_DIR = os.path.join(TESTS_QUALITY_DIR, "fixtures")
MANIFEST_PATH = os.path.join(TESTS_QUALITY_DIR, "manifest.json")
THRESHOLDS_PATH = os.path.join(TESTS_QUALITY_DIR, "thresholds.yaml")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _current_versions() -> dict:
    import importlib.metadata
    versions = {}
    for pkg, key in [
        ("torch", "torch"),
        ("tensorrt", "tensorrt"),
        ("nvidia-modelopt", "nvidia_modelopt"),
        ("diffusers", "diffusers"),
    ]:
        try:
            versions[key] = importlib.metadata.version(pkg)
        except Exception:
            versions[key] = "unknown"
    return versions


def run_fixture(fixture_name: str, fixture: dict) -> str:
    """Run FP16-TRT inference for one fixture and return the saved golden path."""
    golden_path = os.path.join(GOLDENS_DIR, f"{fixture_name}.png")
    os.makedirs(GOLDENS_DIR, exist_ok=True)

    logger.info(f"[{fixture_name}] Running FP16-TRT inference → {golden_path}")

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
    output_image.save(golden_path)

    sha = _sha256(golden_path)
    logger.info(f"[{fixture_name}] Golden saved: {golden_path}  sha256={sha[:16]}...")
    return sha


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update-manifest", action="store_true", help="Update manifest.json with current versions + golden hashes")
    parser.add_argument("--fixture", default=None, help="Run only this fixture (default: all)")
    args = parser.parse_args()

    fixture_files = sorted(f for f in os.listdir(FIXTURES_DIR) if f.endswith(".json"))
    if args.fixture:
        fixture_files = [f"{args.fixture}.json"]

    results = {}
    for fname in fixture_files:
        name = fname[:-5]
        with open(os.path.join(FIXTURES_DIR, fname)) as fp:
            fixture = json.load(fp)
        sha = run_fixture(name, fixture)
        results[name] = sha

    if args.update_manifest:
        with open(MANIFEST_PATH) as fp:
            manifest = json.load(fp)
        manifest["versions"] = _current_versions()
        for name, sha in results.items():
            if name not in manifest["fixtures"]:
                manifest["fixtures"][name] = {}
            manifest["fixtures"][name]["golden_sha256"] = sha
            manifest["fixtures"][name].pop("_note", None)
        with open(MANIFEST_PATH, "w") as fp:
            json.dump(manifest, fp, indent=4)
        logger.info(f"Manifest updated: {MANIFEST_PATH}")

        # Seed thresholds from FP8 baseline if engines already exist,
        # otherwise leave placeholders and print instructions.
        print(
            "\nThreshold seeding: run_compare.py --baseline to measure FP8 vs these goldens "
            "and seed thresholds.yaml with baseline - 0.02 (SSIM) / + 0.05 (LPIPS)."
        )


if __name__ == "__main__":
    main()
