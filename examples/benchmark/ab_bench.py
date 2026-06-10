"""
ab_bench.py — A/B benchmark harness for StreamDiffusion performance work.

Runs warmup + timed frames, records per-frame CUDA-event timings and per-region
profiler stats, and writes a JSON keyed by git-SHA + config-hash so before/after
runs can be compared without manual bookkeeping.

USAGE
-----
# Bare-pipeline (inline defaults, no GPU_PROFILER env needed for frame timing):
python examples/benchmark/ab_bench.py

# With a full config (includes ControlNet / IPAdapter / ESRGAN):
GPU_PROFILER=1 python examples/benchmark/ab_bench.py --config path/to/config.yaml

# Selective config override:
GPU_PROFILER=1 python examples/benchmark/ab_bench.py \\
    --config configs/cn_tile.yaml \\
    --iterations 200 \\
    --warmup 20 \\
    --image /path/to/input.jpg \\
    --style-image /path/to/style.jpg \\
    --output-dir examples/benchmark/results

# Save output frames as PNGs for visual before/after comparison:
python examples/benchmark/ab_bench.py --save-goldens --n-golden-frames 5

READING RESULTS
---------------
Each run writes:
    <output_dir>/<git_sha>_<config_hash>_<YYYYMMDD_HHMMSS>.json

The JSON contains:
  - "run": metadata (sha, config_hash, config_path, timestamp, iterations, warmup)
  - "frame_ms": per-frame CUDA timings {p50, p95, p99, mean, min, max}
  - "fps": fps stats derived from frame timings
  - "regions": per-region profiler stats (only present when GPU_PROFILER=1)

COMPARISON
----------
Diff two runs:
    python -c "
    import json, sys
    a, b = [json.load(open(p)) for p in sys.argv[1:3]]
    for k in ['p50','p95','p99']:
        va, vb = a['frame_ms'][k], b['frame_ms'][k]
        print(f'frame {k}: {va:.2f} -> {vb:.2f} ms ({vb-va:+.2f} ms)')
    " results/sha1_hash1_*.json results/sha2_hash2_*.json

GPU_PROFILER env vars
---------------------
  GPU_PROFILER=1          - enable region timing
  GPU_PROFILER_NVTX=0     - disable NVTX (required when CUDA graphs active)
  GPU_PROFILER_EVENTS=1   - (default) CUDA-event timing
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


# ── repo root on sys.path so streamdiffusion is importable without install ─────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from streamdiffusion.tools.gpu_profiler import configure as _prof_configure  # noqa: E402
from streamdiffusion.tools.gpu_profiler import profiler  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _git_sha() -> str:
    """Return the current HEAD short SHA, or 'unknown' when git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except FileNotFoundError:
        return "unknown"


def _config_hash(config: Dict[str, Any]) -> str:
    """Return an 8-char SHA-1 of the sorted JSON-serialised config."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:8]


def _make_synthetic_image(width: int, height: int) -> "PIL.Image.Image":
    """Create a solid grey PIL image so the benchmark runs without a real photo."""
    import PIL.Image
    import PIL.ImageDraw

    img = PIL.Image.new("RGB", (width, height), color=(128, 128, 128))
    draw = PIL.ImageDraw.Draw(img)
    # Add a simple pattern so it's not a zero tensor (avoids edge-case norms).
    for x in range(0, width, 64):
        draw.line([(x, 0), (x, height)], fill=(160, 160, 160), width=1)
    for y in range(0, height, 64):
        draw.line([(0, y), (width, y)], fill=(160, 160, 160), width=1)
    return img


def _load_or_synth_image(path: Optional[str], width: int, height: int) -> "PIL.Image.Image":
    if path:
        import PIL.Image
        return PIL.Image.open(path).convert("RGB").resize((width, height))
    return _make_synthetic_image(width, height)


def _percentile_stats(samples: List[float]) -> Dict[str, float]:
    arr = np.array(samples)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _fps_stats(frame_ms_stats: Dict[str, float]) -> Dict[str, float]:
    """Convert ms-per-frame stats to FPS stats (note: p50 ms → median FPS, etc.)."""
    return {
        k: round(1000.0 / v, 2) if v > 0 else 0.0
        for k, v in frame_ms_stats.items()
    }


# ──────────────────────────────────────────────────────────────────────────────
# Core benchmark loop
# ──────────────────────────────────────────────────────────────────────────────


def _to_pil(frame: Any) -> "Optional[PIL.Image.Image]":
    """Best-effort conversion of a pipeline output to PIL Image for golden saving."""
    import PIL.Image

    if isinstance(frame, PIL.Image.Image):
        return frame
    if isinstance(frame, np.ndarray):
        arr = frame
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = arr.transpose(1, 2, 0)  # CHW → HWC
        return PIL.Image.fromarray(arr.squeeze())
    if hasattr(frame, "cpu"):  # torch.Tensor
        arr = frame.cpu().float().numpy()
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = arr.transpose(1, 2, 0)
        return PIL.Image.fromarray(arr.squeeze())
    return None


def _run_loop(
    stream: Any,
    image_tensor: Any,
    iterations: int,
    warmup: int,
    n_capture: int = 0,
) -> Tuple[List[float], List[Any]]:
    """Warmup then time `iterations` frames.

    Returns
    -------
    frame_times : list of per-frame ms values (length == iterations)
    captured    : first ``n_capture`` raw pipeline outputs (empty when n_capture=0)
    """

    # ── warmup (no timing) ─────────────────────────────────────────────────
    print(f"[ab_bench] Warming up ({warmup} frames)…")
    for _ in range(warmup):
        stream(image=image_tensor)

    torch.cuda.synchronize()

    # ── timed loop ─────────────────────────────────────────────────────────
    print(f"[ab_bench] Timing {iterations} frames…")
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    frame_times: List[float] = []
    captured: List[Any] = []

    for _ in tqdm(range(iterations)):
        start_evt.record()
        output = stream(image=image_tensor)
        end_evt.record()
        torch.cuda.synchronize()
        frame_times.append(start_evt.elapsed_time(end_evt))
        if n_capture > 0 and len(captured) < n_capture:
            captured.append(output)

    return frame_times, captured


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────


def run(
    config: Optional[str] = None,
    iterations: int = 200,
    warmup: int = 10,
    image: Optional[str] = None,
    style_image: Optional[str] = None,
    output_dir: str = "examples/benchmark/results",
    # ── bare-pipeline defaults (ignored when --config is provided) ─────────
    model_id: str = "KBlueLeaf/kohaku-v2.1",
    width: int = 512,
    height: int = 512,
    prompt: str = "1girl with brown dog hair, thick glasses, smiling",
    negative_prompt: str = "bad image, bad quality",
    acceleration: str = "tensorrt",
    # ── behaviour flags ────────────────────────────────────────────────────
    gpu_profiler: bool = False,
    nvtx: bool = False,
    # ── golden capture ─────────────────────────────────────────────────────
    save_goldens: bool = False,
    n_golden_frames: int = 5,
) -> None:
    """
    A/B benchmark harness for StreamDiffusion performance improvements.

    Parameters
    ----------
    config : str, optional
        Path to a StreamDiffusion YAML/JSON config file.  When provided the
        full config (including ControlNet, IPAdapter, ESRGAN) drives the run.
        CLI flags below are ignored except --iterations/--warmup/--image/
        --style-image/--output-dir.
    iterations : int
        Number of timed frames (after warmup).  Default 200.
    warmup : int
        Warmup frames (not timed).  Overrides config.warmup when --config given.
        Default 10.
    image : str, optional
        Path to input image.  Synthetic grey image used when absent.
    style_image : str, optional
        Path to style/reference image for IPAdapter.  Synthetic image used when
        absent and an IPAdapter is active.
    output_dir : str
        Directory for JSON result files.  Created if absent.
    model_id : str
        HuggingFace model ID for the bare-pipeline (no --config) path.
    width / height : int
        Image resolution for the bare-pipeline path.
    prompt / negative_prompt : str
        Prompts for the bare-pipeline path.
    acceleration : str
        Acceleration backend for the bare-pipeline path ("tensorrt", "xformers", "none").
    gpu_profiler : bool
        Activate region-level profiling (equivalent to GPU_PROFILER=1).
        NVTX is off by default; set --nvtx to enable (breaks CUDA graphs).
    nvtx : bool
        Enable NVTX markers (only useful for Nsight Systems; disable with CUDA graphs).
    save_goldens : bool
        Capture the first ``n_golden_frames`` output frames and save them as
        PNG files alongside the JSON result.  Useful for visual before/after
        comparison after pipeline changes (e.g. the antialias resize fix).
        Files are named ``<sha>_<cfg_hash>_golden_NN.png`` in ``output_dir``.
    n_golden_frames : int
        Number of output frames to capture when ``--save-goldens`` is set.
        Default 5.
    """
    # ── activate profiler (env var takes precedence, flag is an alternative) ─
    _prof_configure(enabled=gpu_profiler, nvtx=nvtx)  # reads GPU_PROFILER env internally
    # After configure, profiler is either active or a null-op depending on env/flag.

    # ── build the stream ───────────────────────────────────────────────────
    bench_config: Dict[str, Any] = {}

    if config is not None:
        # Config-file path (CN / IPA / ESRGAN configs live here)
        from streamdiffusion.config import create_wrapper_from_config, load_config

        bench_config = load_config(config)
        # CLI overrides for iteration control
        if warmup != 10:
            bench_config["warmup"] = warmup
        print(f"[ab_bench] Building stream from config: {config}")
        stream = create_wrapper_from_config(bench_config)
        _width = bench_config.get("width", 512)
        _height = bench_config.get("height", 512)
        _has_ipa = bool(bench_config.get("ipadapters") or bench_config.get("use_ipadapter"))
    else:
        # Bare-pipeline inline defaults (no config file)
        from streamdiffusion import StreamDiffusionWrapper

        bench_config = {
            "model_id": model_id,
            "width": width,
            "height": height,
            "acceleration": acceleration,
            "warmup": warmup,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
        }
        print(f"[ab_bench] Building bare-pipeline stream (model={model_id})")
        stream = StreamDiffusionWrapper(
            model_id_or_path=model_id,
            t_index_list=[32, 45],
            mode="img2img",
            frame_buffer_size=1,
            width=width,
            height=height,
            warmup=warmup,
            acceleration=acceleration,
            use_tiny_vae=True,
            enable_similar_image_filter=False,
            use_denoising_batch=True,
            cfg_type="self",
            seed=2,
        )
        stream.prepare(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            guidance_scale=1.2,
            delta=0.5,
        )
        _width, _height = width, height
        _has_ipa = False

    # ── prepare input image ────────────────────────────────────────────────
    pil_image = _load_or_synth_image(image, _width, _height)
    image_tensor = stream.preprocess_image(pil_image)
    print(f"[ab_bench] Input: {'file ' + image if image else 'synthetic'} ({_width}×{_height})")

    # ── inject style image for IPAdapter if needed ─────────────────────────
    if _has_ipa:
        pil_style = _load_or_synth_image(style_image, _width, _height)
        print(f"[ab_bench] Style: {'file ' + style_image if style_image else 'synthetic'}")
        stream.update_style_image(pil_style)

    # ── run the loop ───────────────────────────────────────────────────────
    n_capture = n_golden_frames if save_goldens else 0
    frame_times, captured_frames = _run_loop(stream, image_tensor, iterations, warmup, n_capture)

    # ── flush profiler and collect region stats ────────────────────────────
    profiler.flush()
    profiler.report()

    # ── compute stats ─────────────────────────────────────────────────────
    frame_stats = _percentile_stats(frame_times)
    fps_stats = _fps_stats(frame_stats)

    sha = _git_sha()
    cfg_hash = _config_hash(bench_config)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_fname = f"{sha}_{cfg_hash}_{timestamp}.json"

    # ── collect region stats if profiler is active ─────────────────────────
    region_stats: List[Dict] = []
    try:
        inner = object.__getattribute__(profiler, "_inner")
        if hasattr(inner, "_regions"):
            region_stats = [s.to_dict() for s in inner._regions.values()]
    except (AttributeError, TypeError):
        pass

    result: Dict[str, Any] = {
        "run": {
            "git_sha": sha,
            "config_hash": cfg_hash,
            "config_path": config,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "iterations": iterations,
            "warmup": warmup,
            "width": _width,
            "height": _height,
        },
        "frame_ms": {k: round(v, 3) for k, v in frame_stats.items()},
        "fps": fps_stats,
        "regions": region_stats,
    }

    # ── write JSON ─────────────────────────────────────────────────────────
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    result_path = out_path / result_fname

    # Also export profiler's own JSON if active
    if region_stats:
        profiler.export_stats(str(out_path / f"{sha}_{cfg_hash}_{timestamp}_regions.json"))

    with open(result_path, "w") as fh:
        json.dump(result, fh, indent=2)

    # ── save goldens ───────────────────────────────────────────────────────
    if save_goldens and captured_frames:
        import PIL.Image  # noqa: F811  (PIL already imported transitively above)

        saved = 0
        for i, frame in enumerate(captured_frames):
            pil = _to_pil(frame)
            if pil is None:
                print(f"[ab_bench] --save-goldens: cannot serialise frame {i} (type {type(frame).__name__}), skipping")
                continue
            golden_path = out_path / f"{sha}_{cfg_hash}_golden_{i:02d}.png"
            pil.save(str(golden_path))
            saved += 1
        print(f"[ab_bench] {saved}/{len(captured_frames)} goldens saved to {out_path}/")

    # ── human-readable summary ─────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"[ab_bench] Results  ({iterations} frames, {warmup} warmup)")
    print(f"  SHA: {sha}  cfg: {cfg_hash}")
    print(f"  frame p50: {frame_stats['p50']:.2f} ms  ({fps_stats['p50']:.1f} FPS)")
    print(f"  frame p95: {frame_stats['p95']:.2f} ms  ({fps_stats['p95']:.1f} FPS)")
    print(f"  frame p99: {frame_stats['p99']:.2f} ms  ({fps_stats['p99']:.1f} FPS)")
    print(f"  frame mean: {frame_stats['mean']:.2f} ms  ({fps_stats['mean']:.1f} FPS)")
    if region_stats:
        top = sorted(region_stats, key=lambda r: r.get("total_ms", 0), reverse=True)[:8]
        print()
        print(f"  {'Region':<30} {'p50':>8}  {'p95':>8}  {'count':>6}")
        print("  " + "-" * 58)
        for r in top:
            print(
                f"  {r['name']:<30} {r['p50_ms']:>7.2f}ms  "
                f"{r['p95_ms']:>7.2f}ms  {r['count']:>6}"
            )
    print()
    print(f"  JSON -> {result_path}")
    print("=" * 60)


if __name__ == "__main__":
    import fire

    fire.Fire(run)
