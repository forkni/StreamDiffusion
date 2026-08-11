"""
Schedule diagnostics probe -- do_add_noise / t_index investigation.

Context: chasing noisy, "boiling" output at t_index_list: [49], the initial
suspicion was that do_add_noise was injecting noise at the prediction step.
Tracing it end to end showed the opposite: do_add_noise is read in exactly
two places in pipeline.py's predict_x0_batch, both gated (batched-LCM buffer
rebuild needs denoising_steps_num > 1; the sequential/non-batched loop needs
a non-final step) -- a single-index config reaches neither, so toggling the
flag cannot change a single pixel. The noise actually reaching the frame
comes from encode_image's add_noise call, which is ungated. Deriving the
realised t / sqrt(alpha) / sqrt(1-alpha) / c_skip / c_out for a given
t_index_list by hand is exactly the kind of thing that should be printed,
not recomputed on paper every time -- that's what this script does.

Pure scheduler math: builds only the scheduler (from the target model's
scheduler_config.json, via <SchedulerClass>.load_config(..., subfolder=
"scheduler")), no pipeline load, no UNet/VAE, no TensorRT, no TouchDesigner.
Runs in seconds. Deliberately runs on CPU/float32 regardless of the target
config's device/dtype -- the numbers are for human diagnostic reading, not a
bit-exact reproduction of GPU fp16 execution, and CPU keeps this runnable
without a CUDA-visible machine.

Reuses rather than reimplements the same helpers pipeline.py's prepare() and
_log_schedule_diagnostics() use, so this script and the live pipeline can
never silently drift apart on what a "schedule" is:
  - param_schema.materialise_timestep_grid -- grid construction + the
    _SPACING_SAMPLERS override (also shared with wrapper.py's fp8
    calibration path).
  - param_schema.compute_sub_timesteps -- t_index -> realised timestep.
  - param_schema.bleed_risk_message -- the same ghost-bleed check
    _log_schedule_diagnostics and stream_parameter_updater.py's
    _update_timestep_calculations both call.
  - StreamDiffusion._initialize_scheduler / ._get_spaced_timesteps /
    ._get_scheduler_scalings -- called unbound (with a lightweight stub
    standing in for self) rather than copied, since none of the three touch
    any StreamDiffusion state beyond scheduler/device/dtype and instantiating
    a real StreamDiffusion would require an already-loaded pipe.

Usage (from the StreamDiffusion/ directory):
    venv/Scripts/python.exe scripts/probe_schedule.py
    venv/Scripts/python.exe scripts/probe_schedule.py --t-index-list 30 45
    venv/Scripts/python.exe scripts/probe_schedule.py --sweep
    venv/Scripts/python.exe scripts/probe_schedule.py --config path/to/other.yaml
"""

import argparse
import logging
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import yaml

# ---------------------------------------------------------------------------
# Repo root on sys.path so `from streamdiffusion` works without install
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch
from diffusers import LCMScheduler, TCDScheduler

from streamdiffusion.param_schema import (
    DEFAULTS,
    bleed_risk_message,
    compute_sub_timesteps,
    materialise_timestep_grid,
)
from streamdiffusion.pipeline import StreamDiffusion

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("probe_schedule")

_DEFAULT_CONFIG = _REPO_ROOT / "StreamDiffusionTD" / "td_config.yaml"
# Matches config.py's _extract_wrapper_params default (config.py:107) — model_id
# has no param_schema.PARAMS entry since it's construction-time only.
_DEFAULT_MODEL_ID = "stabilityai/sd-turbo"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"YAML config to read defaults from (default: the live TD config, {_DEFAULT_CONFIG}). "
        "Missing file/fields fall back to the same defaults config.py uses.",
    )
    p.add_argument("--model-id", default=None, help="Override model_id / HF repo id")
    p.add_argument("--t-index-list", type=int, nargs="+", default=None, metavar="IDX")
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument("--scheduler", choices=["lcm", "tcd"], default=None)
    p.add_argument("--sampler", choices=["simple", "sgm_uniform", "normal", "ddim", "beta", "karras"], default=None)
    p.add_argument("--use-denoising-batch", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--do-add-noise", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument(
        "--sweep",
        action="store_true",
        help="Print every grid index 0..num_inference_steps-1 instead of just t_index_list, "
        "so a t_index can be chosen by target noise authority instead of trial and error.",
    )
    return p.parse_args()


def _load_yaml_defaults(path: Path) -> Dict[str, Any]:
    if not path.exists():
        logger.warning("config not found at %s -- using built-in defaults for anything not passed on the CLI", path)
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_scheduler(model_id: str, scheduler_type: str, sampler_type: str):
    """Materialise the scheduler exactly as StreamDiffusion._initialize_scheduler
    does (pipeline.py, called from __init__ and update_stream_params), except
    the config comes from the model repo's scheduler_config.json directly
    instead of an already-loaded pipe.scheduler.config — prepare() has not
    run and no pipe is loaded, so this is the only config source available
    that doesn't require one. load_config is inherited unmodified from
    diffusers' ConfigMixin and reads whatever scheduler_config.json lives in
    the repo's "scheduler" subfolder regardless of which scheduler subclass
    calls it (this is diffusers' own supported mechanism for swapping
    scheduler types), so calling it on LCMScheduler here does not bias the
    result toward LCM.
    """
    try:
        config = LCMScheduler.load_config(model_id, subfolder="scheduler")
    except Exception as e:
        raise RuntimeError(
            f"Could not load scheduler config for model_id={model_id!r} (subfolder='scheduler'): {e}\n"
            "Pass --model-id to point at a different repo, or ensure it's present in the local HF cache."
        ) from e
    # _initialize_scheduler's body never reads `self` (verified against the
    # current source) — it's a pure function of (scheduler_type, sampler_type,
    # config) — so it's safe to call unbound with self=None rather than
    # duplicating its sampler_config table here.
    return StreamDiffusion._initialize_scheduler(None, scheduler_type, sampler_type, config)


def _scaling_stub(scheduler, device: torch.device, dtype: torch.dtype) -> Any:
    """Duck-typed stand-in for `self` so StreamDiffusion._get_spaced_timesteps
    and ._get_scheduler_scalings — both of which only ever read
    self.scheduler / self.device / self.dtype — can be called unbound,
    without constructing a real StreamDiffusion (which needs an already-
    loaded pipe this probe deliberately never loads)."""
    return types.SimpleNamespace(scheduler=scheduler, device=device, dtype=dtype)


def _schedule_row(scheduler, stub, timestep) -> Dict[str, float]:
    alpha_prod_t_sqrt = scheduler.alphas_cumprod[timestep].sqrt()
    beta_prod_t_sqrt = (1 - scheduler.alphas_cumprod[timestep]).sqrt()
    c_skip, c_out = StreamDiffusion._get_scheduler_scalings(stub, timestep)
    return {
        "t": int(timestep),
        "a_sqrt": float(alpha_prod_t_sqrt),
        "b_sqrt": float(beta_prod_t_sqrt),
        "c_skip": float(c_skip),
        "c_out": float(c_out),
    }


_TABLE_HEADER = " idx |    t |   a_sqrt |  b_sqrt  |   b/a   |   c_skip |    c_out"


def _format_row(idx: int, row: Dict[str, float]) -> str:
    ratio = (row["b_sqrt"] / row["a_sqrt"]) if row["a_sqrt"] else float("inf")
    return (
        f" {idx:>3} | {row['t']:>4} | {row['a_sqrt']:>8.4f} | {row['b_sqrt']:>8.4f} | "
        f"{ratio:>7.4f} | {row['c_skip']:>8.2e} | {row['c_out']:>8.6f}"
    )


def _do_add_noise_verdict(scheduler, use_denoising_batch: bool, denoising_steps_num: int) -> tuple:
    """Same three-way branch _log_schedule_diagnostics prints from the live
    pipeline (pipeline.py's predict_x0_batch) — kept in sync by hand since
    it's a summary of branches, not extractable arithmetic like the helpers
    above. See pipeline.py:927-942 for the identical logic."""
    if isinstance(scheduler, TCDScheduler):
        return False, "TCD scheduler never reads do_add_noise (predict_x0_batch TCD branch, pipeline.py:1448-1458)"
    elif use_denoising_batch and isinstance(scheduler, LCMScheduler):
        reachable = denoising_steps_num > 1
        return reachable, (
            "batched-LCM buffer rebuild only reads do_add_noise when denoising_steps_num > 1 "
            f"(pipeline.py:1393,1420); denoising_steps_num={denoising_steps_num}"
        )
    else:
        reachable = denoising_steps_num > 1
        return reachable, (
            "sequential/non-batched loop only reads do_add_noise on a non-final step "
            f"(pipeline.py:1463-1464); t_index_list has {denoising_steps_num} "
            f"entr{'y' if denoising_steps_num == 1 else 'ies'}"
        )


def main() -> None:
    args = _parse_args()
    cfg = _load_yaml_defaults(args.config)

    model_id = args.model_id or cfg.get("model_id", _DEFAULT_MODEL_ID)
    t_index_list = args.t_index_list or cfg.get("t_index_list", list(DEFAULTS["t_index_list"]))
    num_inference_steps = args.num_inference_steps or cfg.get("num_inference_steps", DEFAULTS["num_inference_steps"])
    # scheduler/sampler are construction-time-only (absent from param_schema.PARAMS,
    # see pipeline.py's StreamDiffusion.__init__) — fall back to its own defaults.
    scheduler_type = args.scheduler or cfg.get("scheduler", "lcm")
    sampler_type = args.sampler or cfg.get("sampler", "normal")
    use_denoising_batch = (
        args.use_denoising_batch if args.use_denoising_batch is not None else cfg.get("use_denoising_batch", True)
    )
    do_add_noise = args.do_add_noise if args.do_add_noise is not None else cfg.get("do_add_noise", True)

    device = torch.device("cpu")
    dtype = torch.float32

    scheduler = _build_scheduler(model_id, scheduler_type, sampler_type)
    stub = _scaling_stub(scheduler, device, dtype)

    def _get_spaced_timesteps(spacing: str, n: int) -> torch.Tensor:
        return StreamDiffusion._get_spaced_timesteps(stub, spacing, n)

    timesteps = materialise_timestep_grid(scheduler, num_inference_steps, sampler_type, device, _get_spaced_timesteps)

    lines: List[str] = []
    lines.append(
        f"grid: scheduler={type(scheduler).__name__} sampler={sampler_type} "
        f"steps={num_inference_steps} model_id={model_id!r}"
    )
    lines.append(f"t_index_list={list(t_index_list)}")
    lines.append(_TABLE_HEADER)

    sub_timesteps = compute_sub_timesteps(timesteps, list(t_index_list))
    rows = [_schedule_row(scheduler, stub, t) for t in sub_timesteps]
    for idx, (t_idx, row) in enumerate(zip(t_index_list, rows)):
        lines.append(_format_row(t_idx, row))

    lines.append("")
    lines.append("entry noising (encode_image, ALWAYS on -- not gated by do_add_noise):")
    first = rows[0]
    lines.append(f"  x_t = {first['a_sqrt']:.4f}*z_img + {first['b_sqrt']:.4f}*init_noise[0]")
    lines.append("")

    denoising_steps_num = len(t_index_list)
    reachable, reason = _do_add_noise_verdict(scheduler, use_denoising_batch, denoising_steps_num)
    verdict_line = f"do_add_noise: configured={do_add_noise} EFFECTIVE={reachable}"
    if not reachable:
        verdict_line += f"\n  unreachable: {reason}"
    lines.append(verdict_line)

    if len(rows) > 1:
        inter_step_betas = [r["b_sqrt"] for r in rows[1:]]
        bleed_msg = bleed_risk_message(inter_step_betas, list(t_index_list), use_denoising_batch, do_add_noise)
        if bleed_msg is not None:
            lines.append(f"ghost-bleed risk: {bleed_msg}")

    logger.info("\n".join(lines))

    if args.sweep:
        sweep_indices = list(range(num_inference_steps))
        sweep_sub_timesteps = compute_sub_timesteps(timesteps, sweep_indices)
        sweep_rows = [_schedule_row(scheduler, stub, t) for t in sweep_sub_timesteps]
        sweep_lines = ["", f"sweep: idx 0..{num_inference_steps - 1}", _TABLE_HEADER]
        for idx, row in zip(sweep_indices, sweep_rows):
            marker = " <-- t_index_list" if idx in t_index_list else ""
            sweep_lines.append(_format_row(idx, row) + marker)
        logger.info("\n".join(sweep_lines))


if __name__ == "__main__":
    main()
