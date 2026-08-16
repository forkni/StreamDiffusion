"""
Phase 1c-i: measure trigger-word gating with OUT-OF-DOMAIN prompts, plus a numeric
d_gate / d_leak ratio -- plain diffusers, no StreamDiffusion wrapper.

Why this exists (see _reviews/ FACT_CHECK notes and the plan): the earlier 2x2 grid
(test_lora_grid_sdxl_base.py) uses "a tall art deco tower, architectural drawing" as its
prompt, which is itself effectively a training caption for this dataset (concept-B
images are art-deco architectural drawings). That prompt cannot distinguish "the
trigger gates nothing" from "this prompt summons the trained concept on its own,
trigger or not" -- both explanations predict the same observed result. This script
adds prompts far outside the training distribution (a dog, a portrait, food) so LoRA
leakage without the trigger and LoRA response with the trigger can be measured
separately, plus keeps the original tower prompt as an in-domain control for context.

Per prompt, three cells at the same seed:
    off          -- LoRA scale 0 (base model)
    on_notrigger -- LoRA at --weight, prompt WITHOUT the trigger word
    on_trigger   -- LoRA at --weight, prompt WITH the trigger word prefixed

Metrics (mean absolute pixel difference over uint8 RGB, 0-255 scale):
    d_leak = mean|on_notrigger - off|         -- how much the LoRA changes an
                                                  UNtriggered prompt (should be ~0
                                                  if the trigger is a true gate)
    d_gate = mean|on_trigger - on_notrigger|  -- how much the trigger word ADDS
    ratio  = d_gate / d_leak                  -- ~0 means inert trigger (style is
                                                  unconditionally on); >>1 means the
                                                  trigger is doing the gating

Also reports mean HSV saturation per cell, since "did it go sepia" is a saturation
question and raw pixel diff alone won't distinguish a sepia shift from unrelated
resampling noise.

Usage:
    venv/Scripts/python scripts/test_lora_gating_measure.py
    venv/Scripts/python scripts/test_lora_gating_measure.py --lora <path to run-3 prior-pred LoRA>
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

os.environ.setdefault("HF_HOME", r"D:\Users\alexk\.cache\huggingface")

from diffusers import AutoencoderTiny, EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline

DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_LORA = r"D:\dev\RESEARCH\OneTrainer\models\lora_sepiagraph_300.safetensors"
DEFAULT_TRIGGER = "sepiagraph"
DEFAULT_NEG_PROMPT = "blurry, low quality"

# (label, prompt, in_domain) -- in_domain prompts are reported separately: they are
# expected to leak somewhat even with a perfectly gated trigger, since the base model's
# own semantics already point at the training subject.
PROMPTS = [
    ("dog_park", "a golden retriever running in a park, photograph", False),
    ("fisherman_portrait", "portrait of an old fisherman, oil painting", False),
    ("ramen_bowl", "a bowl of ramen noodles, food photography", False),
    ("art_deco_tower", "a tall art deco tower, architectural drawing", True),
]


def mean_abs_diff(a: Image.Image, b: Image.Image) -> float:
    xa = np.asarray(a.convert("RGB"), dtype=np.float64)
    xb = np.asarray(b.convert("RGB"), dtype=np.float64)
    return float(np.abs(xa - xb).mean())


def mean_saturation(img: Image.Image) -> float:
    hsv = np.asarray(img.convert("HSV"), dtype=np.float64)
    return float(hsv[..., 1].mean())  # 0-255 scale


def main() -> int:
    parser = argparse.ArgumentParser(description="Out-of-domain trigger-gating measurement (Phase 1c-i)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--lora", default=DEFAULT_LORA)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--trigger", default=DEFAULT_TRIGGER)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEG_PROMPT)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "lora_gating_measure"),
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=True,
    )
    # See test_lora_grid_sdxl_base.py for why TinyVAE is required in this venv
    # (varshith15/diffusers fork's UNetMidBlock2D.forward was never updated for its
    # own kvo_cache tuple-returning Attention API -- breaks the full AutoencoderKL).
    pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16, local_files_only=True)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    print(f"Loading LoRA weights from {args.lora} ...")
    pipe.load_lora_weights(args.lora)

    results = []
    thumbs = []  # (row_label, [off, off_trigger, notrigger, trigger])
    for label, base_prompt, in_domain in PROMPTS:
        row_imgs = []
        cells = {}
        for tag, scale, prompt in (
            ("off", 0.0, base_prompt),
            # Control: LoRA OFF, trigger word still inserted. SDXL's text encoder
            # conditions on token position, so inserting ANY word shifts the whole
            # cross-attention pattern even with zero LoRA scale -- this is the noise
            # floor that a naive on_trigger-vs-on_notrigger diff would conflate with
            # real trigger-gated LoRA behavior.
            ("off_trigger", 0.0, f"{args.trigger}, {base_prompt}"),
            ("on_notrigger", args.weight, base_prompt),
            ("on_trigger", args.weight, f"{args.trigger}, {base_prompt}"),
        ):
            print(f"--- {label}/{tag} --- prompt={prompt!r} scale={scale}")
            gen = torch.Generator(device="cuda").manual_seed(args.seed)
            img = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=gen,
                cross_attention_kwargs={"scale": scale},
            ).images[0]
            path = out_dir / f"{label}__{tag}.png"
            img.save(path)
            cells[tag] = img
            row_imgs.append(img)

        d_leak = mean_abs_diff(cells["on_notrigger"], cells["off"])
        d_gate = mean_abs_diff(cells["on_trigger"], cells["on_notrigger"])
        d_null = mean_abs_diff(cells["off_trigger"], cells["off"])  # token-insertion noise floor
        d_gate_net = max(0.0, d_gate - d_null)  # gating effect net of the token-shift confound
        ratio = (d_gate / d_leak) if d_leak > 1e-6 else float("inf")
        ratio_net = (d_gate_net / d_leak) if d_leak > 1e-6 else float("inf")
        sat = {tag: mean_saturation(img) for tag, img in cells.items()}

        row = {
            "prompt_label": label,
            "in_domain": in_domain,
            "prompt": base_prompt,
            "d_leak": round(d_leak, 3),
            "d_null_token_insertion_floor": round(d_null, 3),
            "d_gate_net_of_null": round(d_gate_net, 3),
            "ratio_net_d_gate_over_d_leak": round(ratio_net, 4) if ratio_net != float("inf") else None,
            "d_gate": round(d_gate, 3),
            "ratio_d_gate_over_d_leak": round(ratio, 4) if ratio != float("inf") else None,
            "saturation_off": round(sat["off"], 2),
            "saturation_off_trigger": round(sat["off_trigger"], 2),
            "saturation_on_notrigger": round(sat["on_notrigger"], 2),
            "saturation_on_trigger": round(sat["on_trigger"], 2),
        }
        results.append(row)
        thumbs.append((label, row_imgs))
        print(
            f"    d_leak={d_leak:.3f}  d_gate={d_gate:.3f}  d_null={d_null:.3f}  "
            f"d_gate_net={d_gate_net:.3f}  ratio_net={row['ratio_net_d_gate_over_d_leak']}  "
            f"sat(off/offtrig/notrig/trig)={sat['off']:.1f}/{sat['off_trigger']:.1f}/"
            f"{sat['on_notrigger']:.1f}/{sat['on_trigger']:.1f}"
        )

    # Contact sheet: rows = prompts, cols = off / off_trigger / on_notrigger / on_trigger
    w, h = thumbs[0][1][0].size
    cols = 4
    rows = len(thumbs)
    contact = Image.new("RGB", (w * cols, h * rows), (32, 32, 32))
    col_labels = ["off", "off_trigger", "on_notrigger", "on_trigger"]
    for r, (label, imgs) in enumerate(thumbs):
        for c, img in enumerate(imgs):
            tile = img.copy()
            draw = ImageDraw.Draw(tile)
            draw.rectangle([0, 0, 220, 24], fill=(0, 0, 0))
            draw.text((4, 4), f"{label}/{col_labels[c]}", fill=(255, 255, 255))
            contact.paste(tile, (c * w, r * h))
    contact_path = out_dir / "contact_sheet.png"
    contact.save(contact_path)

    # Aggregate over out-of-domain prompts only (the real gating measurement -- the
    # in-domain tower prompt is reported for context but excluded from the headline
    # number, since it can leak even under a perfectly gated LoRA). Report both the
    # naive ratio and the null-corrected ratio side by side so the confound is visible
    # rather than silently baked into one number.
    ood = [r for r in results if not r["in_domain"]]
    ood_ratio = [r["ratio_d_gate_over_d_leak"] for r in ood if r["ratio_d_gate_over_d_leak"] is not None]
    ood_ratio_net = [r["ratio_net_d_gate_over_d_leak"] for r in ood if r["ratio_net_d_gate_over_d_leak"] is not None]
    agg_ratio = sum(ood_ratio) / len(ood_ratio) if ood_ratio else None
    agg_ratio_net = sum(ood_ratio_net) / len(ood_ratio_net) if ood_ratio_net else None

    summary = {
        "lora": args.lora,
        "weight": args.weight,
        "seed": args.seed,
        "results": results,
        "mean_ratio_out_of_domain_naive": round(agg_ratio, 4) if agg_ratio is not None else None,
        "mean_ratio_out_of_domain_net_of_null": round(agg_ratio_net, 4) if agg_ratio_net is not None else None,
    }
    summary_path = out_dir / "gating_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("=" * 70)
    print(f"Saved contact sheet: {contact_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Mean d_gate/d_leak, naive:        {summary['mean_ratio_out_of_domain_naive']}")
    print(f"Mean d_gate/d_leak, net of null:  {summary['mean_ratio_out_of_domain_net_of_null']}")
    print("  net-of-null ~0   -> trigger is inert once token-insertion noise is subtracted")
    print("  net-of-null >>1  -> trigger is doing real gating work beyond token-shift noise")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
