"""
SDXL base 1.0 LoRA validation grid — plain diffusers, no StreamDiffusion wrapper.

Runs a 2x2 matrix at a normal (non-turbo) step count: LoRA on/off x trigger word
present/absent, same seed. Settles three things in one pass: did training take,
does the trigger word gate the style, does the style leak without the trigger.

SDXL base 1.0 is not in the TouchDesigner component's model menu (hardcoded to
sdxl-turbo/sd-turbo/dreamshaper-8/openjourney-v4 in Daydream__Text__DaydreamExt__td.py),
so this test has to run outside the component via plain diffusers.

Usage:
    venv/Scripts/python scripts/test_lora_grid_sdxl_base.py
"""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image

os.environ.setdefault("HF_HOME", r"D:\Users\alexk\.cache\huggingface")

from diffusers import AutoencoderTiny, EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline

DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_LORA = r"D:\dev\RESEARCH\OneTrainer\models\lora.safetensors"
DEFAULT_TRIGGER = "sepiagraph"
DEFAULT_PROMPT = "a tall art deco tower, architectural drawing"
DEFAULT_NEG_PROMPT = "blurry, low quality"


def main() -> int:
    parser = argparse.ArgumentParser(description="SDXL base 1.0 LoRA 2x2 validation grid")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--lora", default=DEFAULT_LORA)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--trigger", default=DEFAULT_TRIGGER)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEG_PROMPT)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "lora_sdxl_base_grid"),
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
    # The installed diffusers build (varshith15/diffusers kvo_cache fork, baked in natively —
    # not applied by streamdiffusion's runtime patch) never updated AutoencoderKL's
    # UNetMidBlock2D.forward for its own Attention/AttnProcessor2_0 API change (added
    # kvo_cache, return type became a tuple). The full VAE's self-attention mid-block passes
    # that tuple straight into the next resnet and crashes with
    # "AttributeError: 'tuple' object has no attribute 'dim'". TinyVAE has no attention
    # blocks, so it never touches the broken path — and it's what the component uses by
    # default anyway.
    pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16, local_files_only=True)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    print(f"Loading LoRA weights from {args.lora} ...")
    pipe.load_lora_weights(args.lora)

    cells = {}
    for lora_on in (False, True):
        scale = args.weight if lora_on else 0.0
        for trigger_on in (False, True):
            prompt = f"{args.trigger}, {args.prompt}" if trigger_on else args.prompt
            label = f"lora_{lora_on}_trigger_{trigger_on}"
            print(f"--- {label} --- prompt={prompt!r} scale={scale}")
            gen = torch.Generator(device="cuda").manual_seed(args.seed)
            img = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=gen,
                cross_attention_kwargs={"scale": scale},
            ).images[0]
            path = out_dir / f"{label}.png"
            img.save(path)
            cells[label] = img
            print(f"Saved: {path}")

    w, h = next(iter(cells.values())).size
    grid = Image.new("RGB", (w * 2, h * 2))
    grid.paste(cells["lora_False_trigger_False"], (0, 0))
    grid.paste(cells["lora_False_trigger_True"], (w, 0))
    grid.paste(cells["lora_True_trigger_False"], (0, h))
    grid.paste(cells["lora_True_trigger_True"], (w, h))
    grid_path = out_dir / "grid_2x2.png"
    grid.save(grid_path)

    print("=" * 60)
    print(f"Saved grid: {grid_path}")
    print("Layout: TL=no LoRA/no trigger  TR=no LoRA/trigger  BL=LoRA/no trigger  BR=LoRA/trigger")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
