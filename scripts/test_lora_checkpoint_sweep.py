"""
SDXL base 1.0 LoRA checkpoint sweep — plain diffusers, no StreamDiffusion wrapper.

Same prompt/seed/weight across every intermediate checkpoint OneTrainer saved during
training, to find the step count where the style takes vs. where it collapses into
overfit/memorized noise. Companion to test_lora_grid_sdxl_base.py, which showed the
final (step ~1900) checkpoint at weight 1.0 producing near-identical output regardless
of trigger word — i.e. the LoRA overriding the prompt rather than gating on the trigger.

Usage:
    venv/Scripts/python scripts/test_lora_checkpoint_sweep.py
"""

import argparse
import os
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw

os.environ.setdefault("HF_HOME", r"D:\Users\alexk\.cache\huggingface")

from diffusers import AutoencoderTiny, EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline

DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_CKPT_DIR = r"D:\dev\RESEARCH\OneTrainer\workspace\run\save"
DEFAULT_FINAL = r"D:\dev\RESEARCH\OneTrainer\models\lora.safetensors"
DEFAULT_TRIGGER = "sepiagraph"
DEFAULT_PROMPT = "a tall art deco tower, architectural drawing"
DEFAULT_NEG_PROMPT = "blurry, low quality"

STEP_RE = re.compile(r"-save-(\d+)-(\d+)-\d+\.safetensors$")


def discover_checkpoints(ckpt_dir: Path, final_path: Path) -> list[tuple[int, Path]]:
    found = []
    for p in sorted(ckpt_dir.glob("*.safetensors")):
        m = STEP_RE.search(p.name)
        if m:
            found.append((int(m.group(1)), p))
    found.sort(key=lambda t: t[0])
    if final_path.exists():
        found.append((-1, final_path))  # -1 = sort-last sentinel, label overridden below
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="LoRA checkpoint sweep across training steps")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    parser.add_argument("--final", default=DEFAULT_FINAL)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--trigger", default=DEFAULT_TRIGGER)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEG_PROMPT)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "lora_checkpoint_sweep"),
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = discover_checkpoints(Path(args.ckpt_dir), Path(args.final))
    if not checkpoints:
        print("No checkpoints found.")
        return 1

    print(f"Loading {args.model} ...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=True,
    )
    pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16, local_files_only=True)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    prompt = f"{args.trigger}, {args.prompt}"
    cells = []
    for train_step, ckpt_path in checkpoints:
        label = "final(~1900)" if train_step == -1 else f"step{train_step}"
        print(f"--- {label} --- {ckpt_path.name}")
        pipe.load_lora_weights(str(ckpt_path))
        gen = torch.Generator(device="cuda").manual_seed(args.seed)
        img = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=gen,
            cross_attention_kwargs={"scale": args.weight},
        ).images[0]
        pipe.unload_lora_weights()

        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 160, 24], fill=(0, 0, 0))
        draw.text((4, 4), label, fill=(255, 255, 255))

        path = out_dir / f"{label}.png"
        img.save(path)
        cells.append(img)
        print(f"Saved: {path}")

    cols = 5
    rows = (len(cells) + cols - 1) // cols
    w, h = cells[0].size
    contact = Image.new("RGB", (w * cols, h * rows), (32, 32, 32))
    for i, img in enumerate(cells):
        r, c = divmod(i, cols)
        contact.paste(img, (c * w, r * h))
    contact_path = out_dir / "contact_sheet.png"
    contact.save(contact_path)

    print("=" * 60)
    print(f"Saved contact sheet: {contact_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
