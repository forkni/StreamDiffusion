"""
Workstream A verification step 1 -- ONNX graph check for the dynamic-LoRA export path.

Cheap by design: monkeypatches builder.optimize_onnx to raise a sentinel exception
the instant it's called, so the real StreamDiffusionWrapper construction path pays
for genuine ONNX export (~94s) but skips the ~289s optimize and ~54s TRT-build
stages. Runs the export twice -- once with no LoRA, once with one LoRA -- so
"lora_scale is in the graph inputs and the input count grew by exactly 1" is a
true A/B against a real no-LoRA export, not a self-referential check against
models.py's declared input list.

Usage:
    venv/Scripts/python scripts/test_lora_graph_check.py
"""

import gc
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import onnx
import torch

import streamdiffusion.acceleration.tensorrt.builder as builder_mod
from streamdiffusion import StreamDiffusionWrapper

MODEL = "stabilityai/sdxl-turbo"
LORA_PATH = r"D:\dev\RESEARCH\OneTrainer\models\lora_sepiagraph_300.safetensors"
OUT_DIR = _REPO_ROOT / "outputs" / "lora_graph_check"


class _StopAfterExport(Exception):
    def __init__(self, onnx_path: str):
        self.onnx_path = onnx_path


_real_optimize_onnx = builder_mod.optimize_onnx


def _sentinel_optimize_onnx(onnx_path, onnx_opt_path, model_data):
    # Compilation order is VAE decoder -> VAE encoder -> UNet, so only abort on
    # the UNet stage; let the (fast) VAE engines optimize+build for real, since
    # aborting them would just raise the sentinel before the UNet is ever reached.
    if Path(onnx_path).name != "unet.engine.onnx":
        return _real_optimize_onnx(onnx_path=onnx_path, onnx_opt_path=onnx_opt_path, model_data=model_data)
    raise _StopAfterExport(onnx_path)


def _find_stop_after_export(exc: BaseException):
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, _StopAfterExport):
            return exc
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return None


def export_onnx_only(lora_dict, engine_dir: Path, label: str) -> str:
    builder_mod.optimize_onnx = _sentinel_optimize_onnx
    print(f"--- [{label}] building wrapper (acceleration=tensorrt, lora_dict={lora_dict}) ---")
    try:
        StreamDiffusionWrapper(
            model_id_or_path=MODEL,
            t_index_list=[10, 35],
            frame_buffer_size=1,
            width=512,
            height=512,
            warmup=1,
            acceleration="tensorrt",
            mode="img2img",
            use_denoising_batch=True,
            cfg_type="self",
            seed=42,
            use_tiny_vae=True,
            lora_dict=lora_dict,
            engine_dir=str(engine_dir),
            compile_engines_only=True,
        )
    except Exception as e:
        stop = _find_stop_after_export(e)
        if stop is not None:
            print(f"--- [{label}] sentinel fired -- onnx_path = {stop.onnx_path} ---")
            return stop.onnx_path
        raise
    finally:
        builder_mod.optimize_onnx = _real_optimize_onnx
    raise RuntimeError(f"[{label}] optimize_onnx sentinel never fired -- export did not run")


def graph_inputs(onnx_path: str):
    model = onnx.load(onnx_path, load_external_data=False)
    return list(model.graph.input)


def main() -> int:
    baseline_path = export_onnx_only(None, OUT_DIR / "engines_baseline", "baseline")
    gc.collect()
    torch.cuda.empty_cache()

    lora_path = export_onnx_only({LORA_PATH: 1.0}, OUT_DIR / "engines_lora", "lora")

    baseline_inputs = graph_inputs(baseline_path)
    lora_inputs = graph_inputs(lora_path)
    baseline_names = [i.name for i in baseline_inputs]
    lora_names = [i.name for i in lora_inputs]

    print()
    print(f"baseline input count: {len(baseline_names)}")
    print(f"lora input count:     {len(lora_names)}")
    print(f"baseline inputs: {baseline_names}")
    print(f"lora inputs:     {lora_names}")

    ok = True
    if "lora_scale" in baseline_names:
        print("FAIL: lora_scale present in baseline (no-LoRA) graph inputs")
        ok = False
    if "lora_scale" not in lora_names:
        print("FAIL: lora_scale missing from lora graph inputs")
        ok = False
    delta = len(lora_names) - len(baseline_names)
    if delta != 1:
        print(f"FAIL: input count grew by {delta}, expected exactly 1")
        ok = False

    if "lora_scale" in lora_names:
        lora_scale_input = next(i for i in lora_inputs if i.name == "lora_scale")
        dims = [d.dim_value if d.dim_value else d.dim_param for d in lora_scale_input.type.tensor_type.shape.dim]
        print(f"lora_scale shape: {dims}")

    print()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
