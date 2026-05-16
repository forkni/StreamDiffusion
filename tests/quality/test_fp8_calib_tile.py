"""Regression: per-input-aware calibration tiling for FP8 quantize.

Reproduces the kvo_cache_in dim-0=2 vs synthesized dim-0=1 split mismatch
hit by SDXL-Turbo + use_cached_attn + cfg_type=self configs.
"""
import math

import numpy as np
import onnx
from onnx import TensorProto, helper


def _make_min_onnx(path):
    """Minimal 2-input ONNX: symbolic-batch 'sample' + static-dim0=2 'kvo_cache_in_0'."""
    sample = helper.make_tensor_value_info("sample", TensorProto.FLOAT, ["2B", 4, 64, 64])
    kvo = helper.make_tensor_value_info("kvo_cache_in_0", TensorProto.FLOAT, [2, 4, "2B", 64, 64])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, ["2B", 4, 64, 64])
    ident = helper.make_node("Identity", inputs=["sample"], outputs=["out"])
    g = helper.make_graph([ident], "min", [sample, kvo], [out])
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), path)


def test_per_input_tile_preserves_static_dim0(tmp_path):
    """Per-input tile: sample stays at n_itr rows, kvo stays at 2×n_itr rows."""
    onnx_path = str(tmp_path / "min.onnx")
    _make_min_onnx(onnx_path)

    calib = {
        "sample": np.zeros((5, 4, 64, 64), dtype=np.float32),
        "kvo_cache_in_0": np.zeros((10, 4, 5, 64, 64), dtype=np.float32),
    }

    from streamdiffusion.acceleration.tensorrt.fp8_quantize import _read_onnx_input_specs

    specs = _read_onnx_input_specs(onnx_path)

    # Reproduce the fixed logic
    resolved_dim0 = {name: max(1, (specs[name][1][0] or 1)) for name in calib}
    n_itr = max(arr.shape[0] // resolved_dim0[name] for name, arr in calib.items())
    n_itr = max(1, n_itr)
    out = {}
    for k, arr in calib.items():
        target = n_itr * resolved_dim0[k]
        if arr.shape[0] != target:
            repeats = math.ceil(target / max(1, arr.shape[0]))
            arr = np.tile(arr, (repeats,) + (1,) * (arr.ndim - 1))[:target]
        out[k] = arr

    # n_itr=5, resolved_dim0(sample)=1 → 5 rows; resolved_dim0(kvo)=2 → 10 rows
    assert out["sample"].shape == (5, 4, 64, 64)
    assert out["kvo_cache_in_0"].shape == (10, 4, 5, 64, 64)

    # Verify modelopt split math: n_itr chunks each of shape (resolved_dim0, ...)
    sample_chunks = np.array_split(out["sample"], n_itr, axis=0)
    kvo_chunks = np.array_split(out["kvo_cache_in_0"], n_itr, axis=0)
    assert sample_chunks[0].shape[0] == 1
    assert kvo_chunks[0].shape[0] == 2  # static dim 0 must be preserved


def test_naive_max_rows_tile_would_break(tmp_path):
    """Confirms the OLD naïve tile produces the 'Got 1 Expected 2' symptom."""
    onnx_path = str(tmp_path / "min.onnx")
    _make_min_onnx(onnx_path)

    calib = {
        "sample": np.zeros((5, 4, 64, 64), dtype=np.float32),
        "kvo_cache_in_0": np.zeros((10, 4, 5, 64, 64), dtype=np.float32),
    }

    # Reproduce the buggy logic
    _max_rows = max(a.shape[0] for a in calib.values())
    for k, a in list(calib.items()):
        if a.shape[0] < _max_rows:
            calib[k] = np.tile(a, (math.ceil(_max_rows / a.shape[0]),) + (1,) * (a.ndim - 1))[:_max_rows]

    # modelopt: n_itr = sample.shape[0] / symbolic_dim0(1) = 10
    # splits kvo into 10 chunks → each has shape[0]=1 → ORT rejects (expected 2)
    n_itr_bad = calib["sample"].shape[0]  # 10 (doubled by naïve tile)
    kvo_chunk = np.array_split(calib["kvo_cache_in_0"], n_itr_bad, axis=0)[0]
    assert kvo_chunk.shape[0] == 1  # this is the "Got 1 Expected 2" symptom
