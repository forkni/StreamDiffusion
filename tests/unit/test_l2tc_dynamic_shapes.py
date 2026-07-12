"""
Regression test for the l2tc `dynamic_shapes` fix (Phase 5, Commit B —
docs/perf_bestpractices_audit_2026-07-10.md follow-up).

Root cause: `build_engine()` passed `dynamic_shapes=build_dynamic_shape` into
`Engine.build()`, which tracks only dynamic *resolution*. Batch-dynamic /
resolution-static engines (e.g. the default "Flexible" UNet preset,
`build_static_batch=False, build_dynamic_shape=False`) got `dynamic_shapes=False`,
so `_apply_gpu_profile_to_config`'s tiling branch ran against a graph that still
has a symbolic batch dim, and TRT emitted a benign but noisy
"[l2tc] VALIDATE FAIL - Graph contains symbolic shape" warning for every
applicable layer.

Fix: `dynamic_shapes=build_dynamic_shape or not build_static_batch` — True
whenever *any* dim (resolution or batch) is symbolic, matching
`_apply_gpu_profile_to_config`'s actual requirement (l2tc needs ALL dims
concrete). Fully-static engines (e.g. ControlNet: `build_static_batch=True,
build_dynamic_shape=False`) are unaffected.

These tests exercise only `build_engine`'s pure-Python wiring: GPU detection,
the memory query, and the real `Engine.build()` TRT call are all monkeypatched
out, so no CUDA device or TensorRT context is required.
"""

from unittest.mock import MagicMock

import pytest

try:
    from streamdiffusion.acceleration.tensorrt import utilities as trt_utilities

    IMPORT_OK = True
except ImportError:
    IMPORT_OK = False

pytestmark = pytest.mark.skipif(
    not IMPORT_OK,
    reason="acceleration.tensorrt.utilities not importable (TensorRT/onnx/polygraphy missing)",
)


class _FakeModelData:
    """Stand-in for a BaseModel subclass — build_engine only calls get_input_profile()."""

    def get_input_profile(self, *args, **kwargs):
        return {}


def _computed_dynamic_shapes(monkeypatch, *, build_static_batch, build_dynamic_shape):
    """Call build_engine() with GPU detection / memory query / the real TRT build
    all monkeypatched out, and return the `dynamic_shapes` kwarg it actually
    passed to Engine.build()."""
    monkeypatch.setattr(trt_utilities.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(trt_utilities, "detect_gpu_profile", lambda device: trt_utilities._fallback_profile())
    monkeypatch.setattr(trt_utilities.cudart, "cudaMemGetInfo", lambda: (0, 8 * 2**30, 16 * 2**30))

    captured_build = MagicMock(return_value=None)
    monkeypatch.setattr(trt_utilities.Engine, "build", captured_build)

    trt_utilities.build_engine(
        engine_path="fake.engine",
        onnx_opt_path="fake.onnx",
        model_data=_FakeModelData(),
        opt_image_height=512,
        opt_image_width=512,
        opt_batch_size=1,
        build_static_batch=build_static_batch,
        build_dynamic_shape=build_dynamic_shape,
    )

    return captured_build.call_args.kwargs["dynamic_shapes"]


class TestL2tcDynamicShapesFix:
    def test_batch_dynamic_resolution_static_is_dynamic(self, monkeypatch):
        """Default 'Flexible' UNet preset: static resolution, dynamic batch —
        this is the bug case; must compute True (previously computed False)."""
        result = _computed_dynamic_shapes(monkeypatch, build_static_batch=False, build_dynamic_shape=False)
        assert result is True

    def test_fully_static_engine_is_not_dynamic(self, monkeypatch):
        """ControlNet-style fully-static engine: unaffected, still computes False."""
        result = _computed_dynamic_shapes(monkeypatch, build_static_batch=True, build_dynamic_shape=False)
        assert result is False

    def test_resolution_dynamic_is_dynamic_regardless_of_batch(self, monkeypatch):
        """build_dynamic_shape=True alone was already sufficient before this fix."""
        result = _computed_dynamic_shapes(monkeypatch, build_static_batch=True, build_dynamic_shape=True)
        assert result is True
