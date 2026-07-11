"""
Unit tests for Sub-phase 5.6 (`trt.input_staging` DtoD copy elision,
docs/perf_bestpractices_audit_2026-07-10.md follow-up — see the Phase 5
continuation plan).

`Engine.infer()` copies every feed_dict input into its own persistent staging
buffer every frame, purely to give TensorRT a stable contiguous address to
bind. The kvo/fio UNet cache inputs are already persistent, address-stable,
TRT-contiguous tensors (see models/utils.py create_kvo_cache/create_fi_cache),
so that copy is pure waste for them — 5.6 lets `Engine.infer()` bind directly
to an opt-in set of caller tensors instead.

`_staging_action()` is the pure decision function extracted from `Engine.infer()`
so the per-input zero-copy/copy/reset decision table is testable without a real
CUDA graph (a `cuda_graph_instance` can't be meaningfully MagicMock'd — the same
constraint Sub-phase 5.1's rebind guard hit). This module tests only that pure
function: no CUDA device, no TensorRT context required beyond importing the
module (guarded below, same pattern as test_l2tc_dynamic_shapes.py).
"""

import pytest


try:
    from streamdiffusion.acceleration.tensorrt.utilities import _staging_action

    IMPORT_OK = True
except ImportError:
    IMPORT_OK = False

pytestmark = pytest.mark.skipif(
    not IMPORT_OK,
    reason="acceleration.tensorrt.utilities not importable (TensorRT/onnx/polygraphy missing)",
)


PTR_A = 0xDEAD_BEEF
PTR_B = 0xFEED_FACE


class TestStagingActionNotZeroCopy:
    """Anything not opted in always falls back to the original copy path."""

    def test_name_not_in_zero_copy_names_copies(self):
        result = _staging_action(
            "sample",
            frozenset({"kvo_cache_in_0"}),
            True,
            True,
            None,
            PTR_A,
            False,
        )
        assert result == "copy"

    def test_empty_zero_copy_names_always_copies(self):
        """Default frozenset() → today's behavior exactly, for every name."""
        result = _staging_action(
            "kvo_cache_in_0",
            frozenset(),
            True,
            True,
            PTR_A,
            PTR_A,
            True,
        )
        assert result == "copy"


class TestStagingActionSafetyGuards:
    """A zero-copy candidate that fails a safety guard falls back to copy,
    never silently binding a non-contiguous tensor or mismatched dtype."""

    def test_non_contiguous_falls_back_to_copy(self):
        result = _staging_action(
            "kvo_cache_in_0",
            frozenset({"kvo_cache_in_0"}),
            False,  # is_contiguous
            True,
            None,
            PTR_A,
            False,
        )
        assert result == "copy"

    def test_dtype_mismatch_falls_back_to_copy(self):
        result = _staging_action(
            "kvo_cache_in_0",
            frozenset({"kvo_cache_in_0"}),
            True,
            False,  # dtype_match
            None,
            PTR_A,
            False,
        )
        assert result == "copy"

    def test_both_guards_fail_falls_back_to_copy(self):
        result = _staging_action(
            "kvo_cache_in_0",
            frozenset({"kvo_cache_in_0"}),
            False,
            False,
            None,
            PTR_A,
            True,
        )
        assert result == "copy"


class TestStagingActionBind:
    """Eligible names bind directly; a reset is forced only when a live graph's
    baked address would otherwise go stale."""

    def test_no_graph_yet_binds_without_reset(self):
        """First-ever call: no graph exists, so a fresh bind is always safe."""
        result = _staging_action(
            "kvo_cache_in_0",
            frozenset({"kvo_cache_in_0"}),
            True,
            True,
            None,
            PTR_A,
            False,
        )
        assert result == "bind"

    def test_graph_exists_pointer_unchanged_binds_without_reset(self):
        """Steady state: same persistent tensor, same address — no reset needed."""
        result = _staging_action(
            "kvo_cache_in_0",
            frozenset({"kvo_cache_in_0"}),
            True,
            True,
            PTR_A,
            PTR_A,
            True,
        )
        assert result == "bind"

    def test_graph_exists_pointer_changed_binds_and_resets(self):
        """Belt-and-suspenders: an unforeseen pointer change while a graph is
        live must force a re-capture, not silently rebind a stale graph."""
        result = _staging_action(
            "kvo_cache_in_0",
            frozenset({"kvo_cache_in_0"}),
            True,
            True,
            PTR_A,
            PTR_B,
            True,
        )
        assert result == "bind_and_reset"

    def test_no_graph_pointer_changed_still_just_binds(self):
        """No live graph to invalidate → no reset needed even if the pointer moved
        (e.g. right after allocate_buffers already reset the graph for a shape change)."""
        result = _staging_action(
            "kvo_cache_in_0",
            frozenset({"kvo_cache_in_0"}),
            True,
            True,
            PTR_A,
            PTR_B,
            False,
        )
        assert result == "bind"

    def test_first_bind_with_no_prior_ptr_recorded(self):
        """prev_ptr=None (nothing bound yet) with no live graph must not crash and
        must bind cleanly — exercises the None-vs-int comparison short-circuit."""
        result = _staging_action(
            "fio_cache_in_0",
            frozenset({"fio_cache_in_0"}),
            True,
            True,
            None,
            PTR_A,
            False,
        )
        assert result == "bind"
