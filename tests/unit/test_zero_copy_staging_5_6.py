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


PTR_A = 0x1000_0000  # 256-byte aligned (audit M2 guard)
PTR_B = 0x2000_0000  # 256-byte aligned (audit M2 guard)
PTR_MISALIGNED = PTR_A + 1  # not 256-byte aligned


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
        """Default frozenset() → today's behavior exactly, for every name.

        prev_ptr=None is the realistic value here: a name that is never in
        zero_copy_names is never bound, so Engine._bound_ptrs never has an
        entry for it and _bound_ptrs.get(name) is always None.
        """
        result = _staging_action(
            "kvo_cache_in_0",
            frozenset(),
            True,
            True,
            None,
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


class TestStagingActionAlignmentGuard:
    """Audit M2: setTensorAddress requires >=256-byte alignment. A mis-aligned
    cur_ptr must fall back to "copy" even when every other eligibility check
    would otherwise allow a bind — never silently bind an unaligned address."""

    def test_misaligned_pointer_falls_back_to_copy_no_graph(self):
        result = _staging_action(
            "kvo_cache_in_0",
            frozenset({"kvo_cache_in_0"}),
            True,
            True,
            None,
            PTR_MISALIGNED,
            False,
        )
        assert result == "copy"

    def test_misaligned_pointer_falls_back_to_copy_with_live_graph(self):
        """A name previously bound zero-copy (prev_ptr set from an earlier,
        valid aligned bind) whose current address is invalid for
        setTensorAddress must copy — but a live graph is still replaying the
        stale bound address, so this must ALSO force a reset.

        This is the regression lock: f8ff50f's bind->copy fallback left the
        live UNet CUDA graph replaying a frozen buffer forever once a
        zero-copy input fell back to copy mid-stream (see unet_engine.py's
        header comment for the full incident writeup). Before the
        copy_and_reset outcome existed, this exact transition asserted plain
        "copy" here — i.e. the test documented the bug as correct behavior.
        """
        result = _staging_action(
            "kvo_cache_in_0",
            frozenset({"kvo_cache_in_0"}),
            True,
            True,
            PTR_A,  # previously bound at a valid, aligned address
            PTR_MISALIGNED,  # now invalid -> must fall back to copy
            True,
        )
        assert result == "copy_and_reset"


class TestStagingActionCopyAndReset:
    """copy_and_reset is the general-purpose safety net for ANY zero-copy name
    (currently kvo/fio only — see unet_engine.py) that falls back to copy while
    a graph built from its previously-bound address is still live. Exercises
    the non-contiguous fallback path specifically (the misaligned-pointer path
    is covered by TestStagingActionAlignmentGuard above)."""

    def test_non_contiguous_previously_bound_with_live_graph_copies_and_resets(self):
        """A previously-bound name (e.g. a cache tensor that got sliced/viewed
        this frame) that is no longer contiguous, with a live graph, must copy
        AND force a reset — the graph is still reading the old bound address."""
        result = _staging_action(
            "kvo_cache_in_0",
            frozenset({"kvo_cache_in_0"}),
            False,  # is_contiguous
            True,
            PTR_A,  # previously bound
            PTR_A,
            True,
        )
        assert result == "copy_and_reset"

    def test_non_contiguous_previously_bound_no_live_graph_just_copies(self):
        """Same fallback, but no graph exists yet — nothing to invalidate, so a
        plain copy is correct; forcing a reset here would be a needless no-op
        reset on every non-graphed frame."""
        result = _staging_action(
            "kvo_cache_in_0",
            frozenset({"kvo_cache_in_0"}),
            False,
            True,
            PTR_A,
            PTR_A,
            False,
        )
        assert result == "copy"

    def test_never_bound_name_falls_back_with_live_graph_just_copies(self):
        """prev_ptr=None means this name has never been bound zero-copy, so no
        graph could possibly be reading a stale address for it — plain copy,
        no spurious reset."""
        result = _staging_action(
            "kvo_cache_in_0",
            frozenset({"kvo_cache_in_0"}),
            False,
            True,
            None,
            PTR_A,
            True,
        )
        assert result == "copy"


class TestStagingActionControlNetResiduals:
    """Phase-2 D2 attempted to opt input_control_* UNet inputs into this same
    zero-copy path; that was reverted (see unet_engine.py header comment) after
    it reproduced a "ControlNet produces no visual change" regression on the
    rig, driven by a lost engine-stream copy ordering guarantee — not by the
    bind->copy fallback covered above. input_control_* names are PERMANENTLY
    excluded from the real UNet engine's _zero_copy_names. The tests below
    exercise _staging_action as a pure function using ControlNet-shaped names
    only as convenient example zero-copy names; they do not describe current
    runtime behavior for actual ControlNet residuals."""

    def test_steady_state_control_residual_binds_without_reset(self):
        """Same persistent merge/output buffer across frames — plain bind."""
        result = _staging_action(
            "input_control_00",
            frozenset({"input_control_00"}),
            True,
            True,
            PTR_A,
            PTR_A,
            True,
        )
        assert result == "bind"

    def test_idle_active_toggle_binds_and_resets(self):
        """Idle dummy-zero buffer -> active ControlNet residual buffer address
        flip while a graph is live must force a re-capture."""
        result = _staging_action(
            "input_control_00",
            frozenset({"input_control_00"}),
            True,
            True,
            PTR_A,
            PTR_B,
            True,
        )
        assert result == "bind_and_reset"

    def test_middle_control_residual_binds_without_reset(self):
        result = _staging_action(
            "input_control_middle",
            frozenset({"input_control_middle"}),
            True,
            True,
            PTR_A,
            PTR_A,
            True,
        )
        assert result == "bind"
