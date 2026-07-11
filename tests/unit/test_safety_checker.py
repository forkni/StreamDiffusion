"""
Regression tests for StreamDiffusionWrapper._apply_safety_checker.

These tests are deliberately CPU-only and model-free.  They construct a
minimal StreamDiffusionWrapper shell via object.__new__ and wire in only the
attributes the helper reads, so the full GPU/TRT stack is not required.

Root cause being guarded: when use_cuda_ipc_output=True and output_type='pt',
postprocess_image() exports the frame and returns None.  The old code fed that
None to self.safety_checker, which called torchvision T.Resize on None and
raised:
    TypeError: Unexpected type <class 'NoneType'>
This crashed the streaming loop every frame.

Fix: _apply_safety_checker() runs *before* postprocess_image, operating on the
raw diffusion-range [-1, 1] pipeline tensor so it is always a real tensor
regardless of output path.

Sub-phase 5.3: the checker's verdict is now read 1-frame-delayed from a pinned
async buffer (mirrors SimilarImageFilter) instead of a synchronous .item() call.
`safety_checker` is now `(tensor, prob_pin) -> None`: it writes into `prob_pin`
rather than returning a bool, and `_apply_safety_checker` reads the *previous*
call's pinned value before launching (and writing) this frame's result. Tests
below seed `w._nsfw_prob_pin` directly where they need to bypass the
first-frame pass-through and exercise a specific verdict.
"""

import torch

from streamdiffusion.wrapper import StreamDiffusionWrapper


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_wrapper(
    *,
    use_safety_checker: bool = True,
    fallback_type: str = "blank",
    threshold: float = 0.5,
):
    """Construct a minimal StreamDiffusionWrapper without model loading."""
    w = object.__new__(StreamDiffusionWrapper)
    w.use_safety_checker = use_safety_checker
    w.safety_checker_threshold = threshold
    w.safety_checker_fallback_type = fallback_type
    w._prev_clean_tensor = None
    w._nsfw_prob_pin = None
    # safety_checker is set per-test
    w.safety_checker = None
    return w


def _black_denorm(t: torch.Tensor) -> bool:
    """True when t is all-zeros after _denormalize_on_gpu (i.e. all -1.0 raw)."""
    return torch.allclose(t, torch.zeros_like(t))


# ---------------------------------------------------------------------------
# test cases
# ---------------------------------------------------------------------------


class TestApplySafetyChecker:
    # ── Case 1: No-None contract (directly reproduces the old crash) ────────
    def test_checker_never_receives_none(self):
        """
        The safety checker must never be called with a None tensor.
        This is the exact condition that produced:
            TypeError: Unexpected type <class 'NoneType'>
        in the old post-hoc code path when output_type='pt' and IPC was active.
        """
        received: list = []

        def capturing_checker(tensor, prob_pin):
            received.append(tensor)
            prob_pin.fill_(1.0)

        w = _make_wrapper()
        w.safety_checker = capturing_checker

        dummy = torch.randn(1, 3, 64, 64)
        w._apply_safety_checker(dummy)

        assert len(received) == 1
        arg = received[0]
        assert arg is not None, "safety checker received None — regression"
        assert isinstance(arg, torch.Tensor), f"expected Tensor, got {type(arg)}"

    # ── Case 2: NSFW → black frame (blank fallback) ──────────────────────────
    def test_nsfw_blank_fallback_is_black(self):
        w = _make_wrapper(fallback_type="blank")
        w.safety_checker = lambda t, pin: pin.fill_(1.0)  # always "flag"

        frame1 = torch.randn(1, 3, 64, 64)
        _ = w._apply_safety_checker(frame1)  # first frame: pass-through, primes the pin

        frame2 = torch.randn(1, 3, 64, 64)
        result = w._apply_safety_checker(frame2)  # reads frame1's flagged verdict

        assert result is not frame2, "flagged frame should be replaced"
        # Denormalize: (x/2+0.5).clamp(0,1); -1.0 → 0.0 = black
        denorm = (result / 2 + 0.5).clamp(0, 1)
        assert _black_denorm(denorm), "NSFW blank fallback should produce a black frame"

    # ── Case 3: NSFW → previous frame ────────────────────────────────────────
    def test_nsfw_previous_fallback_returns_cached_clean_frame(self):
        w = _make_wrapper(fallback_type="previous")
        call_count = [0]

        def checker(t, pin):
            call_count[0] += 1
            # Call #1 (launched during frame1) writes a flagged prob; frame2's
            # read consumes that async result (1-frame delay).
            pin.fill_(1.0 if call_count[0] == 1 else 0.0)

        w.safety_checker = checker

        clean = torch.randn(1, 3, 64, 64)
        _ = w._apply_safety_checker(clean)  # first frame: pass-through, primes _prev_clean_tensor

        flagged_frame = torch.randn(1, 3, 64, 64)
        result = w._apply_safety_checker(flagged_frame)

        assert result is w._prev_clean_tensor, "previous-fallback should return the cached clean tensor"
        assert torch.equal(w._prev_clean_tensor, clean)

    # ── Case 4: Clean frame passthrough ──────────────────────────────────────
    def test_clean_frame_returned_unchanged(self):
        w = _make_wrapper()
        w.safety_checker = lambda t, pin: pin.fill_(0.0)  # never flag

        frame1 = torch.randn(1, 3, 64, 64)
        _ = w._apply_safety_checker(frame1)  # first frame: pass-through, primes the pin

        frame2 = torch.randn(1, 3, 64, 64)
        result = w._apply_safety_checker(frame2)

        assert torch.equal(result, frame2), "clean frame should be returned unchanged"

    # ── Case 5: use_safety_checker=False bypasses entirely ───────────────────
    def test_disabled_bypasses_checker(self):
        called = [False]

        def should_not_be_called(t, pin):
            called[0] = True

        w = _make_wrapper(use_safety_checker=False)
        w.safety_checker = should_not_be_called

        dummy = torch.randn(1, 3, 64, 64)
        result = w._apply_safety_checker(dummy)

        assert not called[0], "safety checker must not be called when disabled"
        assert torch.equal(result, dummy), "disabled checker should return input unchanged"

    # ── Case 6: flagged verdict with no cached frame → black ─────────────────
    def test_nsfw_previous_fallback_no_cache_falls_back_to_black(self):
        """
        If a frame is flagged (per the previous frame's pinned verdict) and
        _prev_clean_tensor is None, the fallback should still produce a black
        frame rather than raise. Seeds _nsfw_prob_pin directly to bypass the
        first-frame pass-through and exercise this state combination.
        """
        w = _make_wrapper(fallback_type="previous")
        w.safety_checker = lambda t, pin: pin.fill_(0.0)
        w._nsfw_prob_pin = torch.tensor([1.0])  # a prior flagged verdict already landed
        assert w._prev_clean_tensor is None  # no cache yet

        dummy = torch.randn(1, 3, 64, 64)
        result = w._apply_safety_checker(dummy)

        denorm = (result / 2 + 0.5).clamp(0, 1)
        assert _black_denorm(denorm), "when prev cache is empty and content is flagged, fallback must be black"

    # ── Case 7: clean frame caches for previous strategy ─────────────────────
    def test_clean_frame_cached_for_previous_strategy(self):
        w = _make_wrapper(fallback_type="previous")
        w.safety_checker = lambda t, pin: pin.fill_(0.0)
        w._nsfw_prob_pin = torch.tensor([0.0])  # a prior clean verdict already landed

        assert w._prev_clean_tensor is None
        dummy = torch.randn(1, 3, 64, 64)
        w._apply_safety_checker(dummy)

        assert w._prev_clean_tensor is not None, (
            "clean frame with previous strategy should populate _prev_clean_tensor"
        )
        assert torch.equal(w._prev_clean_tensor, dummy)

    # ── Case 8: clean frame NOT cached for blank strategy ────────────────────
    def test_clean_frame_not_cached_for_blank_strategy(self):
        w = _make_wrapper(fallback_type="blank")
        w.safety_checker = lambda t, pin: pin.fill_(0.0)
        w._nsfw_prob_pin = torch.tensor([0.0])  # a prior clean verdict already landed

        dummy = torch.randn(1, 3, 64, 64)
        w._apply_safety_checker(dummy)

        assert w._prev_clean_tensor is None, "_prev_clean_tensor should stay None when fallback_type='blank'"

    # ── Case 9: first frame always passes through ────────────────────────────
    def test_first_frame_always_passes_through_regardless_of_verdict(self):
        """
        The very first frame has no prior async result to read, so it must
        pass through unscreened even if the classifier would flag it — the
        documented 1-frame pass-through edge case of the delayed-readback design.
        """
        w = _make_wrapper(fallback_type="blank")
        w.safety_checker = lambda t, pin: pin.fill_(1.0)  # would flag if read this frame

        dummy = torch.randn(1, 3, 64, 64)
        result = w._apply_safety_checker(dummy)

        assert torch.equal(result, dummy), "first frame must pass through (no prior verdict to act on)"

    # ── Case 10: _process_skip_diffusion wiring — checker is called, result is black ──
    def test_skip_diffusion_routes_through_safety_checker(self):
        """
        Verify that _process_skip_diffusion actually feeds its tensor through
        _apply_safety_checker before postprocess_image, and that a flagged frame
        produces the black substitution rather than passing through unscreened.

        This is a wiring test — the behaviour of _apply_safety_checker itself is
        covered by the earlier cases.  We stub all model-dependent calls with
        identity/no-op lambdas so the test is CPU-only and model-free.
        """
        received: list = []

        def capturing_checker(tensor, pin):
            received.append(tensor)
            pin.fill_(0.0)  # this frame's own async result — irrelevant to this frame's own read

        w = _make_wrapper(fallback_type="blank")
        w.safety_checker = capturing_checker
        w._nsfw_prob_pin = torch.tensor([1.0])  # a prior frame's flagged verdict already landed
        w.mode = "img2img"
        w.device = torch.device("cpu")
        w.dtype = torch.float32

        # Stub the stream's pre/post hooks to identity
        class _FakeStream:
            def _apply_image_preprocessing_hooks(self, t):
                return t

            def _apply_image_postprocessing_hooks(self, t):
                return t

        w.stream = _FakeStream()

        # _normalize_on_gpu / _denormalize_on_gpu are identity for this test
        w._normalize_on_gpu = lambda t: t
        w._denormalize_on_gpu = lambda t: t

        # postprocess_image returns its input so we can inspect the tensor
        w.postprocess_image = lambda t, output_type=None: t
        w.output_type = "pt"

        dummy_image = torch.randn(1, 3, 64, 64)
        result = w._process_skip_diffusion(dummy_image)

        # Checker must have been called exactly once with a real tensor
        assert len(received) == 1, "safety checker should be called exactly once"
        assert isinstance(received[0], torch.Tensor), "checker arg must be a Tensor, not None"

        # Flagged frame (per the pre-seeded pinned verdict) must produce the black substitution
        denorm = (result / 2 + 0.5).clamp(0, 1)
        assert torch.allclose(denorm, torch.zeros_like(denorm)), (
            "flagged skip-diffusion frame should produce a black output"
        )
