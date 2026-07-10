"""
Regression tests for Phase 3 correctness & resource-safety fixes
(docs/perf_bestpractices_audit_2026-07-10.md quick-wins #4 and #5).

Covers two independent fixes:

1. `StreamDiffusion.prepare()`'s `generator` parameter previously defaulted to
   `torch.Generator()` -- a mutable default evaluated once at function-definition
   time, so every `StreamDiffusion` instance that didn't pass an explicit
   `generator` shared the SAME `torch.Generator` object. The fix changes the
   default to `None` and constructs a fresh `torch.Generator()` inside the
   method body when `generator is None`.

2. `StreamDiffusionWrapper.prepare()`'s two runtime `self.stream.prepare(...)`
   calls (single-prompt and prompt-blending paths) previously omitted `seed`,
   so `StreamDiffusion.prepare()`'s own default (`seed=2`) silently reset the
   RNG on every runtime prompt change. The fix forwards
   `seed=getattr(self.stream, "current_seed", 2)` so the active seed persists
   across prompt changes.

3. `StreamDiffusionWrapper.postprocess_image()`'s `"latent"` output branch
   previously returned the internal decode buffer by reference (aliased
   across frames). The fix clones at this public API boundary.

All tests are deliberately CPU-only and model-free (no real diffusers model
is loaded), following the object.__new__ shell pattern used elsewhere in
tests/unit/ (see test_wrapper_exception_hygiene.py, test_safety_checker.py).
"""

import inspect

import torch

from streamdiffusion.pipeline import StreamDiffusion
from streamdiffusion.wrapper import StreamDiffusionWrapper


# ---------------------------------------------------------------------------
# StreamDiffusion.prepare() -- mutable default removed
# ---------------------------------------------------------------------------


class TestPrepareGeneratorDefault:
    def test_generator_default_is_none_not_a_shared_instance(self):
        """
        The `generator` parameter's default must be `None`, not a
        pre-constructed `torch.Generator()` -- the latter is a classic mutable
        default footgun: the SAME Generator instance would be shared and
        mutated by every StreamDiffusion instance that omits `generator`.
        """
        default = inspect.signature(StreamDiffusion.prepare).parameters["generator"].default
        assert default is None, (
            f"expected generator default to be None (constructed fresh in the method body), "
            f"got a pre-built {type(default).__name__} instance -- this is a shared mutable default"
        )


# ---------------------------------------------------------------------------
# StreamDiffusionWrapper.prepare() -- seed threaded across prompt changes
# ---------------------------------------------------------------------------


class _FakeStream:
    """Stand-in for StreamDiffusion that only records prepare() kwargs."""

    def __init__(self, current_seed):
        self.current_seed = current_seed
        self.captured_kwargs = None

    def prepare(self, *args, **kwargs):
        self.captured_kwargs = kwargs


def _make_wrapper_shell_for_prepare(current_seed) -> StreamDiffusionWrapper:
    """Construct a minimal StreamDiffusionWrapper for exercising prepare()."""
    w = object.__new__(StreamDiffusionWrapper)
    w.stream = _FakeStream(current_seed)
    w._reload_text_encoders = lambda: None
    w._offload_text_encoders = lambda: None
    return w


class TestPrepareSeedThreading:
    def test_single_prompt_path_forwards_current_seed(self):
        """
        Single-prompt runtime prepare() must forward the stream's active
        `current_seed`, not silently fall back to StreamDiffusion.prepare()'s
        own default (seed=2) and reset the RNG.
        """
        w = _make_wrapper_shell_for_prepare(current_seed=12345)
        w.prepare("a test prompt")
        assert w.stream.captured_kwargs is not None, "stream.prepare() was not called"
        assert w.stream.captured_kwargs.get("seed") == 12345

    def test_prompt_blending_path_forwards_current_seed(self):
        """Prompt-blending runtime prepare() must also forward current_seed."""
        w = _make_wrapper_shell_for_prepare(current_seed=54321)
        w.update_stream_params = lambda **kwargs: None  # blending step, not under test here
        w.prepare([("cat", 0.7), ("dog", 0.3)])
        assert w.stream.captured_kwargs is not None, "stream.prepare() was not called"
        assert w.stream.captured_kwargs.get("seed") == 54321

    def test_falls_back_to_default_seed_when_stream_has_no_current_seed(self):
        """If the stream has no `current_seed` attribute yet, fall back to 2."""
        w = object.__new__(StreamDiffusionWrapper)
        fake_stream = _FakeStream(current_seed=0)
        del fake_stream.current_seed  # simulate an attribute that was never set
        w.stream = fake_stream
        w._reload_text_encoders = lambda: None
        w._offload_text_encoders = lambda: None
        w.prepare("a test prompt")
        assert w.stream.captured_kwargs.get("seed") == 2


# ---------------------------------------------------------------------------
# StreamDiffusionWrapper.postprocess_image() -- "latent" clones at boundary
# ---------------------------------------------------------------------------


def _make_wrapper_shell_for_postprocess() -> StreamDiffusionWrapper:
    """Construct a minimal StreamDiffusionWrapper for exercising postprocess_image()."""
    w = object.__new__(StreamDiffusionWrapper)
    w.use_cuda_ipc_output = False
    w._cuda_ipc_shm_name = None
    return w


class TestPostprocessImageLatentClone:
    def test_latent_output_is_an_independent_tensor(self):
        """
        `postprocess_image(..., output_type="latent")` must return a tensor
        that does NOT share storage with the input -- the input may be an
        internal decode buffer reused across frames (see
        StreamDiffusion.__call__/txt2img). A caller retaining the returned
        tensor across frames must not see it mutate out from under them.
        """
        buf = torch.arange(48, dtype=torch.float32).reshape(1, 3, 4, 4)
        w = _make_wrapper_shell_for_postprocess()
        out = w.postprocess_image(buf, output_type="latent")
        assert out.data_ptr() != buf.data_ptr(), "returned tensor still aliases the input buffer"
        assert torch.equal(out, buf), "cloned tensor must have identical values"

        # Mutating the source buffer (simulating buffer reuse on the next frame)
        # must NOT be visible in the previously-returned tensor.
        buf.fill_(-1.0)
        assert not torch.equal(out, buf), "returned tensor was mutated by a later write to the source buffer"
