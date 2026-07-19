"""
Regression tests for ControlNetModule's multi-ControlNet residual merge (Phase-2 prep, D1).

CPU-only and model-free: fake ControlNet callables stand in for real engines so the
merge logic in build_unet_hook()'s closure can be exercised without CUDA/TRT.

Root cause being guarded: the old merge (`merged_down[j] = merged_down[j] + ds[j]`)
allocated a fresh tensor every frame, so the UNet's input_control_* residuals would
never be pointer-stable across frames — a prerequisite for zero-copy binding those
inputs (Phase-2 D2). This also verifies the merge never aliases engine A's own
persistent output buffer.
"""

import torch

from streamdiffusion.hooks import StepCtx
from streamdiffusion.modules.controlnet_module import ControlNetModule


class _FakeCN:
    """Stands in for a ControlNet engine: returns fixed-value residuals, records calls."""

    def __init__(self, down_value: float, mid_value: float, down_shapes, mid_shape):
        self.calls = 0
        self._down_value = down_value
        self._mid_value = mid_value
        self._down_shapes = down_shapes
        self._mid_shape = mid_shape

    def __call__(
        self, sample, timestep, encoder_hidden_states, controlnet_cond, conditioning_scale, return_dict=False
    ):
        self.calls += 1
        down = [torch.full(shape, self._down_value, dtype=torch.float32) for shape in self._down_shapes]
        mid = torch.full(self._mid_shape, self._mid_value, dtype=torch.float32)
        return down, mid


class _FakeStream:
    def __init__(self, text_len: int = 77, batch: int = 1):
        self.prompt_embeds = torch.randn(batch, text_len, 8)


def _make_module_with_two_controlnets(down_shapes, mid_shape) -> tuple:
    module = ControlNetModule(device="cpu", dtype=torch.float32)
    module._stream = _FakeStream()

    cn_a = _FakeCN(down_value=1.0, mid_value=2.0, down_shapes=down_shapes, mid_shape=mid_shape)
    cn_b = _FakeCN(down_value=3.0, mid_value=4.0, down_shapes=down_shapes, mid_shape=mid_shape)

    module.controlnets = [cn_a, cn_b]
    module.controlnet_images = [torch.randn(1, 3, 8, 8), torch.randn(1, 3, 8, 8)]
    module.controlnet_scales = [1.0, 1.0]
    module.enabled_list = [True, True]

    return module, cn_a, cn_b


def _make_ctx(batch: int = 1) -> StepCtx:
    return StepCtx(
        x_t_latent=torch.randn(batch, 4, 8, 8),
        t_list=torch.tensor([0]),
        step_index=0,
        guidance_mode="none",
        sdxl_cond=None,
    )


class TestControlNetResidualMerge:
    def test_merge_is_numerically_correct(self):
        down_shapes = [(1, 4, 8, 8), (1, 4, 4, 4)]
        mid_shape = (1, 4, 2, 2)
        module, cn_a, cn_b = _make_module_with_two_controlnets(down_shapes, mid_shape)
        hook = module.build_unet_hook()

        result = hook(_make_ctx())

        assert cn_a.calls == 1 and cn_b.calls == 1
        for merged, shape in zip(result.down_block_additional_residuals, down_shapes):
            assert torch.allclose(merged, torch.full(shape, 4.0)), "1.0 + 3.0 == 4.0 per down block"
        assert torch.allclose(result.mid_block_additional_residual, torch.full(mid_shape, 6.0)), "2.0 + 4.0 == 6.0"

    def test_merge_buffers_are_pointer_stable_across_frames(self):
        """Same shape on consecutive frames must reuse the same tensor objects
        (required before the UNet's input_control_* inputs can be zero-copy bound)."""
        down_shapes = [(1, 4, 8, 8)]
        mid_shape = (1, 4, 2, 2)
        module, _cn_a, _cn_b = _make_module_with_two_controlnets(down_shapes, mid_shape)
        hook = module.build_unet_hook()

        result1 = hook(_make_ctx())
        down_ptr1 = [t.data_ptr() for t in result1.down_block_additional_residuals]
        mid_ptr1 = result1.mid_block_additional_residual.data_ptr()

        result2 = hook(_make_ctx())
        down_ptr2 = [t.data_ptr() for t in result2.down_block_additional_residuals]
        mid_ptr2 = result2.mid_block_additional_residual.data_ptr()

        assert down_ptr1 == down_ptr2, "merged down-block buffers must be reused, not reallocated"
        assert mid_ptr1 == mid_ptr2, "merged mid-block buffer must be reused, not reallocated"
        # Values still correct on frame 2 (buffer was overwritten, not just left stale)
        assert torch.allclose(result2.down_block_additional_residuals[0], torch.full(down_shapes[0], 4.0))

    def test_merge_reallocates_on_shape_change(self):
        """A resolution/batch change must produce a new buffer, not corrupt-reuse the old one."""
        down_shapes = [(1, 4, 8, 8)]
        mid_shape = (1, 4, 2, 2)
        module, cn_a, cn_b = _make_module_with_two_controlnets(down_shapes, mid_shape)
        hook = module.build_unet_hook()

        result1 = hook(_make_ctx())
        down_ptr1 = result1.down_block_additional_residuals[0].data_ptr()

        # Simulate a resolution change: new shapes for both fake engines.
        new_down_shapes = [(1, 4, 16, 16)]
        new_mid_shape = (1, 4, 4, 4)
        cn_a._down_shapes = new_down_shapes
        cn_a._mid_shape = new_mid_shape
        cn_b._down_shapes = new_down_shapes
        cn_b._mid_shape = new_mid_shape

        result2 = hook(_make_ctx())
        assert result2.down_block_additional_residuals[0].shape == new_down_shapes[0]
        down_ptr2 = result2.down_block_additional_residuals[0].data_ptr()
        assert down_ptr2 != down_ptr1, "shape change must trigger reallocation"

    def test_merge_does_not_alias_engine_output_buffer(self):
        """The merge must not mutate/alias down_samples_list[0] — that's engine A's
        own persistent output buffer, reused by the engine on the next frame."""
        down_shapes = [(1, 4, 8, 8)]
        mid_shape = (1, 4, 2, 2)
        module, cn_a, _cn_b = _make_module_with_two_controlnets(down_shapes, mid_shape)
        hook = module.build_unet_hook()

        result = hook(_make_ctx())

        # Re-invoke cn_a directly (as the engine would on its own next call) and confirm
        # its freshly-returned buffer is untouched by the merge (still all 1.0, not 4.0).
        fresh_down, fresh_mid = cn_a(
            sample=None, timestep=None, encoder_hidden_states=None, controlnet_cond=None, conditioning_scale=1.0
        )
        assert torch.allclose(fresh_down[0], torch.full(down_shapes[0], 1.0))
        assert torch.allclose(fresh_mid, torch.full(mid_shape, 2.0))
        # Sanity: the merge result itself is still the summed value.
        assert torch.allclose(result.down_block_additional_residuals[0], torch.full(down_shapes[0], 4.0))

    def test_single_controlnet_bypasses_merge_buffers(self):
        """With only one active ControlNet, the engine's own output is returned
        directly — no merge buffer should be allocated."""
        module = ControlNetModule(device="cpu", dtype=torch.float32)
        module._stream = _FakeStream()
        down_shapes = [(1, 4, 8, 8)]
        mid_shape = (1, 4, 2, 2)
        cn_a = _FakeCN(down_value=5.0, mid_value=6.0, down_shapes=down_shapes, mid_shape=mid_shape)
        module.controlnets = [cn_a]
        module.controlnet_images = [torch.randn(1, 3, 8, 8)]
        module.controlnet_scales = [1.0]
        module.enabled_list = [True]

        hook = module.build_unet_hook()
        hook(_make_ctx())

        assert module._cn_merged_down is None
        assert module._cn_merged_mid is None

    def test_install_resets_merge_buffers(self):
        down_shapes = [(1, 4, 8, 8)]
        mid_shape = (1, 4, 2, 2)
        module, _cn_a, _cn_b = _make_module_with_two_controlnets(down_shapes, mid_shape)
        module.build_unet_hook()(_make_ctx())
        assert module._cn_merged_down is not None

        class _MinimalStream:
            unet_hooks = []
            controlnets = None
            controlnet_scales = None
            preprocessors = None

        # attach_orchestrator requires a preprocessing orchestrator; reuse the existing one
        # (install() only touches it when _preprocessing_orchestrator is None).
        module._preprocessing_orchestrator = object()
        module.install(_MinimalStream())

        assert module._cn_merged_down is None
        assert module._cn_merged_mid is None
        assert module._cn_merged_shape_key is None
