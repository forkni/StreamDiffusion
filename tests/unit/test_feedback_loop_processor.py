"""
Headless tests for FeedbackLoopPreprocessor (custom_processors/sdtd_fx/feedback_loop.py).

Coverage was zero before this file — nothing under StreamDiffusion/tests referenced
feedback_loop.py or FeedbackLoopPreprocessor. These tests run entirely on CPU against
a stub pipeline_ref (no model, no TRT) and pin the behavior the endless-magnification
fix (Step 3/Step 4 of the feedback_loop plan) depends on:

  - the persistent canvas accumulator compounds zoom geometrically (zoom**n) — the
    property "endless magnification" requires;
  - the canvas advances every call regardless of whether prev_image_result actually
    changed between calls (geometric advance decoupled from diffusion latency);
  - output stays within [-1, 1];
  - border_mode is read live each call, not cached at construction (D1 regression —
    the pre-fix code cached self._padding_mode once in __init__ and never re-read it,
    so an OSC-driven setattr(proc, "border_mode", ...) mid-stream silently did nothing);
  - the Deforum-parity reorder (TestDeforumParityReorder): injected V2V output and
    the live input are composited BEFORE the warp so they ride the same frame's
    motion, and the ephemeral sharpen/noise detail pass never persists into the
    canvas.

Geometry tests pass explicit sharpen_amount=0.0, noise_amount=0.0,
resample_mode="bilinear" — the shipped defaults are nonzero/bicubic and would
perturb exact-radius assertions.

Run with: pytest tests/unit/test_feedback_loop_processor.py -v
"""

import itertools
import types

import pytest
import torch

from streamdiffusion.preprocessing.processors import get_preprocessor

RES = 64


def _make_processor(**params):
    pipeline_ref = types.SimpleNamespace(prev_image_result=None)
    params.setdefault("image_resolution", RES)
    params.setdefault("device", "cpu")
    params.setdefault("dtype", torch.float32)
    proc = get_preprocessor("feedback_loop", pipeline_ref=pipeline_ref, params=params)
    return proc, pipeline_ref


def _disk_image(radius_px: float, value: float = 1.0, size: int = RES) -> torch.Tensor:
    """[3, size, size] float32 tensor in [0, 1]: `value` inside a centered disk of
    `radius_px`, 0 outside."""
    ys, xs = torch.meshgrid(
        torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0,
        torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0,
        indexing="ij",
    )
    dist = torch.sqrt(xs**2 + ys**2)
    disk = (dist <= radius_px).float() * value
    return disk.unsqueeze(0).expand(3, size, size).contiguous()


def _measure_radius_px(tensor_neg1_1: torch.Tensor, threshold: float = 0.5, size: int = RES) -> float:
    """Radius (px) at which the center row, scanning outward from the geometric
    center, first drops below `threshold` (in [0, 1] terms)."""
    img01 = (tensor_neg1_1 + 1.0) / 2.0
    if img01.dim() == 4:
        img01 = img01[0]
    center = (size - 1) / 2.0
    start_col = round(center)
    row = img01[0, start_col, :]
    for col in range(start_col, size):
        if row[col].item() < threshold:
            return col - center
    return float(size - 1 - center)


class TestZoomCompounding:
    def test_edge_radius_tracks_zoom_power_n(self):
        """With feedback_strength=1.0 and no StreamV2V output available, the output
        each frame IS the canvas (blended == canvas exactly), and the canvas warps by
        the same theta every call with nothing re-seeding it — so after n calls the
        disk edge should sit at r0 * zoom**n px (endless magnification, compounding)."""
        zoom = 1.05
        r0 = 10.0
        proc, _ = _make_processor(
            feedback_strength=1.0,
            zoom=zoom,
            pan_x=0.0,
            pan_y=0.0,
            rotation=0.0,
            sharpen_amount=0.0,
            noise_amount=0.0,
            resample_mode="bilinear",
        )
        disk = _disk_image(r0)
        inp = (disk * 2.0 - 1.0).unsqueeze(0)  # image_pre hook expects [-1, 1]

        for n in range(1, 16):
            out = proc._process_tensor_core(inp)
            measured = _measure_radius_px(out)
            expected = r0 * (zoom**n)
            # bilinear resampling softens the edge a little more each compounding
            # step (empirically ~10% at worst); generous relative tolerance rather
            # than pinning exact pixels.
            tolerance = max(2.5, 0.2 * expected)
            assert abs(measured - expected) <= tolerance, (
                f"frame {n}: measured radius {measured:.1f}px, expected ~{expected:.1f}px "
                f"(zoom={zoom}**{n}) — magnification is not compounding as zoom**n"
            )

    def test_zoom_below_one_shrinks_instead_of_growing(self):
        """Sanity check on the other side of 1.0."""
        zoom = 0.95
        r0 = 20.0
        proc, _ = _make_processor(
            feedback_strength=1.0,
            zoom=zoom,
            sharpen_amount=0.0,
            noise_amount=0.0,
            resample_mode="bilinear",
        )
        disk = _disk_image(r0)
        inp = (disk * 2.0 - 1.0).unsqueeze(0)

        radii = [_measure_radius_px(proc._process_tensor_core(inp)) for _ in range(8)]

        for a, b in itertools.pairwise(radii):
            assert b <= a + 0.5, f"radius sequence {radii}: expected monotonic shrink at zoom={zoom}"


class TestDefaultInjectionGeometry:
    """Regression coverage for the shipped-but-inert bug: at the ORIGINAL shipped defaults
    (inject_radius=1.0, inject_softness=0.0) the radial mask evaluated to 1.0 everywhere,
    so feedback_loop.py:293's `mask * prev_n + (1 - mask) * canvas` collapsed to
    `canvas = prev_n` every frame, discarding the warp applied one line earlier and making
    the whole accumulator inert. TestZoomCompounding sidesteps this (prev_image_result=None
    skips the injection block entirely) and TestGeometricAdvanceDecoupledFromLatency
    sidesteps it too (passes an explicit non-default inject_radius=0.15) — neither exercises
    DEFAULT parameters with a live prev, which is exactly the TouchDesigner scenario that
    shipped broken. Defaults are now inject_radius=0.25 / inject_softness=0.15."""

    def test_default_mask_is_not_uniformly_one(self):
        proc, _ = _make_processor()
        mask = proc._get_radial_mask(RES, RES, torch.device("cpu"))
        assert mask.min().item() < 1.0, (
            f"default inject_radius/inject_softness produce a mask with min={mask.min().item()} "
            "— at min==1.0 the injection blend at feedback_loop.py:293 collapses to "
            "self._canvas = prev_n every frame, discarding the warp and disabling "
            "magnification entirely regardless of zoom"
        )

    def test_identity_transform_matches_plain_blend(self):
        """zoom=1.0 (no zoom/pan/rotation) must render as a plain blend even with the new
        non-trivial default mask — this is the Step 2 identity guard in _advance_canvas
        that keeps existing zoom=1.0 presets unaffected by the defaults change."""
        proc, pipeline_ref = _make_processor(feedback_strength=0.8, zoom=1.0)

        # _get_prev_output() returns None on the very first call regardless of
        # prev_image_result (self._first_frame gate) — warm up once so the second call
        # actually exercises the injection path.
        proc._process_tensor_core(torch.rand(1, 3, RES, RES) * 2.0 - 1.0)

        prev = torch.rand(1, 3, RES, RES) * 2.0 - 1.0
        pipeline_ref.prev_image_result = prev
        inp = torch.rand(1, 3, RES, RES) * 2.0 - 1.0

        out = proc._process_tensor_core(inp)

        inp01 = (inp[0] / 2.0 + 0.5).clamp(0, 1)
        prev01 = (prev[0] / 2.0 + 0.5).clamp(0, 1)
        expected01 = ((1 - proc.feedback_strength) * inp01 + proc.feedback_strength * prev01).clamp(0, 1)
        expected = expected01 * 2.0 - 1.0

        assert torch.allclose(out[0], expected, atol=1e-4), (
            "zoom=1.0 output diverges from a plain (1-fs)*inp + fs*prev blend — the "
            "identity guard in _advance_canvas is not bypassing the radial mask at zoom=1.0"
        )

    def test_default_parameters_compound_with_live_prev(self):
        """The scenario that shipped broken, reproduced directly: DEFAULT inject_radius/
        inject_softness (and default feedback_strength=1.0 pure loopback + bicubic) with
        a live (non-None) prev_image_result fed back frame-to-frame, like a real
        StreamV2V closed loop. A disk seeded outside the default injection core (~11px
        of 64) must still expand by zoom**n. Two params are pinned explicitly: zoom
        (default 1.01 is too slow to measure in 20 frames) and sharpen/noise set to 0 —
        the detail pass recirculates through the injection core via prev and, with no
        diffusion model in this harness to absorb it, random-walks the core dark."""
        zoom = 1.02
        r0 = 20.0  # px — outside the default injection core, safely in pure-warp territory
        frames = 20
        proc, pipeline_ref = _make_processor(zoom=zoom, sharpen_amount=0.0, noise_amount=0.0)
        disk = _disk_image(r0)
        inp = (disk * 2.0 - 1.0).unsqueeze(0)

        out = inp
        for _ in range(frames):
            pipeline_ref.prev_image_result = out
            out = proc._process_tensor_core(inp)

        measured = _measure_radius_px(out)
        expected = r0 * (zoom**frames)
        tolerance = max(3.0, 0.25 * expected)
        assert abs(measured - expected) <= tolerance, (
            f"after {frames} frames with DEFAULT inject_radius/inject_softness and a live "
            f"prev, measured radius {measured:.1f}px vs expected ~{expected:.1f}px "
            f"(zoom={zoom}**{frames}) — magnification is not compounding at shipped defaults"
        )


class TestDeforumParityReorder:
    """The pre-warp composite reorder: Deforum warps the previous diffused output and
    only then hands it to img2img — nothing enters the frame after the warp. The old
    order injected prev_image_result and blended the live input AFTER warping the
    canvas, so both sat unwarped (pinned to screen space, one frame behind the
    motion). These tests pin the new order's two signatures and the ephemerality of
    the sharpen/noise detail pass."""

    def test_injected_prev_rides_same_frame_warp(self):
        """A disk present ONLY in prev_image_result must come out zoom-scaled in the
        same call's output. Old order: injection happened after the warp, so the disk
        landed unwarped at exactly r0."""
        zoom = 1.15
        r0 = 16.0
        proc, pipeline_ref = _make_processor(
            feedback_strength=1.0,
            zoom=zoom,
            inject_radius=0.5,
            inject_softness=0.0,
            sharpen_amount=0.0,
            noise_amount=0.0,
            resample_mode="bilinear",
        )
        black = torch.full((1, 3, RES, RES), -1.0)
        proc._process_tensor_core(black)  # seeds the canvas (black), clears _first_frame

        pipeline_ref.prev_image_result = (_disk_image(r0) * 2.0 - 1.0).unsqueeze(0)
        out = proc._process_tensor_core(black)

        measured = _measure_radius_px(out)
        assert measured >= r0 + 1.0, (
            f"disk injected via prev_image_result measured {measured:.1f}px — at ~{r0}px it "
            "was NOT warped in the same call (old post-warp injection order)"
        )
        assert abs(measured - r0 * zoom) <= 1.5, (
            f"injected disk measured {measured:.1f}px, expected ~{r0 * zoom:.1f}px (r0*zoom)"
        )

    def test_live_input_rides_warp_no_static_ghost(self):
        """At feedback_strength<1 the old order blended the unwarped live input in
        after the warp, pinning a screen-locked imprint at exactly r0 forever. Now
        the freshest input contribution is warped too, so the full-intensity edge
        (threshold 0.75 — above the 0.5-weight single-frame contribution) sits at
        r0*zoom."""
        zoom = 1.15
        r0 = 12.0
        proc, _ = _make_processor(
            feedback_strength=0.5,
            zoom=zoom,
            sharpen_amount=0.0,
            noise_amount=0.0,
            resample_mode="bilinear",
        )
        inp = (_disk_image(r0) * 2.0 - 1.0).unsqueeze(0)

        out = None
        for _ in range(3):
            out = proc._process_tensor_core(inp)

        ghost_edge = _measure_radius_px(out, threshold=0.75)
        assert ghost_edge >= r0 + 1.0, (
            f"full-intensity edge at {ghost_edge:.1f}px — the live input is imprinted "
            f"unwarped at ~{r0}px (static screen-locked ghost, old post-warp blend)"
        )
        assert abs(ghost_edge - r0 * zoom) <= 1.5, (
            f"full-intensity edge at {ghost_edge:.1f}px, expected ~{r0 * zoom:.1f}px (r0*zoom)"
        )

    def test_detail_pass_not_persisted_to_canvas(self):
        """Sharpen/noise are Deforum's pre-diffusion seasoning: applied to the
        returned tensor only. Persisting them would compound grain through the
        loop. A constant input keeps the canvas exactly uniform, so any canvas
        variance would be leaked noise."""
        proc, _ = _make_processor(
            feedback_strength=1.0,
            zoom=1.05,
            noise_amount=0.2,
            sharpen_amount=0.0,
            resample_mode="bilinear",
        )
        gray = torch.full((1, 3, RES, RES), 0.5)  # mid-gray, already [0,1]

        out = None
        for _ in range(5):
            out = proc._process_tensor_core(gray)

        out01 = (out[0] + 1.0) / 2.0
        assert (out01 - proc._canvas).abs().mean().item() > 0.05, (
            "output does not differ from the canvas — noise_amount=0.2 is not being applied"
        )
        assert proc._canvas.std().item() < 1e-3, (
            f"canvas std {proc._canvas.std().item():.4f} after 5 noisy frames — per-frame "
            "noise is leaking into the persistent canvas and will compound into grain"
        )

    def test_bicubic_warp_keeps_canvas_in_range(self):
        """Bicubic resampling overshoots [0,1] at high-contrast edges; the post-warp
        clamp must contain it or the excursion compounds through the loop."""
        proc, _ = _make_processor(
            feedback_strength=1.0,
            zoom=1.05,
            resample_mode="bicubic",
            sharpen_amount=0.0,
            noise_amount=0.0,
        )
        inp = (_disk_image(10.0) * 2.0 - 1.0).unsqueeze(0)

        for n in range(10):
            proc._process_tensor_core(inp)
            assert proc._canvas.min().item() >= 0.0, f"frame {n}: canvas min {proc._canvas.min().item()}"
            assert proc._canvas.max().item() <= 1.0, f"frame {n}: canvas max {proc._canvas.max().item()}"

    def test_metadata_and_ctor_defaults_in_lockstep(self):
        """Live OSC updates arrive via setattr behind a hasattr guard
        (stream_parameter_updater.py:1979) — a param present in metadata but absent
        from __init__ is silently dropped. Constructing with no kwargs must therefore
        yield exactly the metadata defaults for every declared parameter."""
        proc, _ = _make_processor()
        meta = proc.get_preprocessor_metadata()
        for name, spec in meta["parameters"].items():
            assert hasattr(proc, name), f"metadata param {name!r} missing on the instance"
            assert getattr(proc, name) == spec["default"], (
                f"param {name!r}: instance has {getattr(proc, name)!r}, metadata default "
                f"is {spec['default']!r} — metadata and __init__ drifted apart"
            )

    def test_inject_radius_range_allows_full_frame(self):
        """inject_radius metadata range must reach 1.0 — the discovery-metadata cap
        used to stop at 0.5 even though the ctor/runtime clamp (max(0.05, min(1.0, ...)))
        already permitted the full range. A processor constructed at the new range
        ceiling must retain the value unclamped."""
        proc, _ = _make_processor()
        meta = proc.get_preprocessor_metadata()
        assert meta["parameters"]["inject_radius"]["range"] == [0.05, 1.0]

        proc_full, _ = _make_processor(inject_radius=1.0)
        assert proc_full.inject_radius == 1.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 long-run needs CUDA")
    def test_fp16_long_run_stays_finite(self):
        pipeline_ref = types.SimpleNamespace(prev_image_result=None)
        from streamdiffusion.preprocessing.processors import get_preprocessor as _gp

        proc = _gp(
            "feedback_loop",
            pipeline_ref=pipeline_ref,
            params={"image_resolution": RES, "device": "cuda", "dtype": torch.float16, "zoom": 1.02},
        )
        inp = torch.rand(1, 3, RES, RES, device="cuda", dtype=torch.float16) * 2.0 - 1.0
        out = inp
        for _ in range(200):
            pipeline_ref.prev_image_result = out
            out = proc._process_tensor_core(inp)
        assert torch.isfinite(out).all(), "NaN/Inf after 200 fp16 frames"
        assert out.min().item() >= -1.0 - 1e-2 and out.max().item() <= 1.0 + 1e-2


class TestGeometricAdvanceDecoupledFromLatency:
    def test_canvas_advances_every_call_even_with_unchanged_prev_output(self):
        """The whole point of the canvas accumulator: geometric advance must not
        stall waiting for a fresh StreamV2V output. Simulate multi-frame diffusion
        lag by handing the SAME prev_image_result tensor to several consecutive
        calls and confirm the canvas still visibly changes every call."""
        proc, pipeline_ref = _make_processor(
            feedback_strength=0.7,
            zoom=1.05,
            inject_radius=0.15,
            inject_softness=0.05,
            sharpen_amount=0.0,
            noise_amount=0.0,
            resample_mode="bilinear",
        )
        disk = _disk_image(15.0)
        inp = (disk * 2.0 - 1.0).unsqueeze(0)

        proc._process_tensor_core(inp)  # seeds the canvas, clears _first_frame

        stale_prev = torch.full((1, 3, RES, RES), 0.3)  # constant "lagging" V2V output
        pipeline_ref.prev_image_result = stale_prev

        prev_canvas = proc._canvas.clone()
        for _ in range(5):
            proc._process_tensor_core(inp)
            assert not torch.allclose(proc._canvas, prev_canvas, atol=1e-6), (
                "canvas did not change between calls despite an unchanged "
                "prev_image_result — geometric advance is stalling on diffusion latency"
            )
            prev_canvas = proc._canvas.clone()


class TestOutputRange:
    def test_output_stays_within_minus_one_to_one(self):
        proc, pipeline_ref = _make_processor(feedback_strength=0.8, zoom=1.05, rotation=3.0, pan_x=0.05)
        pipeline_ref.prev_image_result = torch.rand(1, 3, RES, RES) * 2.0 - 1.0
        inp = torch.rand(1, 3, RES, RES) * 2.0 - 1.0
        for _ in range(10):
            out = proc._process_tensor_core(inp)
            assert out.min().item() >= -1.0 - 1e-4, f"output min {out.min().item()} < -1.0"
            assert out.max().item() <= 1.0 + 1e-4, f"output max {out.max().item()} > 1.0"


class TestBorderModeLiveUpdate:
    def test_border_mode_setattr_takes_effect_next_frame(self):
        """D1 regression: the pre-fix code read border_mode into self._padding_mode
        once in __init__ and never again, so a live OSC setattr(proc, 'border_mode',
        ...) mid-stream silently had no effect. Same processor instance throughout —
        only the attribute is mutated between calls, exactly like the live OSC path.
        Canvas is reset between the two measurements purely to isolate each
        border_mode's behavior in a single warp; it is not part of what's regression-
        tested (the setattr is)."""
        proc, _ = _make_processor(
            feedback_strength=1.0,
            zoom=0.8,
            border_mode="zeros",
            sharpen_amount=0.0,
            noise_amount=0.0,
            resample_mode="bilinear",
        )
        constant = torch.full((1, 3, RES, RES), 0.7 * 2.0 - 1.0)  # uniform 0.7 content

        out_zeros = proc._process_tensor_core(constant)
        corner_zeros = out_zeros[0, 0, 2, 2].item()
        assert corner_zeros < -0.9, f"zeros border_mode: corner {corner_zeros} not near -1.0"

        proc.border_mode = "reflection"
        proc._canvas = None  # force a fresh seed+warp under the new mode

        out_reflect = proc._process_tensor_core(constant)
        corner_reflect = out_reflect[0, 0, 2, 2].item()
        assert corner_reflect > -0.9, (
            f"reflection border_mode: corner {corner_reflect} still pinned near -1.0 — "
            f"setattr(proc, 'border_mode', ...) did not take effect (D1 regression)"
        )
