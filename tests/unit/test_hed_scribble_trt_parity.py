"""CPU-only guards for the HED / scribble TensorRT preprocessors.

Regression context: the original HEDExportWrapper fed the network [0, 1] pixels
(it expects 0-255) and returned only the block-1 side output, so the exporter
dropped four of the five VGG blocks and the scribble map looked like a
texture/contour response.  These tests pin the export contract and the
controlnet_aux-faithful scribble post-process without needing a GPU or TRT.
"""

import types

import pytest
import torch

torch.manual_seed(0)

from streamdiffusion.preprocessing.processors.hed_tensorrt import HEDExportWrapper, HEDTensorrtPreprocessor
from streamdiffusion.preprocessing.processors.scribble_tensorrt import (
    DEFAULT_SCRIBBLE_THRESHOLD,
    ScribbleTensorrtPreprocessor,
    _directional_nms,
    _gaussian_blur,
    _scribble_nms_gpu,
)


class _FakeHED(torch.nn.Module):
    """Mimics ControlNetHED_Apache2's 5 side outputs at H, H/2, H/4, H/8, H/16.

    Records the input it received so the test can check the 0-255 scaling.
    """

    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, x):
        self.seen = x
        b, _, h, w = x.shape
        outs = []
        for k in range(5):
            s = 2**k
            # Constant logit k at each scale, independent of x (shape checks only)
            outs.append(torch.full((b, 1, h // s, w // s), float(k)))
        return outs


# ---------------------------------------------------------------------------
# Export wrapper contract
# ---------------------------------------------------------------------------


def test_wrapper_scales_input_to_0_255():
    net = _FakeHED()
    x = torch.rand(1, 3, 64, 64)
    HEDExportWrapper(net)(x)
    assert torch.allclose(net.seen, x * 255.0)


def test_wrapper_fuses_five_scales_with_sigmoid():
    net = _FakeHED()
    out = HEDExportWrapper(net)(torch.rand(2, 3, 96, 160))
    assert out.shape == (2, 1, 96, 160)
    # mean of logits 0..4 = 2.0 -> sigmoid(2.0); every scale must contribute
    assert torch.allclose(out, torch.sigmoid(torch.tensor(2.0)).expand_as(out), atol=1e-6)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_wrapper_output_name_is_edge_map():
    assert HEDTensorrtPreprocessor.ENGINE_OUTPUT_NAME == "edge_map"
    assert ScribbleTensorrtPreprocessor.ENGINE_OUTPUT_NAME == "edge_map"


# ---------------------------------------------------------------------------
# Stale-engine detection
# ---------------------------------------------------------------------------


def _fake_trt_engine(names):
    eng = types.SimpleNamespace(num_io_tensors=len(names), get_tensor_name=lambda i: names[i])
    return types.SimpleNamespace(engine=eng)


def test_stale_engine_detected_by_output_name():
    proc = HEDTensorrtPreprocessor.__new__(HEDTensorrtPreprocessor)
    assert proc._engine_is_current(_fake_trt_engine(["input", "edge_map"])) is True
    assert proc._engine_is_current(_fake_trt_engine(["input", "output"])) is False


# ---------------------------------------------------------------------------
# Scribble post-process (controlnet_aux nms port)
# ---------------------------------------------------------------------------


def test_gaussian_blur_preserves_mass_and_shape():
    x = torch.zeros(1, 1, 64, 64)
    x[0, 0, 32, 32] = 1.0
    y = _gaussian_blur(x, 3.0)
    assert y.shape == x.shape
    assert abs(y.sum().item() - 1.0) < 1e-4
    assert y[0, 0, 32, 32] > y[0, 0, 32, 36] > y[0, 0, 32, 44]


def test_directional_nms_keeps_ridge_and_drops_slope():
    # Horizontal ridge: row 10 is the max of every column -> vertical scan keeps it.
    x = torch.zeros(1, 1, 21, 21)
    for r in range(21):
        x[0, 0, r, :] = 1.0 - abs(r - 10) / 10.0
    y = _directional_nms(x)
    assert torch.all(y[0, 0, 10, :] == 1.0)

    # Tilted plane r + 2c: strictly monotonic along all four scan lines, so no
    # interior pixel is a local max -> everything except the replicate-padded
    # border corner is suppressed.
    rr, cc = torch.meshgrid(torch.arange(15.0), torch.arange(15.0), indexing="ij")
    plane = (rr + 2.0 * cc).reshape(1, 1, 15, 15)
    yp = _directional_nms(plane)
    assert torch.all(yp[0, 0, 1:-1, 1:-1] == 0.0)
    assert yp[0, 0, 14, 14] == plane[0, 0, 14, 14]  # global max survives


def test_scribble_nms_line_survives_and_thickens():
    h, w = 96, 96
    edge = torch.zeros(h, w)
    edge[48, 16:80] = 1.0  # one-pixel HED ridge at full confidence
    thin = _scribble_nms_gpu(edge, threshold=0.05, thicken=False)
    thick = _scribble_nms_gpu(edge, threshold=0.05, thicken=True)
    assert thin.shape == (h, w)
    assert set(thin.unique().tolist()) <= {0.0, 1.0}
    assert thin[48, 40] == 1.0
    assert thin.sum() < thick.sum()  # thickening widens the stroke
    # The stroke stays localised: nothing far from the ridge
    assert thick[10, 40] == 0.0 and thick[86, 40] == 0.0


def test_scribble_nms_flat_frame_stays_black():
    # Flat low-probability frame (no edges): the old min/max normalisation turned
    # this into all-white; with absolute thresholds it must stay black.
    edge = torch.full((64, 64), 0.05)
    out = _scribble_nms_gpu(edge)
    assert out.sum() == 0.0


def test_scribble_default_threshold_matches_controlnet_aux():
    assert pytest.approx(127 / 255) == DEFAULT_SCRIBBLE_THRESHOLD
    meta = ScribbleTensorrtPreprocessor.get_preprocessor_metadata()["parameters"]["scribble_threshold"]
    assert meta["range"] == [0.0, 1.0]
    assert meta["default"] == pytest.approx(DEFAULT_SCRIBBLE_THRESHOLD, abs=1e-3)


@pytest.mark.skipif(
    pytest.importorskip("cv2", reason="cv2 needed for controlnet_aux nms parity") is None,
    reason="cv2 missing",
)
def test_scribble_nms_matches_controlnet_aux_reference():
    """Bit-for-bit-ish parity with controlnet_aux.util.nms + the thickening step."""
    import cv2
    import numpy as np
    from controlnet_aux.util import nms as ref_nms

    torch.manual_seed(1)
    # Smooth random field with a few strong ridges
    field = _gaussian_blur(torch.rand(1, 1, 128, 128), 2.0)[0, 0]
    field = (field - field.min()) / (field.max() - field.min())
    field[64, :] = 1.0
    field[:, 40] = 0.9

    # Reference (CPU, uint8 like HEDdetector.__call__)
    ref_u8 = (field.numpy() * 255.0).clip(0, 255).astype(np.uint8)
    ref = ref_nms(ref_u8, 127, 3.0)
    ref = cv2.GaussianBlur(ref, (0, 0), 3.0)
    ref = (ref > 4).astype(np.float32)

    ours = _scribble_nms_gpu(field).numpy()
    agree = (ours == ref).mean()
    assert agree > 0.98, f"only {agree:.3%} pixel agreement with controlnet_aux"
