import io
from importlib import import_module
from types import ModuleType

import torch
from PIL import Image
from torchvision.io import decode_jpeg, encode_jpeg


def get_pipeline_class(pipeline_name: str) -> ModuleType:
    try:
        module = import_module(f"pipelines.{pipeline_name}")
    except ModuleNotFoundError:
        raise ValueError(f"Pipeline {pipeline_name} module not found") from None

    pipeline_class = getattr(module, "Pipeline", None)

    if pipeline_class is None:
        raise ValueError(f"'Pipeline' class not found in module '{pipeline_name}'.")

    return pipeline_class


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes))
    return image


def bytes_to_pt(image_bytes: bytes) -> torch.Tensor:
    """
    Convert JPEG bytes directly to a GPU float32 tensor via torchvision nvJPEG.

    Decodes on CUDA when available (nvJPEG path), eliminating the CPU decode +
    host→device DMA transfer that the CPU path incurs.  Falls back to CPU decode
    on machines without CUDA.

    Args:
        image_bytes: Raw JPEG bytes (PNG bytes fall back to CPU automatically
                     since nvJPEG only handles JPEG)


    Returns:
        torch.Tensor: Image tensor with shape (C, H, W), values in [0, 1],
                      dtype float32, on the same device as the decode.
    """
    byte_tensor = torch.frombuffer(image_bytes, dtype=torch.uint8)

    # Decode directly on GPU when CUDA is available — nvJPEG avoids the
    # CPU decode + H2D copy incurred by the plain decode_jpeg(byte_tensor) call.
    if torch.cuda.is_available():
        image_tensor = decode_jpeg(byte_tensor, device="cuda")
    else:
        image_tensor = decode_jpeg(byte_tensor)

    # Normalise to [0, 1] on the decode device (fused kernel on GPU).
    image_tensor = image_tensor.float() / 255.0

    return image_tensor


def pil_to_frame(image: Image.Image) -> bytes:
    frame_data = io.BytesIO()
    image.save(frame_data, format="JPEG")
    frame_data = frame_data.getvalue()
    return (
        b"--frame\r\n"
        + b"Content-Type: image/jpeg\r\n"
        + f"Content-Length: {len(frame_data)}\r\n\r\n".encode()
        + frame_data
        + b"\r\n"
    )


def pt_to_frame(tensor: torch.Tensor) -> bytes:
    """
    Convert PyTorch tensor directly to JPEG frame bytes using torchvision

    Args:
        tensor: PyTorch tensor with shape (C, H, W) or (1, C, H, W), values in [0, 1]

    Returns:
        bytes: JPEG frame data for streaming
    """
    # Handle batch dimension - take first image if batched
    if tensor.dim() == 4:
        tensor = tensor[0]

    # Convert to uint8 format (0-255) and ensure correct shape (C, H, W)
    tensor_uint8 = (tensor * 255).clamp(0, 255).to(torch.uint8)

    # Encode directly to JPEG bytes using torchvision
    jpeg_bytes = encode_jpeg(tensor_uint8, quality=90)
    frame_data = jpeg_bytes.cpu().numpy().tobytes()

    return (
        b"--frame\r\n"
        + b"Content-Type: image/jpeg\r\n"
        + f"Content-Length: {len(frame_data)}\r\n\r\n".encode()
        + frame_data
        + b"\r\n"
    )


def is_firefox(user_agent: str) -> bool:
    return "Firefox" in user_agent
