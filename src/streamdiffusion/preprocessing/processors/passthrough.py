import numpy as np
from PIL import Image
import torch
from typing import Union, Optional
from .base import BasePreprocessor


class PassthroughPreprocessor(BasePreprocessor):
    """
    Passthrough preprocessor for ControlNet
    
    Simply passes the input image through without any processing.
    Useful for ControlNets that expect the raw input image, such as:
    - Tile ControlNet
    - Reference ControlNet
    - Custom ControlNets that don't need preprocessing
    """

    gpu_native = True  # _process_tensor_core is a no-op identity — no CPU/PIL round-trip


    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Passthrough",
            "description": (
                "Sends the input image directly to the ControlNet with no preprocessing.  "
                "Use when the input is already a pre-rendered conditioning map — e.g. a "
                "depth pass, hand-drawn scribble, or OpenPose skeleton rendered externally."
            ),
            "parameters": {},
            "use_cases": [
                "Pre-rendered depth / normal maps",
                "Hand-drawn scribble inputs",
                "Externally generated pose skeletons",
                "Image-to-image with structure preservation",
            ],
        }
    
    def __init__(self, 
                 image_resolution: int = 512,
                 **kwargs):
        """
        Initialize passthrough preprocessor
        
        Args:
            image_resolution: Output image resolution
            **kwargs: Additional parameters (ignored for passthrough)
        """
        super().__init__(
            image_resolution=image_resolution,
            **kwargs
        )
    
    def _process_core(self, image: Image.Image) -> Image.Image:
        """
        Pass through the input image with no processing
        """
        return image
    
    def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Pass through tensor with no processing
        """
        return tensor 