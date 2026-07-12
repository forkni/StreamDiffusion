from typing import Any, Optional, Tuple, Union

import torch
from PIL import Image

from streamdiffusion.tools.gpu_profiler import profiler

from .base import BasePreprocessor


class IPAdapterEmbeddingPreprocessor(BasePreprocessor):
    """
    Preprocessor that generates IPAdapter embeddings instead of spatial conditioning.
    Leverages existing preprocessing infrastructure for parallel IPAdapter embedding generation.
    """

    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "IPAdapter Embedding",
            "description": "Generates IPAdapter embeddings for style transfer and image conditioning instead of spatial control maps.",
            "parameters": {},
            "use_cases": ["Style transfer", "Image conditioning", "Semantic control", "Content-aware generation"],
        }

    def __init__(self, ipadapter: Any, **kwargs):
        super().__init__(**kwargs)
        self.ipadapter = ipadapter
        # Verify the ipadapter has the required method
        if not hasattr(ipadapter, "get_image_embeds"):
            raise ValueError("IPAdapterEmbeddingPreprocessor: ipadapter must have 'get_image_embeds' method")

        # Create dedicated CUDA stream for IPAdapter processing to avoid TensorRT conflicts
        self._ipadapter_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        # CUDA event for GPU-side stream sync — CPU thread NOT blocked (vs .synchronize()).
        # Lazily allocated on first _process_core call so the constructor stays CUDA-context free.
        self._completion_event: Optional[torch.cuda.Event] = None

        # Per-preprocessor embedding cache: avoids CLIP re-encode when the style image is
        # unchanged across consecutive frames (the common streaming scenario).
        # Keyed by tensor.data_ptr() — stable while the storage is live, changes on realloc.
        self._last_input_ptr: int = -1
        self._cached_embeds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _process_core(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (positive_embeds, negative_embeds) instead of processed image"""
        if self._ipadapter_stream is not None:
            # Lazy-init the completion event (avoids CUDA context init in constructor).
            if self._completion_event is None:
                self._completion_event = torch.cuda.Event()

            # Use dedicated stream to avoid TensorRT stream capture conflicts
            with torch.cuda.stream(self._ipadapter_stream):
                with profiler.region("ipa.clip_encode"):
                    image_embeds, negative_embeds = self.ipadapter.get_image_embeds(images=[image])
                # Record the event on the IPA stream immediately after encode.
                # The default stream will GPU-wait on this event; the CPU thread is NOT blocked.
                with profiler.region("ipa.sync"):
                    self._completion_event.record()

            # GPU-side dependency: the default stream defers until the IPA stream event fires.
            # Replaces the blocking _ipadapter_stream.synchronize() — CPU thread continues now.
            torch.cuda.current_stream().wait_event(self._completion_event)

            # Mark tensors as owned by the default stream (cross-stream memory safety)
            if hasattr(image_embeds, "record_stream"):
                image_embeds.record_stream(torch.cuda.current_stream())
            if hasattr(negative_embeds, "record_stream"):
                negative_embeds.record_stream(torch.cuda.current_stream())
        else:
            # Fallback for non-CUDA environments
            with profiler.region("ipa.clip_encode"):
                image_embeds, negative_embeds = self.ipadapter.get_image_embeds(images=[image])

        return image_embeds, negative_embeds

    def _process_tensor_core(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """GPU-optimized path for tensor inputs.

        Checks the per-preprocessor embedding cache before running CLIP encode.
        CLIP requires PIL input, so the GPU→CPU tensor_to_pil step is unavoidable;
        caching avoids it on frames where the style image is unchanged.
        """
        # Cache check: skip CLIP re-encode if the tensor's storage pointer is unchanged.
        # data_ptr() is stable as long as the tensor storage is not reallocated, which
        # is the common case in streaming (same style-image tensor reused across frames).
        current_ptr = tensor.data_ptr() if tensor.is_cuda else id(tensor)
        if self._cached_embeds is not None and current_ptr == self._last_input_ptr:
            return self._cached_embeds

        pil_image = self.tensor_to_pil(tensor)
        result = self._process_core(pil_image)

        # Update cache for next frame
        self._last_input_ptr = current_ptr
        self._cached_embeds = result
        return result

    def process(self, image: Union[Image.Image, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Override base process to return embeddings tuple instead of PIL Image"""
        if isinstance(image, torch.Tensor):
            result = self._process_tensor_core(image)
        else:
            image = self.validate_input(image)
            result = self._process_core(image)

        return result

    def process_tensor(self, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Override base process_tensor to return embeddings tuple"""
        tensor = self.validate_tensor_input(image_tensor)
        return self._process_tensor_core(tensor)
