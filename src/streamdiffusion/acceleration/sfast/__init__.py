from typing import Optional

from sfast.compilers.stable_diffusion_pipeline_compiler import CompilationConfig, compile

from ...pipeline import StreamDiffusion


def accelerate_with_stable_fast(
    stream: StreamDiffusion,
    config: Optional[CompilationConfig] = None,
):
    if config is None:
        config = CompilationConfig.Default()
        # xformers and Triton are suggested for achieving best performance.
        try:
            import xformers

            config.enable_xformers = True
        except ImportError:
            print("xformers not installed, skip")
        try:
            import triton

            config.enable_triton = True
        except ImportError:
            print("Triton not installed, skip")
        # CUDA Graph reduces CPU overhead for small batches/resolutions.
        # Disable when the UNet is a TRT engine (which has its own CUDA-graph regime)
        # to avoid double-capture overhead and potential replay conflicts.
        # TRT engines expose `dump_profile`; standard nn.Module does not.
        _unet = getattr(stream.pipe, "unet", None)
        _trt_active = _unet is not None and hasattr(_unet, "dump_profile")
        config.enable_cuda_graph = not _trt_active
    stream.pipe = compile(stream.pipe, config)
    stream.unet = stream.pipe.unet
    stream.vae = stream.pipe.vae
    stream.text_encoder = stream.pipe.text_encoder
    return stream
