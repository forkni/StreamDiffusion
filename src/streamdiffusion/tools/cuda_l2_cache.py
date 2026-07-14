"""
L2 Cache Persistence Utility for StreamDiffusion UNet.

Reserves a portion of the GPU's L2 cache for persistent data (UNet weights),
reducing cache evictions for memory-bandwidth-bound layers.

Requires: CUDA >= 11.2, compute capability >= 8.0 (Ampere+).
RTX 5090 has 128MB L2, compute 12.0 — full support.

Control precedence (highest to lowest):
    1. Explicit kwargs to setup_l2_persistence() (e.g. the wrapper's `l2_persist` config key).
    2. Environment variables below, read at CALL time (not import time, so a value set after
       this module first imports -- e.g. by TouchDesigner's embedded Python -- still applies).
    3. Mode-aware default: off when acceleration=="tensorrt" (Tier 1 is an inert soft
       carve-out on a serialized TRT engine, and Tier 2 requires an nn.Module UNet), on
       otherwise.

Environment variables:
    SDTD_L2_PERSIST=1         Enable L2 persistence (default: on, except TRT mode -- see above)
    SDTD_L2_PERSIST_MB=64     MB of L2 to reserve for persistent data (default: 64)
    SDTD_L2_PERSIST_TIER2=0   Enable per-tensor access policy window (default: 0, nn.Module UNet only)
    SDTD_L2_PERSIST_LAYERS=   Comma-separated layer names for access policy (default: auto)

Expected impact: 5-16% on memory-bandwidth-bound layers (normalization, small GEMMs).
Hot layers on SDXL: mid_block, up_blocks.1 (most FF hooks + V2V cached attention).
"""

import ctypes
import os
import sys
from typing import Optional

import torch

# =============================================================================
# Environment Controls (read at call time -- see module docstring precedence)
# =============================================================================


def _env_enabled() -> Optional[bool]:
    v = os.environ.get("SDTD_L2_PERSIST")
    return None if v is None else v == "1"  # None = "unset" so it can fall through


def _env_persist_mb() -> int:
    return int(os.environ.get("SDTD_L2_PERSIST_MB", "64"))


def _env_tier2() -> bool:
    return os.environ.get("SDTD_L2_PERSIST_TIER2", "0") == "1"


# Hot layer prefixes — these contain the most attention + FF hook computation.
# mid_block: 1 transformer block, seq_len=1024, 16 FF hooks
# up_blocks.1: up-sampling path, seq_len=4096
_DEFAULT_HOT_LAYER_PREFIXES = ["mid_block", "up_blocks.1"]


# =============================================================================
# CUDA Runtime Access Policy Structs (for Tier 2 per-tensor persistence)
# =============================================================================


class _CudaAccessPolicyWindow(ctypes.Structure):
    """cudaAccessPolicyWindow struct for cudaStreamSetAttribute."""

    _fields_ = [
        ("base_ptr", ctypes.c_void_p),  # void* — start of memory region
        ("num_bytes", ctypes.c_size_t),  # size_t — size of region in bytes
        ("hitRatio", ctypes.c_float),  # float — fraction in [0, 1] to keep persistent
        (
            "hitProp",
            ctypes.c_int,
        ),  # cudaAccessProperty: 2 = cudaAccessPropertyPersisting
        (
            "missProp",
            ctypes.c_int,
        ),  # cudaAccessProperty: 1 = cudaAccessPropertyStreaming
    ]


# cudaAccessProperty enum values
_CUDA_ACCESS_PROPERTY_NORMAL = 0
_CUDA_ACCESS_PROPERTY_STREAMING = 1
_CUDA_ACCESS_PROPERTY_PERSISTING = 2

# cudaStreamAttrID
_CUDA_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1

# cudaLimit enum (CUDA 11.2+)
# 0x06 = cudaLimitPersistingL2CacheSize (size in bytes — correct for L2 persistence)
_CUDA_LIMIT_PERSISTING_L2_CACHE_SIZE = 0x06


# =============================================================================
# CUDA Runtime Handle
# =============================================================================

_cudart: Optional[ctypes.CDLL] = None
_cudart_loaded: bool = False


def _get_cudart() -> Optional[ctypes.CDLL]:
    """Load the CUDA runtime DLL. Cached after first call."""
    global _cudart, _cudart_loaded
    if _cudart_loaded:
        return _cudart

    _cudart_loaded = True

    if sys.platform != "win32":
        # Non-Windows: use libcudart.so — typically already loaded by PyTorch
        try:
            _cudart = ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
            return _cudart
        except OSError:
            pass
        try:
            from ctypes.util import find_library

            lib = find_library("cudart")
            if lib:
                _cudart = ctypes.CDLL(lib)
                return _cudart
        except OSError:
            pass
        return None

    # Windows: find cudart64_*.dll shipped with PyTorch or CUDA toolkit
    import glob

    # Option 1: PyTorch ships cudart in torch/lib/
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    candidates = sorted(glob.glob(os.path.join(torch_lib, "cudart64_*.dll")), reverse=True)

    # Option 2: CUDA toolkit installation
    cuda_path = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8")
    candidates += sorted(glob.glob(os.path.join(cuda_path, "bin", "cudart64_*.dll")), reverse=True)

    for dll_path in candidates:
        try:
            _cudart = ctypes.WinDLL(dll_path)
            return _cudart
        except OSError:
            continue

    return None


# =============================================================================
# Tier 1: Reserve L2 Persisting Cache Size
# =============================================================================


def reserve_l2_persisting_cache(persist_mb: Optional[int] = None) -> bool:
    """
    Reserve a portion of L2 cache for persistent data.

    This is Tier 1 of L2 persistence: informs the driver that `persist_mb` MB
    of L2 should not be evicted by regular (streaming) accesses. Hot data set
    via access policy windows will preferentially stay in this reserved region.

    Args:
        persist_mb: Megabytes of L2 to reserve. Should be <= half of total L2.
                    RTX 5090 has 128MB L2 → 64MB is a safe default.
                    None -> resolved from SDTD_L2_PERSIST_MB at call time.

    Returns:
        True if successful, False if unsupported or failed.
    """
    if persist_mb is None:
        persist_mb = _env_persist_mb()

    if not torch.cuda.is_available():
        return False

    # Check compute capability — L2 persistence requires Ampere (8.0+)
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    major, minor = props.major, props.minor
    if major < 8:
        print(f"[L2] L2 persistence skipped — compute {major}.{minor} < 8.0 (Ampere required)")
        return False

    l2_total_mb = props.L2_cache_size // (1024 * 1024)
    persist_bytes = min(persist_mb * 1024 * 1024, props.L2_cache_size // 2)
    persist_mb_actual = persist_bytes // (1024 * 1024)

    cudart = _get_cudart()
    if cudart is None:
        print("[L2] CUDA runtime not found — L2 persistence unavailable")
        return False

    try:
        result = cudart.cudaDeviceSetLimit(
            ctypes.c_int(_CUDA_LIMIT_PERSISTING_L2_CACHE_SIZE),
            ctypes.c_size_t(persist_bytes),
        )
        # CRITICAL: Always clear CUDA error state after ctypes calls.
        # cudaDeviceSetLimit sets the thread-local CUDA error on failure, and
        # PyTorch's C10_CUDA_KERNEL_LAUNCH_CHECK() reads it on the next kernel
        # launch — causing a stale error to crash an unrelated operation.
        cudart.cudaGetLastError()
        if result != 0:
            print(f"[L2] cudaDeviceSetLimit failed: error {result}")
            return False

        print(
            f"[L2] Reserved {persist_mb_actual}MB of {l2_total_mb}MB L2 for persisting cache "
            f"(compute {major}.{minor}, {props.name})"
        )
        return True

    except (OSError, ctypes.ArgumentError, AttributeError) as e:
        print(f"[L2] L2 reservation failed: {e}")
        return False


# =============================================================================
# Tier 2: Per-Tensor Access Policy (stream attribute window)
# =============================================================================


def set_tensor_persisting(tensor: torch.Tensor, hit_ratio: float = 1.0) -> bool:
    """
    Mark a tensor's memory region as L2-persistent.

    Uses cudaStreamSetAttribute with cudaAccessPolicyWindow to request that
    `hit_ratio` fraction of the tensor's data stays in the L2 persisting region.

    Args:
        tensor: CUDA tensor whose weights should persist in L2.
        hit_ratio: Fraction [0, 1] of accesses to serve from persisting cache.
                   1.0 = always try to keep in L2 (good for weights).

    Returns:
        True if successful.
    """
    if not tensor.is_cuda or not tensor.is_contiguous():
        return False

    cudart = _get_cudart()
    if cudart is None:
        return False

    try:
        stream_ptr = torch.cuda.current_stream().cuda_stream

        window = _CudaAccessPolicyWindow(
            base_ptr=ctypes.c_void_p(tensor.data_ptr()),
            num_bytes=ctypes.c_size_t(tensor.nbytes),
            hitRatio=ctypes.c_float(hit_ratio),
            hitProp=ctypes.c_int(_CUDA_ACCESS_PROPERTY_PERSISTING),
            missProp=ctypes.c_int(_CUDA_ACCESS_PROPERTY_STREAMING),
        )

        result = cudart.cudaStreamSetAttribute(
            ctypes.c_void_p(stream_ptr),
            ctypes.c_int(_CUDA_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW),
            ctypes.byref(window),
        )
        cudart.cudaGetLastError()  # Clear any stale CUDA error from ctypes call
        return result == 0

    except (RuntimeError, OSError, ctypes.ArgumentError, AttributeError):
        return False


def clear_tensor_persisting(tensor: torch.Tensor) -> bool:
    """
    Remove L2 persistence policy for a tensor (reset to normal access).

    Call when a tensor is no longer hot (e.g., model unload) to release
    the L2 persisting budget for other tensors.
    """
    if not tensor.is_cuda:
        return False

    cudart = _get_cudart()
    if cudart is None:
        return False

    try:
        stream_ptr = torch.cuda.current_stream().cuda_stream
        window = _CudaAccessPolicyWindow(
            base_ptr=ctypes.c_void_p(tensor.data_ptr()),
            num_bytes=ctypes.c_size_t(0),
            hitRatio=ctypes.c_float(0.0),
            hitProp=ctypes.c_int(_CUDA_ACCESS_PROPERTY_NORMAL),
            missProp=ctypes.c_int(_CUDA_ACCESS_PROPERTY_STREAMING),
        )
        result = cudart.cudaStreamSetAttribute(
            ctypes.c_void_p(stream_ptr),
            ctypes.c_int(_CUDA_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW),
            ctypes.byref(window),
        )
        cudart.cudaGetLastError()  # Clear any stale CUDA error from ctypes call
        return result == 0
    except (RuntimeError, OSError, ctypes.ArgumentError, AttributeError):
        return False


# =============================================================================
# High-Level: Pin Hot UNet Layer Weights
# =============================================================================


def pin_hot_unet_weights(
    unet: torch.nn.Module,
    hot_prefixes: Optional[list] = None,
    persist_mb: Optional[int] = None,
) -> int:
    """
    Mark the single largest hot UNet attention weight as L2-persistent.

    CUDA allows only one cudaAccessPolicyWindow per stream at a time — registering
    multiple tensors silently replaces the previous window.  This function correctly
    picks the single largest hot attention weight (by byte size) and registers exactly
    one window for it.

    Args:
        unet: The UNet model (already on CUDA, must be torch.nn.Module).
        hot_prefixes: Layer name prefixes to target. Defaults to mid_block + up_blocks.1.
        persist_mb: MB of L2 to reserve (passed to reserve_l2_persisting_cache).
                    None -> resolved from SDTD_L2_PERSIST_MB at call time.

    Returns:
        1 if a tensor was pinned, 0 otherwise.

    Note:
        No on/off gate here -- the sole caller, setup_l2_persistence, already resolves
        `enabled` with the full precedence (config > env > mode default) before calling
        this. Re-checking a raw env value here would let a stale SDTD_L2_PERSIST=0 veto
        an explicit enabled=True config override.
    """
    if persist_mb is None:
        persist_mb = _env_persist_mb()

    if not isinstance(unet, torch.nn.Module):
        print("[L2] Tier 2 skipped — model is not nn.Module (e.g. TRT engine). Use Tier 1 only.")
        return 0

    if hot_prefixes is None:
        hot_prefixes = _DEFAULT_HOT_LAYER_PREFIXES

    # Tier 1: Reserve L2 persisting region (skip if persist_mb=0, caller already reserved)
    if persist_mb > 0:
        tier1_ok = reserve_l2_persisting_cache(persist_mb)
        if not tier1_ok:
            return 0

    # Tier 2: Find the single largest hot attention weight.
    # CUDA allows only one cudaAccessPolicyWindow per stream — registering N tensors
    # results in only the Nth window being active (each call replaces the previous).
    # Pinning the largest tensor maximises L2 utilization for the one permitted window.
    _hot_weight_keywords = ["to_q", "to_k", "to_v", "to_out"]
    best_tensor = None
    best_bytes = 0
    candidate_count = 0

    for name, param in unet.named_parameters():
        if not param.is_cuda:
            continue
        is_hot = any(prefix in name for prefix in hot_prefixes)
        is_attn_weight = any(kw in name for kw in _hot_weight_keywords)
        if is_hot and is_attn_weight:
            candidate_count += 1
            if param.data.nbytes > best_bytes:
                best_bytes = param.data.nbytes
                best_tensor = param.data

    if best_tensor is not None and set_tensor_persisting(best_tensor):
        print(
            f"[L2] Pinned 1 of {candidate_count} hot tensors (largest, "
            f"{best_bytes / 1024 / 1024:.1f}MB) — single-window CUDA limit applies"
        )
        return 1

    if candidate_count == 0:
        print("[L2] No tensors pinned (params may require_grad=True before compile — call after freeze)")
    return 0


def setup_l2_persistence(
    unet: torch.nn.Module,
    *,
    enabled: Optional[bool] = None,
    acceleration: Optional[str] = None,
    persist_mb: Optional[int] = None,
    tier2: Optional[bool] = None,
) -> bool:
    """
    Main entry point: set up L2 cache persistence for UNet inference.

    Call this AFTER model is loaded and BEFORE torch.compile.
    For best results with frozen weights, call AFTER torch.compile with freezing=True.

    Tier 1 (L2 set-aside via cudaDeviceSetLimit) is a soft carve-out: it reserves a
    portion of L2 for hot data but is INERT on a serialized TRT engine (nothing is ever
    tagged persistent without Tier 2, which requires an nn.Module). Tier 2 (per-tensor
    access policy window) is opt-in via SDTD_L2_PERSIST_TIER2=1 / tier2=True. It only
    works for PyTorch nn.Module UNets (not TRT engines), and CUDA allows only one window
    per stream — this function registers only the single largest hot tensor.

    Precedence (highest to lowest) for `enabled`: explicit kwarg > SDTD_L2_PERSIST env
    (read here, at call time) > mode-aware default (off when acceleration=="tensorrt",
    since Tier 1 is inert there and Tier 2 is impossible; on otherwise). This inverts
    gpu_profiler's env-over-config precedence ON PURPOSE — TouchDesigner's embedded
    Python cannot set shell env vars but can write config, so config must win.

    Args:
        unet: The UNet model on CUDA (nn.Module for Tier-2 to apply; TRT Engine for Tier-1 only).
        enabled: Explicit on/off override (the wrapper's `l2_persist` config key). None
                 falls through to the env/mode-default resolution described above.
        acceleration: The active acceleration backend (e.g. "tensorrt"), used only for
                      the mode-aware default when `enabled` and the env var are both unset.
        persist_mb: MB of L2 to reserve. None -> SDTD_L2_PERSIST_MB (default 64).
        tier2: Explicit Tier-2 on/off override. None -> SDTD_L2_PERSIST_TIER2 (default off).

    Returns:
        True if at least Tier 1 (L2 reservation) succeeded.
    """
    if enabled is None:
        enabled = _env_enabled()
    if enabled is None:
        enabled = acceleration != "tensorrt"
    if not enabled:
        return False  # silent: no [L2] log when off (e.g. TRT/Performance mode by default)

    if persist_mb is None:
        persist_mb = _env_persist_mb()
    if tier2 is None:
        tier2 = _env_tier2()

    print(f"\n[L2] Setting up L2 cache persistence (SDTD_L2_PERSIST_MB={persist_mb})...")

    # Tier 1: Reserve L2 persisting region — works for all GPU modes, always attempt.
    tier1_ok = reserve_l2_persisting_cache(persist_mb)

    if tier1_ok:
        if tier2:
            # Tier 2: per-tensor access policy window — opt-in, nn.Module only.
            pin_hot_unet_weights(unet, persist_mb=0)  # Tier 1 already reserved above
        else:
            print(
                "[L2] Tier 2 access policy disabled (set SDTD_L2_PERSIST_TIER2=1 to enable; nn.Module UNet required)"
            )

    return tier1_ok
