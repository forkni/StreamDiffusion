"""
Inference-side error report (schema v1).

Builds a self-contained, human-readable diagnostic report when an inference-time
exception is caught (e.g. the TouchDesigner streaming loop), and backs a manual
convenience method on StreamDiffusionWrapper.

Best-effort by design: write_error_report() must never raise past its own
try/except, so a bug in the reporter never masks the original inference error.
This mirrors StreamDiffusion-installer/sd_installer/report.py's schema v1 layout;
the two are deliberately not shared code since they live in separate git repos.
"""

from __future__ import annotations

import collections
import importlib.metadata
import logging
import os
import platform
import re
import subprocess
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


SCHEMA_VERSION = "v1"
# Only these env-var prefixes are dumped -- never the full os.environ (avoids leaking secrets).
ENV_ALLOWLIST_PREFIXES = ("CUDALINK_", "HF_", "SD_", "SDTD_")
# Even an allowlisted-prefix var is dropped if its name contains one of these substrings --
# a prefix match alone isn't enough (e.g. HF_TOKEN matches "HF_" but must never be dumped).
ENV_DENYLIST_SUBSTRINGS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "PASSWD", "CRED", "AUTH", "SESSION", "COOKIE")
# Packages reported in == VERSIONS ==. Looked up individually so one missing/optional
# package (e.g. tensorrt on a non-TensorRT install) never blanks the rest.
VERSION_PACKAGES = (
    "streamdiffusion",
    "torch",
    "numpy",
    "transformers",
    "diffusers",
    "tensorrt",
    "onnx",
    "xformers",
    "cuda-link",
)
# Wrapper attrs pulled into == CONFIG ==. Leading underscore is stripped for display.
WRAPPER_CONFIG_ATTRS = (
    "width",
    "height",
    "batch_size",
    "fp8",
    "static_shapes",
    "_acceleration",
    "_engine_dir",
    "use_controlnet",
    "use_ipadapter",
)

_LOG_TAIL_MAXLEN = 200
_log_tail_buffer: collections.deque = collections.deque(maxlen=_LOG_TAIL_MAXLEN)
_log_tail_lock = threading.Lock()
_log_tail_installed = False


class _TailBufferHandler(logging.Handler):
    """Bounded in-memory ring buffer of recent formatted log lines, for LOG TAIL."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            _log_tail_buffer.append(self.format(record))
        except Exception:
            pass  # never let log-tail capture disrupt logging itself


def _install_log_tail_handler() -> None:
    """Attach the tail-buffer handler to the root logger, once per process. Best-effort."""
    global _log_tail_installed
    with _log_tail_lock:
        if _log_tail_installed:
            return
        try:
            handler = _TailBufferHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            handler.setLevel(logging.INFO)
            logging.getLogger().addHandler(handler)
        except Exception:
            pass
        finally:
            _log_tail_installed = True


def _get_log_tail(n: int = 50) -> list:
    """Return up to the last n buffered log lines. Best-effort; empty list if unavailable."""
    try:
        return list(_log_tail_buffer)[-n:]
    except Exception:
        return []


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _nvidia_smi_driver() -> str:
    """Query the NVIDIA driver version via nvidia-smi. Best-effort."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return "unknown"


def _collect_system(wrapper: Any = None) -> Dict[str, Any]:
    """OS / Python / CUDA driver / GPU / VRAM / compute capability. Deliberately queries
    torch.cuda directly rather than acceleration/tensorrt/utilities.detect_gpu_profile() --
    that module imports tensorrt+onnx at module scope, which are optional deps this
    lightweight collector should not require."""
    info: Dict[str, Any] = {
        "os": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "nvidia_driver": _nvidia_smi_driver(),
    }
    try:
        import torch

        info["cuda_runtime"] = str(torch.version.cuda)
        if torch.cuda.is_available():
            wrapper_device = getattr(wrapper, "device", None)
            device_index = torch.device(wrapper_device).index if wrapper_device is not None else None
            if device_index is None:
                device_index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_index)
            info["gpu_name"] = props.name
            info["compute_capability"] = f"{props.major}.{props.minor}"
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            info["vram_total_mb"] = total_bytes // (1024 * 1024)
            info["vram_free_mb"] = free_bytes // (1024 * 1024)
        else:
            info["gpu_name"] = "no CUDA device available"
    except Exception as torch_exc:
        info["torch_error"] = str(torch_exc)
    return info


def _collect_versions() -> Dict[str, str]:
    versions = {}
    for pkg in VERSION_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
        except Exception as e:
            versions[pkg] = f"error: {e}"
    return versions


def _collect_config(wrapper: Any) -> Dict[str, Any]:
    if wrapper is None:
        return {}
    config: Dict[str, Any] = {}
    for attr in WRAPPER_CONFIG_ATTRS:
        try:
            config[attr.lstrip("_")] = getattr(wrapper, attr)
        except Exception:
            pass
    return config


def _collect_env_allowlist() -> Dict[str, str]:
    """Collect only allow-listed env vars -- never dump full os.environ (secrets)."""
    return {
        k: v
        for k, v in sorted(os.environ.items())
        if k.startswith(ENV_ALLOWLIST_PREFIXES) and not any(bad in k.upper() for bad in ENV_DENYLIST_SUBSTRINGS)
    }


_REDACTED = "***REDACTED***"


def _redact_secrets(value: Any) -> Any:
    """Recursively mask dict values whose key looks secret-ish (reuses ENV_DENYLIST_SUBSTRINGS),
    so a caller-supplied stream/pipeline config (arbitrary, unlike the wrapper-attr allowlist
    used for `config`) can't leak a nested hf_token/api_key/password into a shared report."""
    if isinstance(value, dict):
        return {
            k: (_REDACTED if any(bad in str(k).upper() for bad in ENV_DENYLIST_SUBSTRINGS) else _redact_secrets(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return value


def _render_stream_config(config: Optional[Dict[str, Any]]) -> str:
    """Best-effort pretty-print of the raw stream/pipeline config dict for == STREAM CONFIG ==.
    Tries YAML first (most readable for nested pipeline configs), falls back to JSON, then repr --
    this must never raise and block the rest of the report from writing."""
    if not config:
        return "(none)"
    try:
        import yaml

        return yaml.safe_dump(config, default_flow_style=False, sort_keys=False).rstrip("\n")
    except Exception:
        pass
    try:
        import json

        return json.dumps(config, indent=2, default=str)
    except Exception:
        return repr(config)


def _format_section(title: str, lines: list) -> str:
    body = "\n".join(str(line) for line in lines) if lines else "(none)"
    return f"== {title} ==\n{body}\n"


def collect_diagnostics(wrapper: Any = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Collect best-effort system/version/config/env diagnostics.

    Every collector runs in its own try/except so a failure in one section (e.g. no
    CUDA device present) never prevents the rest of the report from forming.

    Args:
        wrapper: StreamDiffusionWrapper instance to pull config attrs from, if available.
        extra: Extra key/value pairs merged into the CONFIG section (e.g. {"where": "..."}).

    Returns:
        {"system": {...}, "versions": {...}, "config": {...}, "env": {...}}
    """

    def _safe(fn, default):
        try:
            return fn()
        except Exception:
            return default

    config = _safe(lambda: _collect_config(wrapper), {})
    if extra:
        config.update(_redact_secrets({str(k): v for k, v in extra.items()}))

    return {
        "system": _safe(lambda: _collect_system(wrapper), {}),
        "versions": _safe(_collect_versions, {}),
        "config": config,
        "env": _safe(_collect_env_allowlist, {}),
    }


def format_report_text(diag: Dict[str, Any]) -> str:
    """
    Build the schema-v1 diagnostic report text from a diagnostics dict.

    Expected keys (all optional unless noted):
        stage (str): "inference" (defaults to "inference")
        error (str): summary error line, e.g. "TypeError: ..."
        context_note (str): short label for SUMMARY, e.g. "streaming_loop"
        traceback (str): pre-formatted traceback text
        system, versions, config, env: dicts, as returned by collect_diagnostics()
        stream_config (dict): raw stream/pipeline config dict (e.g. td_manager's loaded
            YAML), complementary to `config` which only holds resolved wrapper attrs
        log_tail (list[str]): recent buffered log lines

    Returns:
        Full report text, ready to write to disk.
    """
    stage = diag.get("stage", "inference")
    lines = [
        "StreamDiffusionTD Error Report   (schema v1)",
        f"Generated: {_utc_now().isoformat()}",
        f"Stage: {stage}",
        "-" * 50,
        "",
    ]

    summary_lines = [f"Error: {diag.get('error', 'unknown')}"]
    if diag.get("context_note"):
        summary_lines.append(f"Context: {diag['context_note']}")
    lines.append(_format_section("SUMMARY", summary_lines))

    tb_text = diag.get("traceback") or "(no traceback available)"
    lines.append(_format_section("TRACEBACK", [tb_text.rstrip("\n")]))

    system = diag.get("system") or {}
    lines.append(_format_section("SYSTEM", [f"{k}: {v}" for k, v in system.items()]))

    versions = diag.get("versions") or {}
    lines.append(_format_section("VERSIONS", [f"{k}: {v}" for k, v in versions.items()]))

    config = diag.get("config") or {}
    lines.append(_format_section("CONFIG", [f"{k}: {v}" for k, v in config.items()]))

    lines.append(_format_section("STREAM CONFIG", [_render_stream_config(diag.get("stream_config"))]))

    env = diag.get("env") or {}
    lines.append(_format_section("ENV", [f"{k}={v}" for k, v in env.items()]))

    lines.append(_format_section("LOG TAIL", diag.get("log_tail") or []))

    return "\n".join(lines)


def write_error_report(
    exc: BaseException,
    *,
    stage: str,
    context: Optional[Dict[str, Any]] = None,
    wrapper: Any = None,
    config: Optional[Dict[str, Any]] = None,
    out_dir: Optional[Any] = None,
) -> Optional[Path]:
    """
    Build and write a diagnostic report to disk.

    Best-effort -- any failure here is caught and logged rather than raised, so a
    reporting bug never masks the original inference error. Callers in a retry loop
    (e.g. the TD streaming loop) should debounce calls themselves -- this function
    writes unconditionally on every call.

    Args:
        exc: The caught exception.
        stage: Report stage label, e.g. "inference".
        context: Extra key/value pairs merged into CONFIG; "where" (if present)
            is also surfaced as the SUMMARY "Context:" line.
        wrapper: StreamDiffusionWrapper instance, for config + GPU state.
        config: Raw stream/pipeline config dict (e.g. td_manager's loaded YAML), dumped
            into == STREAM CONFIG == with secret-looking keys recursively redacted (see
            _redact_secrets). Complementary to `wrapper` -- the wrapper only exposes
            resolved runtime attrs, not the original input config.
        out_dir: Directory to write into. Defaults to $SDTD_BASE_FOLDER_PATH/error_reports,
            pinned at install time the same way as CUDALINK_*; falls back to
            <repo root>/error_reports (resolved from this module's own path, reliable
            under the editable install) when the env var isn't set.

    Returns:
        Path to the written report, or None if writing failed.
    """
    try:
        diag = collect_diagnostics(wrapper=wrapper, extra=context)
        diag["stage"] = stage
        diag["error"] = f"{type(exc).__name__}: {exc}"
        diag["context_note"] = (context or {}).get("where", stage)
        diag["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        diag["stream_config"] = _redact_secrets(config) if config else config
        diag["log_tail"] = _get_log_tail()

        if out_dir:
            target_dir = Path(out_dir)
        else:
            base_dir = os.environ.get("SDTD_BASE_FOLDER_PATH")
            target_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parents[3]
            target_dir = target_dir / "error_reports"
        target_dir.mkdir(parents=True, exist_ok=True)

        # Microsecond precision (not just seconds) so back-to-back reports in a tight failure
        # burst (e.g. a retried streaming-loop error) get distinct filenames instead of one
        # silently overwriting the last via write_text()'s truncate-on-open.
        timestamp = _utc_now().strftime("%Y%m%d_%H%M%S_%f")
        stage_slug = re.sub(r"[^A-Za-z0-9_-]", "_", stage) or "unknown"
        report_path = target_dir / f"{stage_slug}_error_report_{timestamp}.txt"
        report_path.write_text(format_report_text(diag), encoding="utf-8")
        return report_path
    except Exception as write_exc:
        try:
            logging.getLogger(__name__).error(f"Failed to write inference error report: {write_exc}")
        except Exception:
            pass
        return None


_install_log_tail_handler()
