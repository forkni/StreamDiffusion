"""
gpu_profiler.py — Portable GPU profiling module for CUDA/PyTorch projects.

PORTABILITY: Copy this single file into any project.  Only stdlib is required
when profiling is disabled.  PyTorch is imported lazily when enabled.

USAGE:
    from streamdiffusion.tools.gpu_profiler import profiler, configure

    configure(enabled=True, nvtx=True, events=True)

    with profiler.region("inference"):
        output = model(input)

    profiler.report()

CUDA GRAPH COMPATIBILITY:
    NVTX push/pop calls break CUDA graph replay — any push/pop recorded during
    graph capture fires only once (at capture time), not on each replay step.
    When CUDA graphs are active set  nvtx=False (or GPU_PROFILER_NVTX=0) so
    only CUDA-event timing is collected — events are always graph-safe.

    For StreamDiffusion's TRT engine path: set GPU_PROFILER_NVTX=0 to use
    events-only mode during graph-replayed inference, and GPU_PROFILER_NVTX=1
    for a non-graph capture run (STREAMDIFFUSION_PROFILE_TRT=1 disables graphs
    when you need per-layer IProfiler timing instead).

NSIGHT SYSTEMS COMMAND:
    nsys profile --trace=cuda,nvtx,cublas,cudnn --cuda-memory-usage=true \\
        -o profiles/sdtd_out --force-overwrite true \\
        .venv/Scripts/python scripts/profiling/profile_nsys.py --target benchmark
"""

from __future__ import annotations

import json
import os
import pickle
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Generator, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# RegionStats — per-region histogram with percentile support
# ─────────────────────────────────────────────────────────────────────────────


class RegionStats:
    """Histogram-based timing statistics for a named profiling region."""

    __slots__ = ("name", "samples", "count", "total_ms")

    MAX_SAMPLES = 10_000  # cap to avoid unbounded memory

    def __init__(self, name: str) -> None:
        self.name = name
        self.samples: List[float] = []
        self.count: int = 0
        self.total_ms: float = 0.0

    def record(self, ms: float) -> None:
        self.count += 1
        self.total_ms += ms
        if len(self.samples) < self.MAX_SAMPLES:
            self.samples.append(ms)

    @property
    def mean(self) -> float:
        return self.total_ms / self.count if self.count else 0.0

    @property
    def p50(self) -> float:
        return self._percentile(50)

    @property
    def p95(self) -> float:
        return self._percentile(95)

    @property
    def p99(self) -> float:
        return self._percentile(99)

    @property
    def min(self) -> float:
        return min(self.samples) if self.samples else 0.0

    @property
    def max(self) -> float:
        return max(self.samples) if self.samples else 0.0

    def _percentile(self, p: int) -> float:
        if not self.samples:
            return 0.0
        s = sorted(self.samples)
        idx = int(len(s) * p / 100)
        return s[min(idx, len(s) - 1)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "mean_ms": round(self.mean, 3),
            "p50_ms": round(self.p50, 3),
            "p95_ms": round(self.p95, 3),
            "p99_ms": round(self.p99, 3),
            "min_ms": round(self.min, 3),
            "max_ms": round(self.max, 3),
            "total_ms": round(self.total_ms, 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
# _RegionCtx — context manager for a single profiled region
# ─────────────────────────────────────────────────────────────────────────────


class _RegionCtx:
    """Context manager that records one entry for a named region.

    On entry: optional NVTX range_push + CUDA event start record.
    On exit:  optional NVTX range_pop + CUDA event elapsed_time -> RegionStats.
    """

    __slots__ = ("_profiler", "_name", "_nvtx", "_start_evt", "_end_evt")

    def __init__(self, profiler: "GPUProfiler", name: str) -> None:
        self._profiler = profiler
        self._name = name
        self._nvtx = profiler._nvtx_enabled
        self._start_evt = None
        self._end_evt = None

    def __enter__(self) -> "_RegionCtx":
        p = self._profiler
        if self._nvtx:
            p._torch.cuda.nvtx.range_push(self._name)
        if p._events_enabled:
            self._start_evt = p._torch.cuda.Event(enable_timing=True)
            self._end_evt = p._torch.cuda.Event(enable_timing=True)
            self._start_evt.record()
        return self

    def __exit__(self, *_: object) -> None:
        p = self._profiler
        if p._events_enabled and self._start_evt is not None:
            self._end_evt.record()
            # Synchronize lazily — elapsed_time blocks only when read.
            # We defer the sync to avoid stalling the GPU here.
            p._pending.append((self._name, self._start_evt, self._end_evt))
        if self._nvtx:
            p._torch.cuda.nvtx.range_pop()


class _NullCtx:
    """Zero-overhead context manager used when profiler is disabled."""

    __slots__ = ()

    def __enter__(self) -> "_NullCtx":
        return self

    def __exit__(self, *_: object) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# GPUProfiler — the real profiler (only instantiated when enabled)
# ─────────────────────────────────────────────────────────────────────────────


class GPUProfiler:
    """
    Unified GPU profiling singleton.

    Activate via module-level ``configure(enabled=True, ...)``.
    All methods are safe to call from any thread/process.
    """

    def __init__(self) -> None:
        self._nvtx_enabled: bool = False
        self._events_enabled: bool = False
        self._memory_enabled: bool = False
        self._trace_path: Optional[str] = None

        self._regions: Dict[str, RegionStats] = {}
        self._pending: List[tuple] = []  # (name, start_evt, end_evt) awaiting sync

        self._torch_profiler = None  # active torch.profiler.profile instance
        self._profiler_step: int = 0

        self._torch = None  # lazy torch reference
        self._cudart = None  # lazy cudart reference

    def configure(
        self,
        enabled: bool = True,
        nvtx: bool = True,
        events: bool = True,
        memory: bool = False,
        trace_path: Optional[str] = None,
    ) -> None:
        """Configure the profiler.  Must be called once before any region()."""
        import torch as _torch

        self._torch = _torch
        self._nvtx_enabled = nvtx and _torch.cuda.is_available()
        self._events_enabled = events and _torch.cuda.is_available()
        self._memory_enabled = memory
        self._trace_path = trace_path

    # ── Core API ─────────────────────────────────────────────────────────────

    def region(self, name: str) -> _RegionCtx:
        """Return a context manager that profiles one execution of ``name``."""
        if name not in self._regions:
            self._regions[name] = RegionStats(name)
        return _RegionCtx(self, name)

    def trace(self, name: str) -> Callable:
        """Decorator that wraps a function in a named profiler region.

        Usage::

            @profiler.trace("cupy_rgba_to_rgb")
            def my_kernel(src, dst): ...
        """

        def decorator(fn: Callable) -> Callable:
            @wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.region(name):
                    return fn(*args, **kwargs)

            return wrapper

        return decorator

    def mark(self, name: str) -> None:
        """Place an NVTX instant marker (zero-duration annotation)."""
        if self._nvtx_enabled and self._torch is not None:
            self._torch.cuda.nvtx.range_push(name)
            self._torch.cuda.nvtx.range_pop()

    def begin(self, name: str) -> None:
        """Open a named NVTX range without a context manager (pair with end())."""
        if name not in self._regions:
            self._regions[name] = RegionStats(name)
        if self._nvtx_enabled and self._torch is not None:
            self._torch.cuda.nvtx.range_push(name)

    def end(self, name: str) -> None:
        """Close a previously opened named NVTX range."""
        if self._nvtx_enabled and self._torch is not None:
            self._torch.cuda.nvtx.range_pop()

    # ── Nsight Systems gated capture ─────────────────────────────────────────

    def nsys_start(self) -> None:
        """Signal Nsight Systems to begin capture (cudaProfilerStart).

        Run your script under nsys: ``nsys profile --trace=cuda,nvtx ...``
        Capture only starts when this is called — useful to skip warmup.
        """
        if self._torch is not None and self._torch.cuda.is_available():
            try:
                self._torch.cuda.cudart().cudaProfilerStart()
            except Exception:  # broad: CUDA profiler API may not be available (no nsys, profiling disabled)
                pass

    def nsys_stop(self) -> None:
        """Signal Nsight Systems to stop capture (cudaProfilerStop)."""
        if self._torch is not None and self._torch.cuda.is_available():
            try:
                self._torch.cuda.cudart().cudaProfilerStop()
            except Exception:  # broad: CUDA profiler API may not be available (no nsys, profiling disabled)
                pass

    # ── torch.profiler integration ────────────────────────────────────────────

    @contextmanager
    def torch_trace(
        self,
        path: Optional[str] = None,
        warmup: int = 1,
        active: int = 5,
    ) -> Generator[None, None, None]:
        """Context manager wrapping torch.profiler.profile.

        Schedule: wait=0, warmup=``warmup``, active=``active``.
        Exports Chrome trace to ``path`` (or self._trace_path if not specified).
        Also prints top-30 ops by CUDA time to stdout.

        Usage::

            with profiler.torch_trace("trace.json", warmup=1, active=5):
                for i in range(warmup + active):
                    profiler.step()
                    run_inference()
        """
        out_path = path or self._trace_path or "gpu_profile_trace.json"
        if self._torch is None:
            yield
            return

        torch_profiler_mod = self._torch.profiler

        def _on_trace_ready(prof: Any) -> None:
            prof.export_chrome_trace(out_path)
            print(f"\n[gpu_profiler] Chrome trace -> {out_path}")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

        with torch_profiler_mod.profile(
            activities=[
                torch_profiler_mod.ProfilerActivity.CPU,
                torch_profiler_mod.ProfilerActivity.CUDA,
            ],
            schedule=torch_profiler_mod.schedule(wait=0, warmup=warmup, active=active),
            on_trace_ready=_on_trace_ready,
            record_shapes=True,
            with_stack=True,
        ) as prof:
            self._torch_profiler = prof
            try:
                yield
            finally:
                self._torch_profiler = None

    def step(self) -> None:
        """Advance the torch.profiler schedule by one step.

        Call once per iteration inside a ``torch_trace`` context.
        """
        if self._torch_profiler is not None:
            self._torch_profiler.step()

    # ── Memory profiling ──────────────────────────────────────────────────────

    @contextmanager
    def memory_trace(self, path: str = "mem_snapshot.pkl") -> Generator[None, None, None]:
        """Context manager that captures a VRAM allocation snapshot.

        The resulting ``.pkl`` file can be converted to interactive HTML via::

            python -c "
            import pickle, torch
            with open('mem_snapshot.pkl','rb') as f:
                snap = pickle.load(f)
            html = torch.cuda._memory_viz.trace_plot(snap)
            open('memory.html','w').write(html)
            "
        """
        if self._torch is None or not self._torch.cuda.is_available():
            yield
            return

        self._torch.cuda.synchronize()
        self._torch.cuda.memory._record_memory_history(
            True,
            trace_alloc_max_entries=100_000,
            trace_alloc_record_context=True,
        )
        try:
            yield
        finally:
            self._torch.cuda.synchronize()
            snapshot = self._torch.cuda.memory._snapshot()
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "wb") as fh:
                pickle.dump(snapshot, fh)
            print(f"[gpu_profiler] Memory snapshot -> {path}")
            try:
                self._torch.cuda.memory._record_memory_history(False)
            except Exception:  # broad: may fail if history was never started, or on older PyTorch without the API
                pass

    # ── Statistics ────────────────────────────────────────────────────────────

    def flush(self) -> None:
        """Resolve all pending CUDA event timings (forces GPU sync).

        Called automatically by report() and export_stats().  You can also
        call it manually after a batch of iterations to get up-to-date stats
        without printing.
        """
        if not self._pending:
            return
        if self._torch is not None:
            self._torch.cuda.synchronize()
        for name, start_evt, end_evt in self._pending:
            try:
                ms = start_evt.elapsed_time(end_evt)
                self._regions[name].record(ms)
            except Exception:  # broad: CUDA event timing fails if event was never recorded (e.g., boundary skipped)
                pass
        self._pending.clear()

    def report(self, top_n: int = 30) -> None:
        """Print a summary table sorted by total CUDA time.

        Flushes pending events first.
        """
        self.flush()
        if not self._regions:
            print("[gpu_profiler] No regions recorded.")
            return

        rows = sorted(
            self._regions.values(),
            key=lambda s: s.total_ms,
            reverse=True,
        )[:top_n]

        col_w = max(len(r.name) for r in rows) + 2
        header = (
            f"{'Region':<{col_w}} {'Count':>6}  "
            f"{'Mean':>8}  {'P50':>8}  {'P95':>8}  {'P99':>8}  "
            f"{'Min':>8}  {'Max':>8}  {'Total':>10}"
        )
        sep = "-" * len(header)
        print(f"\n[gpu_profiler] Timing Report (top {top_n} by total ms)")
        print(sep)
        print(header)
        print(sep)
        for r in rows:
            print(
                f"{r.name:<{col_w}} {r.count:>6}  "
                f"{r.mean:>7.2f}ms  {r.p50:>7.2f}ms  "
                f"{r.p95:>7.2f}ms  {r.p99:>7.2f}ms  "
                f"{r.min:>7.2f}ms  {r.max:>7.2f}ms  "
                f"{r.total_ms:>9.1f}ms"
            )
        print(sep)

    def export_stats(self, path: str = "gpu_profile_stats.json") -> None:
        """Write region statistics to a JSON file.

        Flushes pending events first.
        """
        self.flush()
        data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "regions": [s.to_dict() for s in self._regions.values()],
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
        print(f"[gpu_profiler] Stats -> {path}")

    def reset(self) -> None:
        """Clear all accumulated timing data and pending events."""
        self._regions.clear()
        self._pending.clear()


# ─────────────────────────────────────────────────────────────────────────────
# _NullProfiler — all methods are no-ops, zero imports, zero overhead
# ─────────────────────────────────────────────────────────────────────────────

_NULL_CTX = _NullCtx()


class _NullProfiler:
    """Drop-in replacement for GPUProfiler when profiling is disabled.

    Every method is a no-op.  region() returns a shared _NullCtx singleton
    that has bare __enter__/__exit__ bodies — no attribute lookups, no CUDA
    calls, no allocation.
    """

    __slots__ = ()

    def region(self, name: str) -> _NullCtx:  # noqa: ARG002
        return _NULL_CTX

    def trace(self, name: str) -> Callable:  # noqa: ARG002
        """Return identity decorator — function is NOT wrapped."""

        def decorator(fn: Callable) -> Callable:
            return fn

        return decorator

    def mark(self, name: str) -> None:
        pass  # noqa: E704

    def begin(self, name: str) -> None:
        pass  # noqa: E704

    def end(self, name: str) -> None:
        pass  # noqa: E704

    def nsys_start(self) -> None:
        pass  # noqa: E704

    def nsys_stop(self) -> None:
        pass  # noqa: E704

    def step(self) -> None:
        pass  # noqa: E704

    def flush(self) -> None:
        pass  # noqa: E704

    def report(self, top_n: int = 30) -> None:
        pass  # noqa: E704, ARG002

    def reset(self) -> None:
        pass  # noqa: E704

    @contextmanager
    def torch_trace(self, path: Optional[str] = None, warmup: int = 1, active: int = 5) -> Generator[None, None, None]:  # noqa: ARG002
        yield

    @contextmanager
    def memory_trace(self, path: str = "mem_snapshot.pkl") -> Generator[None, None, None]:  # noqa: ARG002
        yield

    def export_stats(self, path: str = "gpu_profile_stats.json") -> None:  # noqa: ARG002
        pass

    def configure(self, **kwargs: Any) -> None:
        pass  # noqa: E704, ARG002


# ─────────────────────────────────────────────────────────────────────────────
# _ProfilerProxy — stable singleton; configure() mutates the delegate in-place
# ─────────────────────────────────────────────────────────────────────────────


class _ProfilerProxy:
    """Stable proxy that delegates every call to the active inner profiler.

    Importing ``from streamdiffusion.tools.gpu_profiler import profiler`` is
    safe to do once at module load time.  ``configure()`` updates the inner
    delegate in-place, so stale import references keep working correctly
    without requiring a re-import after configure.

    Pattern:  profiler._set_inner(new_instance)  →  all future calls forwarded
    """

    __slots__ = ("_inner",)

    def __init__(self) -> None:
        object.__setattr__(self, "_inner", _NullProfiler())

    def _set_inner(self, inner: Any) -> None:
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_inner"), name)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton + configure()
# ─────────────────────────────────────────────────────────────────────────────

# Stable proxy — safe to import once.  configure() swaps the inner delegate.
profiler: Any = _ProfilerProxy()


def configure(
    enabled: bool = False,
    nvtx: bool = True,
    events: bool = True,
    memory: bool = False,
    trace_path: Optional[str] = None,
) -> None:
    """Configure the module-level profiler singleton.

    Args:
        enabled:    Master switch.  When False, all profiler calls are no-ops.
        nvtx:       Emit NVTX ranges (visible in Nsight Systems timeline).
                    Disable when CUDA graphs are active (GPU_PROFILER_NVTX=0)
                    to avoid incorrect timeline positions on graph replay.
        events:     Collect CUDA-event timing into RegionStats histograms.
                    Always safe with CUDA graphs.
        memory:     Enable torch.cuda.memory._record_memory_history when
                    memory_trace() context is entered.
        trace_path: Default Chrome trace output path for torch_trace().

    Also reads environment variables (take priority over config when GPU_PROFILER=1):
        GPU_PROFILER=1          → enabled=True; resets nvtx/events defaults to True
        GPU_PROFILER_NVTX=0     → nvtx=False   (only when GPU_PROFILER=1)
        GPU_PROFILER_EVENTS=0   → events=False  (only when GPU_PROFILER=1)
    """
    _env_enabled = os.environ.get("GPU_PROFILER", "0") == "1"
    enabled = enabled or _env_enabled
    if _env_enabled:
        # Env activation: use env vars as the sole source for nvtx/events so
        # config "events: false" can't silently suppress CUDA-event collection.
        nvtx = os.environ.get("GPU_PROFILER_NVTX", "1") != "0"
        events = os.environ.get("GPU_PROFILER_EVENTS", "1") != "0"
    else:
        nvtx = nvtx and os.environ.get("GPU_PROFILER_NVTX", "1") != "0"
        events = events and os.environ.get("GPU_PROFILER_EVENTS", "1") != "0"

    if not enabled:
        profiler._set_inner(_NullProfiler())
        return

    p = GPUProfiler()
    p.configure(enabled=enabled, nvtx=nvtx, events=events, memory=memory, trace_path=trace_path)
    profiler._set_inner(p)


def configure_from_dict(cfg: Dict[str, Any]) -> None:
    """Convenience: read profiling settings from a config sub-dict.

    Expected keys (all optional)::

        {
            "profiling": {
                "enabled": false,
                "nvtx": true,
                "events": true,
                "memory": false,
                "trace_path": "profiler_logs/trace.json"
            }
        }
    """
    prof_cfg = cfg.get("profiling", {})
    configure(
        enabled=prof_cfg.get("enabled", False),
        nvtx=prof_cfg.get("nvtx", True),
        events=prof_cfg.get("events", True),
        memory=prof_cfg.get("memory", False),
        trace_path=prof_cfg.get("trace_path", None),
    )
