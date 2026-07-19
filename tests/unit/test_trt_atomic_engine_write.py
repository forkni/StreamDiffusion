"""
Tests for the atomic engine/timing-cache write helper (Phase 4, item 1).

`_atomic_write_bytes` in `acceleration/tensorrt/utilities.py` replaces the plain
`open(path, "wb")` writes used for TRT engine files and timing caches. A crash or
interrupt mid-write must never leave a truncated file at the final path — the
builder's cache check (`builder.py`) is a bare `os.path.exists`, so a truncated
engine would otherwise be silently treated as valid on the next run.

These tests exercise only the pure-Python temp-file + os.replace logic; no TRT
context or GPU is required. The module is skipped if `utilities.py` cannot be
imported (e.g. TensorRT/onnx/polygraphy not installed).

Run with: pytest tests/unit/test_trt_atomic_engine_write.py -v
"""

import pytest

# ---------------------------------------------------------------------------
# Import guard — skip all tests if utilities.py's dependencies are unavailable
# ---------------------------------------------------------------------------

try:
    from streamdiffusion.acceleration.tensorrt import utilities as trt_utilities

    IMPORT_OK = True
except ImportError:
    IMPORT_OK = False

pytestmark = pytest.mark.skipif(
    not IMPORT_OK,
    reason="acceleration.tensorrt.utilities not importable (TensorRT/onnx/polygraphy missing)",
)


class TestAtomicWriteBytes:
    def test_atomic_write_creates_file_with_content(self, tmp_path):
        """Happy path: final file has exact content, no stray .tmp remains."""
        target = tmp_path / "engine.trt"
        payload = b"\x00\x01fake-engine-bytes\xff" * 100

        trt_utilities._atomic_write_bytes(str(target), payload)

        assert target.read_bytes() == payload
        assert not (tmp_path / "engine.trt.tmp").exists()

    def test_failed_write_preserves_existing_file(self, tmp_path, monkeypatch):
        """A failed rebuild must not corrupt a previously-good cached engine."""
        target = tmp_path / "engine.trt"
        good_bytes = b"known-good-engine-bytes"
        target.write_bytes(good_bytes)

        def _boom(*_args, **_kwargs):
            raise OSError("simulated interrupt during rename")

        monkeypatch.setattr(trt_utilities.os, "replace", _boom)

        with pytest.raises(OSError, match="simulated interrupt"):
            trt_utilities._atomic_write_bytes(str(target), b"new-but-never-committed-bytes")

        # Pre-existing file untouched — os.replace never ran.
        assert target.read_bytes() == good_bytes
        # Temp file cleaned up, not left dangling next to the real path.
        assert not (tmp_path / "engine.trt.tmp").exists()

    def test_failed_write_leaves_no_partial_final(self, tmp_path, monkeypatch):
        """No pre-existing file: a failed write must not leave a truncated final file."""
        target = tmp_path / "engine.trt"

        def _boom(*_args, **_kwargs):
            raise OSError("simulated interrupt during rename")

        monkeypatch.setattr(trt_utilities.os, "replace", _boom)

        with pytest.raises(OSError, match="simulated interrupt"):
            trt_utilities._atomic_write_bytes(str(target), b"partial-write-bytes")

        assert not target.exists()
        assert not (tmp_path / "engine.trt.tmp").exists()
