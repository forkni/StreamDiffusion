"""
Regression tests for the §4b fix: engine-progress OSC reporting was dead for
three independent reasons (OSCLoggingHandler sat at ERROR level, builder.py's
progress strings were print()-only, and the root logger was clobbered down to
WARNING). This covers the OSCLoggingHandler.emit half — that an INFO record
containing one of the sniffed substrings forwards to send_engine_progress, and
that an unrelated INFO record does not.

td_main.py is a script, not an importable library: module-level code that runs
immediately after these two classes are defined reads td_config.yaml from disk,
opens a real OSC UDP client, and installs a logging handler on the root logger
- side effects a unit test must not trigger, and there is no td_config.yaml
next to this test to satisfy the read anyway. The class bodies themselves have
no side effects at definition time (only at call time), so this test extracts
just the OSCReporter/OSCLoggingHandler source by AST line range and execs it in
an isolated namespace - exercising the real production source without paying
for (or being broken by the absence of) the rest of the module.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_TD_MAIN_PATH = Path(__file__).parent.parent.parent / "streamdiffusionTD" / "td_main.py"


def _load_osc_logging_handler_class():
    source = _TD_MAIN_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    classes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in ("OSCReporter", "OSCLoggingHandler")
    }
    assert set(classes) == {"OSCReporter", "OSCLoggingHandler"}, (
        "td_main.py's OSCReporter/OSCLoggingHandler class layout changed - update this test's extraction"
    )

    lines = source.splitlines()
    start = min(node.lineno for node in classes.values())
    end = max(node.end_lineno for node in classes.values())
    class_source = "\n".join(lines[start - 1 : end])

    namespace = {"logging": logging}
    exec(compile(class_source, str(_TD_MAIN_PATH), "exec"), namespace)
    return namespace["OSCLoggingHandler"]


@pytest.fixture(scope="module")
def osc_logging_handler_class():
    return _load_osc_logging_handler_class()


def _make_record(msg: str, level: int = logging.INFO) -> logging.LogRecord:
    return logging.LogRecord(
        name="streamdiffusion.acceleration.tensorrt.builder",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )


class TestOscLoggingHandlerEngineProgressForwarding:
    def test_handler_level_is_info_not_error(self, osc_logging_handler_class):
        """Regression guard: the handler used to sit at ERROR, which made the
        sniffing below unreachable for any INFO record regardless of emit()'s logic."""
        handler = osc_logging_handler_class(MagicMock())
        assert handler.level == logging.INFO

    def test_exporting_model_info_record_forwards_engine_progress(self, osc_logging_handler_class):
        reporter = MagicMock()
        handler = osc_logging_handler_class(reporter)

        handler.emit(_make_record("Exporting model: /path/to/model.onnx"))

        reporter.send_engine_progress.assert_called_once_with("exporting_onnx", "/path/to/model.onnx")
        reporter.send_error.assert_not_called()

    def test_unrelated_info_record_does_not_forward(self, osc_logging_handler_class):
        reporter = MagicMock()
        handler = osc_logging_handler_class(reporter)

        handler.emit(_make_record("_load_model: CUDA context reset completed"))

        reporter.send_engine_progress.assert_not_called()
        reporter.send_error.assert_not_called()
