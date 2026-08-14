"""
Regression test for Root cause A: /prompt_list datagrams over 8192 bytes were
silently dropped.

python-osc's OSCUDPServer subclasses socketserver.UDPServer without overriding
max_packet_size, which defaults to 8192 bytes -- any larger UDP datagram is
truncated at recvfrom(), then Dispatcher.call_handlers_for_packet swallows the
resulting osc_packet.ParseError with a bare `pass`. Net effect: a prompt_list
with ~20+ weighted, sentence-length PromptTank prompts (routinely >8 KB once
JSON-encoded) vanished with no log line anywhere, and the stream kept
rendering the last value that fit under the cap (see
i-want-you-to-inherited-lerdorf.md, "Root cause A").

Fix: td_osc_handler.py's _LargePacketOSCUDPServer raises max_packet_size to
65536 and enlarges SO_RCVBUF to match. This test starts the REAL
OSCParameterHandler (the same class TD's backend runs) on a loopback port,
sends an oversized /prompt_list datagram through an actual UDP socket, and
asserts the update reaches the manager. It exercises the real
socketserver/pythonosc stack, not a mock -- it fails on the pre-fix
BlockingOSCUDPServer (datagram truncated, ParseError swallowed, manager never
called) and passes with the fix in place.

td_osc_handler.py lives in StreamDiffusionTD/ (gitignored, deployed-copy
layer -- see ADR-0002) rather than under src/, so it isn't on sys.path by
default. Unlike td_main.py, it has no import-time side effects (no config
read, no socket opened at module scope -- only inside methods), so it's
loaded here as a whole module via importlib rather than AST-extracted.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import socketserver
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pythonosc import udp_client

_OSC_HANDLER_PATH = Path(__file__).parent.parent.parent / "StreamDiffusionTD" / "td_osc_handler.py"


def _load_osc_handler_module():
    if not _OSC_HANDLER_PATH.exists():
        pytest.skip(f"{_OSC_HANDLER_PATH} not present (deployed StreamDiffusionTD/ layer is gitignored)")
    spec = importlib.util.spec_from_file_location("_td_osc_handler_under_test", _OSC_HANDLER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def osc_handler_module():
    return _load_osc_handler_module()


def _free_port() -> int:
    """Reserve an ephemeral TCP-free-looking UDP port by binding and releasing it.

    Small TOCTOU race in principle; acceptable for a test that owns the whole
    loopback interface and isn't run concurrently with itself.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def _make_oversized_prompt_list(n: int = 24, pad_words: int = 8):
    """Build a JSON-encodable [(text, weight), ...] list whose wire size lands
    comfortably above socketserver.UDPServer's 8192-byte default (and well
    under the 65536 ceiling the fix raises it to) -- calibrated to land around
    ~12 KB, matching the plan's realistic-PromptTank-prompt repro size."""
    weight = round(1.0 / n, 6)
    prompts = []
    for i in range(n):
        text = (
            f"cinematic portrait subject variant {i:03d}, "
            + ("intricate detailed volumetric lighting dramatic atmosphere " * pad_words)
            + f"tag-{i}"
        )
        prompts.append([text, weight])
    payload = json.dumps(prompts)
    return prompts, payload


def _wait_for_update_parameters_call(manager: MagicMock, timeout: float = 2.0):
    """Poll the ~60 Hz parameter batch loop (batch_interval=0.016s) until
    manager.update_parameters has been called, or time out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if manager.update_parameters.called:
            return
        time.sleep(0.01)
    pytest.fail(f"manager.update_parameters was not called within {timeout}s -- update never arrived")


class TestOscLargePromptListDelivery:
    def setup_method(self):
        self.manager = MagicMock()
        self.handler = None

    def teardown_method(self):
        if self.handler is not None:
            self.handler.stop()

    def _start_handler(self, osc_handler_module):
        listen_port = _free_port()
        transmit_port = _free_port()
        self.handler = osc_handler_module.OSCParameterHandler(
            self.manager,
            listen_port=listen_port,
            transmit_port=transmit_port,
            transmit_ip="127.0.0.1",
        )
        self.handler.start()
        return listen_port

    def test_prompt_list_over_8192_bytes_is_delivered(self, osc_handler_module):
        prompts, payload = _make_oversized_prompt_list()
        payload_bytes = len(payload.encode("utf-8"))
        assert payload_bytes > socketserver.UDPServer.max_packet_size, (
            "fixture payload must exceed the stdlib UDP default cap for this test to be meaningful "
            f"(got {payload_bytes} bytes, cap is {socketserver.UDPServer.max_packet_size})"
        )
        assert payload_bytes < 65536, "fixture payload must stay under the raised ceiling"

        listen_port = self._start_handler(osc_handler_module)
        client = udp_client.SimpleUDPClient("127.0.0.1", listen_port)
        client.send_message("/prompt_list", payload)

        _wait_for_update_parameters_call(self.manager)

        delivered_batch = self.manager.update_parameters.call_args[0][0]
        assert "prompt_list" in delivered_batch, (
            "prompt_list never reached the manager -- the oversized datagram was dropped "
            "(pre-fix: truncated at recvfrom(), then silently swallowed as an OSC ParseError)"
        )
        delivered = delivered_batch["prompt_list"]
        assert len(delivered) == len(prompts), (
            f"delivered prompt_list has {len(delivered)} entries, expected {len(prompts)} -- "
            "datagram was likely truncated rather than cleanly dropped"
        )
        assert [text for text, _weight in delivered] == [text for text, _weight in prompts]

    def test_small_prompt_list_still_delivered(self, osc_handler_module):
        """Sanity control: ordinary small payloads must keep working unchanged."""
        prompts = [["cat", 0.6], ["dog", 0.4]]
        payload = json.dumps(prompts)
        assert len(payload.encode("utf-8")) < socketserver.UDPServer.max_packet_size

        listen_port = self._start_handler(osc_handler_module)
        client = udp_client.SimpleUDPClient("127.0.0.1", listen_port)
        client.send_message("/prompt_list", payload)

        _wait_for_update_parameters_call(self.manager)

        delivered_batch = self.manager.update_parameters.call_args[0][0]
        assert delivered_batch["prompt_list"] == [("cat", 0.6), ("dog", 0.4)]
