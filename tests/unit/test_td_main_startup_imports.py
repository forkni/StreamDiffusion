"""Regression test for the td_main.py startup-import ordering invariant.

`OSCReporter`'s own docstring states the invariant: "Lives at module level - starts
before any heavy imports. No dependencies on torch, diffusers, or StreamDiffusion."
This was broken once: `from streamdiffusion.utils.diagnostics import ...` was added
eagerly at module scope for error reporting. `diagnostics.py` is itself pure stdlib,
but it is a *submodule* of the `streamdiffusion` package, so importing it runs
`streamdiffusion/__init__.py`, which eagerly imports the wrapper and preprocessor
registry (torch, diffusers, timm, controlnet_aux -- measured 22.03s / 5267
`sys.modules` entries vs. a 0.16s / 192-module baseline without it). That pushed
`osc_reporter.start_heartbeat()` ~22s later, leaving TouchDesigner with no
`/server_active` heartbeat and no state message for the entire heavy-import window.
The fix resolves those diagnostics helpers lazily on first use instead (see
`_resolve_diagnostics()` in td_main.py) so nothing heavy loads before the heartbeat.

This test parses td_main.py with `ast` and asserts no import *executed at
module-import time* above the `osc_reporter.start_heartbeat()` call names a module
from the heavy set -- so this exact regression goes red immediately if reintroduced.

Traversal detail (easy to get wrong -- an earlier draft of this test got it wrong):
a heavy import legitimately exists *after* the anchor (`td_manager` / `td_osc_handler`,
imported once the heartbeat has already started) and inside function bodies (e.g. a
deferred `import torch` used only when actually needed, never at import time). Only
`try`/`if`/`with`/`for`/`while` bodies execute at real module-import time --
`FunctionDef`/`AsyncFunctionDef`/`ClassDef` bodies do not, and must not be descended
into, or the walk produces false positives on legitimate deferred imports.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_TD_MAIN = _PROJECT_ROOT / "StreamDiffusionTD" / "td_main.py"

# Anything in this set pulls in (or is) the heavy ML stack -- see the module docstring
# above for the measured cost. `streamdiffusion` covers every submodule import too,
# since importing any of them runs the same package __init__.
_HEAVY_MODULES = {
    "streamdiffusion",
    "torch",
    "diffusers",
    "transformers",
    "timm",
    "controlnet_aux",
    "td_manager",
    "td_osc_handler",
}


def _module_scope(nodes):
    """Yield every node reachable at *module-import time*: descend through control
    flow (`try`/`if`/`with`/`for`/`while`, since those bodies run as the module
    executes top to bottom) but never into `FunctionDef`/`AsyncFunctionDef`/
    `ClassDef`, since those are deferred until called or instantiated. A plain
    `ast.walk` gets this wrong -- it happily descends into function/class bodies and
    flags deferred imports that never run at import time.
    """
    for node in nodes:
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield from _module_scope(ast.iter_child_nodes(node))


def _heavy_import_root(node: ast.stmt) -> str | None:
    """Return the top-level module name if `node` is an Import/ImportFrom naming a
    heavy module (per `_HEAVY_MODULES`), else None."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in _HEAVY_MODULES:
                return root
        return None
    if isinstance(node, ast.ImportFrom):
        if node.module is None:  # relative `from . import x` -- never a heavy-module case here
            return None
        root = node.module.split(".")[0]
        return root if root in _HEAVY_MODULES else None
    return None


def _find_heartbeat_anchor(tree: ast.Module) -> int:
    """Locate the unique `osc_reporter.start_heartbeat()` call -- the line by which
    the `OSCReporter` docstring's invariant requires every heavy import to sit after."""
    matches = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "start_heartbeat"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "osc_reporter"
    ]
    assert len(matches) == 1, (
        f"Expected exactly one osc_reporter.start_heartbeat() call in td_main.py, "
        f"found {len(matches)} at lines {matches}. This test's anchor assumption no "
        "longer holds -- update it to match the current source."
    )
    return matches[0]


def _module_scope_imports(tree: ast.Module) -> list[tuple[int, str]]:
    """All (line, root_module) pairs for heavy imports reachable at module-import
    time, per `_module_scope`'s traversal rules."""
    return [
        (node.lineno, root)
        for node in _module_scope(tree.body)
        if isinstance(node, (ast.Import, ast.ImportFrom)) and (root := _heavy_import_root(node)) is not None
    ]


def _parse_td_main() -> ast.Module:
    if not _TD_MAIN.is_file():
        pytest.skip(f"td_main.py not found at {_TD_MAIN} -- gitignored dev-only module")
    return ast.parse(_TD_MAIN.read_text(encoding="utf-8"), filename=str(_TD_MAIN))


def test_no_heavy_import_before_heartbeat_start() -> None:
    """The real invariant: nothing in `_HEAVY_MODULES` may be imported at
    module-import time above the line that starts the OSC heartbeat."""
    tree = _parse_td_main()
    anchor_line = _find_heartbeat_anchor(tree)

    violations = [(line, root) for line, root in _module_scope_imports(tree) if line < anchor_line]

    assert violations == [], (
        f"Heavy module(s) imported at module-import time before "
        f"osc_reporter.start_heartbeat() (line {anchor_line}): {violations}. "
        "This reintroduces the ~22s OSC-heartbeat blackout the lazy-diagnostics fix "
        "removed -- resolve the import lazily instead (see _resolve_diagnostics())."
    )


def test_legitimate_heavy_imports_after_heartbeat_are_not_flagged() -> None:
    """Proves the checker discriminates on *position*, not module identity:
    `td_manager` / `td_osc_handler` are genuinely heavy (they import `streamdiffusion`
    themselves) and are imported at true module scope (inside a top-level `try`) in
    td_main.py -- but only after the heartbeat has already started, which is correct
    and must never be flagged as a violation."""
    tree = _parse_td_main()
    anchor_line = _find_heartbeat_anchor(tree)

    after_anchor_roots = {root for line, root in _module_scope_imports(tree) if line > anchor_line}

    assert "td_manager" in after_anchor_roots, (
        "Expected to find the legitimate td_manager import at module scope after the "
        f"heartbeat anchor (line {anchor_line}), but it wasn't detected. Either the "
        "traversal is over-restrictive (a real regression in this test) or td_manager's "
        "import was moved/removed (update this test's expectation)."
    )
    assert "td_osc_handler" in after_anchor_roots, (
        "Expected to find the legitimate td_osc_handler import at module scope after "
        f"the heartbeat anchor (line {anchor_line}), but it wasn't detected."
    )
