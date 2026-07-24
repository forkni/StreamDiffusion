"""Test that the Scripts/ .tox mirror stays in sync with canonical sources.

Scripts/ is a sibling staging folder (`<repo-root>/../Scripts/`), not part of
this git repository — a bare clone (e.g. CI) won't have it, so these tests
skip rather than fail when it's absent.

The authoritative pair list lives in scripts/sync_td_mirror.PAIRS — keep this
docstring free of the listing so it doesn't drift.

To update any paired file: edit the canonical source under streamdiffusionTD/,
then run:

    python scripts/sync_td_mirror.py

Never edit the derived Scripts/ files directly — this test will reject the
drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent

sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
from sync_td_mirror import _TD_MIRROR_ROOT, PAIRS

pytestmark = pytest.mark.skipif(
    not _TD_MIRROR_ROOT.is_dir(),
    reason=f"{_TD_MIRROR_ROOT} not present (bare clone without the surrounding dev tree)",
)


@pytest.mark.parametrize("canonical,derived,mode", PAIRS, ids=[dst.name for _, dst, _ in PAIRS])
def test_td_mirror_file_is_identical(canonical: Path, derived: Path, mode: str) -> None:
    """Verify each paired Scripts/ mirror file is byte-identical to its canonical source."""
    assert canonical.exists(), f"Canonical source not found: {canonical}"
    assert derived.exists(), f"Derived mirror not found: {derived}\nRun: python scripts/sync_td_mirror.py"

    canonical_content = canonical.read_text(encoding="utf-8")
    derived_content = derived.read_text(encoding="utf-8")

    assert canonical_content == derived_content, (
        f"{derived.name} is out of sync with {canonical.name}!\n"
        f"  canonical: {canonical} ({len(canonical_content)} chars)\n"
        f"  derived:   {derived} ({len(derived_content)} chars)\n"
        "\nRun: python scripts/sync_td_mirror.py"
    )
