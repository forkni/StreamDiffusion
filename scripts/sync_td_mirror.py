"""
sync_td_mirror.py — Keep the Scripts/ .tox mirror in sync with canonical sources.

THIS IS THE ONLY LEGITIMATE WAY TO UPDATE THE PAIRED Scripts/ FILES.
Do not edit the derived Scripts/ files by hand — edits will be overwritten the
next time this script runs, and `--check` will reject a hand-edited derived
file that has drifted from its canonical source. Edit the canonical file under
streamdiffusionTD/, then run this script.

Scripts/ is NOT part of this git repository — it is a sibling staging folder
(`<repo-root>/../Scripts/`) baked into the shipped `.tox` component. A bare
clone of this repo (e.g. on CI) will not have it; both the sync and the check
skip cleanly in that case rather than failing (see ADR-0002 in
cuda-link/docs/adr for the pattern this ports).

Only byte_identical mode is used today: the paired Scripts/ files use bare
top-level imports already, with no relative imports to rewrite. The `mode`
field is kept in PAIRS regardless, so adding a rewrite-style pair later
doesn't require reshaping the table.

Usage:
    python scripts/sync_td_mirror.py           # copy canonical -> Scripts/ (update)
    python scripts/sync_td_mirror.py --check   # verify only; exit 1 if any pair differs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parent.parent

# Scripts/ is a sibling of the repo clone, not inside it (see module docstring).
_TD_MIRROR_ROOT = REPO_ROOT.parent / "Scripts"

# ---------------------------------------------------------------------------
# Authoritative pair list: (canonical, derived, mode).
# To add a new mirrored module: add it here.
# ---------------------------------------------------------------------------

PAIRS: list[tuple[Path, Path, Literal["byte_identical"]]] = [
    (
        REPO_ROOT / "streamdiffusionTD" / "td_main.py",
        _TD_MIRROR_ROOT / "streamdiffusionTD__Text__td_main__td.py",
        "byte_identical",
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync Scripts/ .tox mirror from canonical sources.")
    parser.add_argument("--check", action="store_true", help="Check only; exit 1 if any pair differs.")
    args = parser.parse_args()

    if not _TD_MIRROR_ROOT.is_dir():
        print(f"SKIP: {_TD_MIRROR_ROOT} not present (bare clone without the surrounding dev tree) — nothing to do.")
        return 0

    exit_code = 0

    for src, dst, mode in PAIRS:
        if not src.exists():
            print(f"ERROR: canonical source not found: {src}", file=sys.stderr)
            return 1

        src_text = src.read_text(encoding="utf-8")

        if args.check:
            if not dst.exists():
                print(f"FAIL [{mode}]: {dst} does not exist.", file=sys.stderr)
                exit_code = 1
                continue
            on_disk = dst.read_text(encoding="utf-8")
            if on_disk == src_text:
                print(f"OK [{mode}]: {dst.name}")
            else:
                print(
                    f"FAIL [{mode}]: {dst.name} is out of sync with {src.name}. Run: python scripts/sync_td_mirror.py",
                    file=sys.stderr,
                )
                exit_code = 1
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(src_text, encoding="utf-8")
            print(f"Synced [{mode}] {src.relative_to(REPO_ROOT)} -> {dst}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
