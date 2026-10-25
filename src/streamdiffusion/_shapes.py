"""Symbolic tensor dimensions for pyrefly's static shape checking.

Only annotations use these names (``Tensor[[B, 4, Hl, Wl]]``); they are never
part of a runtime expression. Under the type checker they are
``shape_extensions.IntVar`` symbols, so a function annotated with ``B`` on two
parameters demands the same batch size on both. At runtime they are plain
``int``: ``shape_extensions`` is a development dependency only and is never
imported outside the ``TYPE_CHECKING`` block (TouchDesigner's embedded Python
has no such package). Modules that use these names must start with
``from __future__ import annotations`` so the annotations stay unevaluated.

Conventions:

- ``B`` batch, ``C`` channels, ``H`` / ``W`` image height / width (pixels),
- ``Hl`` / ``Wl`` latent height / width (``H // 8`` / ``W // 8`` for the SD VAE),
- ``N`` a generic row count (denoising steps, prompt tokens, ...).

See ``tests/typing/README.md`` for the toolchain and the annotation rules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shape_extensions import Elements, Int, IntVar  # noqa: F401

    B = IntVar("B")
    C = IntVar("C")
    H = IntVar("H")
    W = IntVar("W")
    Hl = IntVar("Hl")
    Wl = IntVar("Wl")
    N = IntVar("N")
else:  # never reached by the checker; keeps the names importable at runtime
    B = C = H = W = Hl = Wl = N = int

__all__ = ["B", "C", "H", "W", "Hl", "Wl", "N"]
