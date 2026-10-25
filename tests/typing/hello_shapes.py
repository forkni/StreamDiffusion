"""Pyrefly tensor-shape smoke check (type-checked only, never executed).

Proves that ``pyrefly-torch-stubs`` + ``pyrefly-shape-extensions`` resolve
through ``venv/Scripts/python.exe`` and that shape tracking is on. If
``venv/Scripts/pyrefly.exe check`` reports an error here, the stubs are not on
the search path (``pyrefly dump-config src/streamdiffusion/pipeline.py``) or
the three packages are out of lockstep. Not a pytest file on purpose: the
``Tensor[[2, 4]]`` form below is a static-only type and would fail at runtime.
"""

from __future__ import annotations

from typing import assert_type

import torch
from torch import Tensor


def matmul_tracks_shape() -> None:
    a = torch.zeros(2, 3)
    b = torch.zeros(3, 4)
    c = a @ b
    assert_type(c, Tensor[[2, 4]])


def unsqueeze_tracks_shape(x: Tensor[[8, 4, 64, 64]]) -> Tensor[[1, 8, 4, 64, 64]]:
    return x.unsqueeze(0)
