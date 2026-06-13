"""Direct boundary diagnostic head."""

from __future__ import annotations

from models.mlp import build_mlp


def build_boundary_head(input_dim: int, hidden: int = 64, depth: int = 2):
    return build_mlp(input_dim, 1, hidden, depth)
