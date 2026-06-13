"""Bounded supervised Delta diagnostic head."""

from __future__ import annotations

from models.mlp import build_mlp


class BoundedDeltaHead:
    @staticmethod
    def build(input_dim: int, hidden: int = 64, depth: int = 2):
        import torch.nn as nn

        return nn.Sequential(build_mlp(input_dim, 1, hidden, depth), nn.Tanh())


def map_delta(raw_unit, is_call):
    """Map tanh output in [-1,1] to call [0,1] or put [-1,0]."""
    return 0.5 * (raw_unit + 1.0) if is_call else 0.5 * (raw_unit - 1.0)
