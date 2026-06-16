"""Bounded supervised Delta diagnostic head."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.mlp import build_mlp


class BoundedDeltaHead(nn.Module):
    """Option-aware Delta regressor, scoped to supervised diagnostics."""

    def __init__(self, raw_network: nn.Module) -> None:
        super().__init__()
        self.raw_network = raw_network

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.raw_network(features).reshape(-1)
        is_call = features[:, 6]
        return map_delta_unit(raw, is_call).reshape(-1, 1)

    @classmethod
    def build(cls, input_dim: int, hidden: int = 128, depth: int = 3, **_: object) -> "BoundedDeltaHead":
        return cls(build_mlp(input_dim, 1, hidden, depth))


def map_delta_unit(raw_value, is_call):
    """Map raw outputs to put [-1,0] or call [0,1] Delta ranges."""
    unit = torch.sigmoid(raw_value)
    return torch.where(is_call > 0.5, unit, unit - 1.0)


def map_delta(raw_unit, is_call):
    """Backward-compatible scalar mapper for older scripts."""
    return 0.5 * (raw_unit + 1.0) if is_call else 0.5 * (raw_unit - 1.0)
