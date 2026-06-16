"""Direct supervised early-exercise boundary diagnostic heads."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.mlp import build_mlp


class ScaledSigmoid(nn.Module):
    """Map unconstrained scalar outputs to a configured boundary range."""

    def __init__(self, lower: float, upper: float) -> None:
        super().__init__()
        if upper <= lower:
            raise ValueError("upper must be greater than lower")
        self.lower = float(lower)
        self.upper = float(upper)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return self.lower + (self.upper - self.lower) * torch.sigmoid(raw)


class BoundaryHeadModel(nn.Module):
    """Boundary regressor for diagnostic estimates, not a price solver."""

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

    @classmethod
    def build(
        cls,
        input_dim: int,
        hidden: int = 128,
        depth: int = 3,
        lower: float | None = None,
        upper: float | None = None,
        **_: object,
    ) -> "BoundaryHeadModel":
        network = build_mlp(input_dim, 1, hidden, depth)
        if lower is not None and upper is not None:
            network = nn.Sequential(network, ScaledSigmoid(float(lower), float(upper)))
        return cls(network)


def build_boundary_head(input_dim: int, hidden: int = 128, depth: int = 3, lower: float | None = None, upper: float | None = None):
    return BoundaryHeadModel.build(input_dim, hidden=hidden, depth=depth, lower=lower, upper=upper)
