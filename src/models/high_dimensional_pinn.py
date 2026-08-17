"""Permutation-invariant DeepSets value surrogate frozen for the HD1 pilot."""
from __future__ import annotations

import torch
from torch import nn

from physics.high_dimensional_black_scholes import geometric_put_payoff


class DeepSetsPINN(nn.Module):
    """Two 64-wide per-asset layers, mean aggregation, four 128-wide value layers."""

    def __init__(self) -> None:
        super().__init__()
        self.asset_encoder = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh()
        )
        layers = []
        in_width = 65
        for _ in range(4):
            layers.extend([nn.Linear(in_width, 128), nn.Tanh()])
            in_width = 128
        layers.append(nn.Linear(128, 1))
        self.value_head = nn.Sequential(*layers)

    def raw_head(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or tau.reshape(-1).shape[0] != x.shape[0]:
            raise ValueError("x/tau batch mismatch")
        encoded = self.asset_encoder(x.unsqueeze(-1)).mean(dim=1)
        return self.value_head(torch.cat([encoded, tau.reshape(-1, 1)], dim=1)).reshape(-1)

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        payoff = geometric_put_payoff(x)
        return payoff + tau.reshape(-1) * torch.nn.functional.softplus(self.raw_head(x, tau))


def initialize_model(seed: int = 101, *, device=None, dtype=torch.float32) -> DeepSetsPINN:
    torch.manual_seed(int(seed))
    return DeepSetsPINN().to(device=device, dtype=dtype)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
