"""Conventional fully-connected PINN baseline for HD3."""
from __future__ import annotations

import torch
from torch import nn

from physics.high_dimensional_black_scholes import geometric_put_payoff


class FCPINN(nn.Module):
    """Six-layer 128-wide tanh MLP with the HD1 hard value transform."""

    def __init__(self, d: int, width: int = 128, hidden_layers: int = 6) -> None:
        super().__init__()
        if d < 1:
            raise ValueError("d must be positive")
        if width < 1 or hidden_layers < 1:
            raise ValueError("width and hidden_layers must be positive")
        self.d = int(d)
        self.width = int(width)
        self.hidden_layers = int(hidden_layers)
        layers: list[nn.Module] = []
        in_width = self.d + 1
        for _ in range(self.hidden_layers):
            layers.extend((nn.Linear(in_width, self.width), nn.Tanh()))
            in_width = self.width
        layers.append(nn.Linear(in_width, 1))
        self.network = nn.Sequential(*layers)

    def raw_head(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError(f"x must have shape (batch, {self.d})")
        tau_flat = tau.reshape(-1)
        if tau_flat.shape[0] != x.shape[0]:
            raise ValueError("x/tau batch mismatch")
        inputs = torch.cat((x, tau_flat.reshape(-1, 1)), dim=1)
        return self.network(inputs).reshape(-1)

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        payoff = geometric_put_payoff(x)
        return payoff + tau.reshape(-1) * torch.nn.functional.softplus(self.raw_head(x, tau))


def initialize_fc_model(
    d: int,
    seed: int = 101,
    *,
    width: int = 128,
    hidden_layers: int = 6,
    device=None,
    dtype=torch.float32,
) -> FCPINN:
    torch.manual_seed(int(seed))
    return FCPINN(d, width=width, hidden_layers=hidden_layers).to(device=device, dtype=dtype)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
