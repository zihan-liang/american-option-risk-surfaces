"""Positive continuation-premium surrogate.

This module is intentionally price/premium-only. Boundary and Greek models live
in separate heads and must be validated separately.
"""

from __future__ import annotations

import numpy as np

from models.mlp import build_mlp


class PositivePremiumModel:
    """Factory wrapper to keep price/premium semantics explicit."""

    @staticmethod
    def build(input_dim: int, hidden: int = 64, depth: int = 3):
        import torch.nn as nn

        return nn.Sequential(build_mlp(input_dim, 1, hidden, depth), nn.Softplus())


class TerminalScaledPositivePremiumModel:
    """Positive premium model that enforces zero premium at theta=tau/T=0."""

    @staticmethod
    def build(input_dim: int, hidden: int = 128, depth: int = 3):
        import torch
        from torch import nn

        base = build_mlp(input_dim, 1, hidden=hidden, depth=depth)

        class _TerminalScaled(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net
                self.softplus = nn.Softplus()

            def forward(self, features):
                theta = torch.clamp(features[:, 1:2], min=0.0)
                return theta * self.softplus(self.net(features))

        return _TerminalScaled(base)


def reconstruct_price(payoff, predicted_premium):
    if _is_torch_tensor(payoff) or _is_torch_tensor(predicted_premium):
        return payoff + predicted_premium
    payoff_array = np.asarray(payoff)
    premium_array = np.asarray(predicted_premium)
    if payoff_array.shape != premium_array.shape:
        raise ValueError(f"payoff and premium shapes differ: {payoff_array.shape} != {premium_array.shape}")
    return payoff + predicted_premium


def _is_torch_tensor(value) -> bool:
    try:
        import torch

        return isinstance(value, torch.Tensor)
    except Exception:
        return False
