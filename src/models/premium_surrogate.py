"""Positive continuation-premium surrogate."""

from __future__ import annotations

from models.mlp import build_mlp


class PositivePremiumModel:
    """Factory wrapper to keep price/premium semantics explicit."""

    @staticmethod
    def build(input_dim: int, hidden: int = 64, depth: int = 3):
        import torch.nn as nn

        return nn.Sequential(build_mlp(input_dim, 1, hidden, depth), nn.Softplus())


def reconstruct_price(payoff, predicted_premium):
    return payoff + predicted_premium
