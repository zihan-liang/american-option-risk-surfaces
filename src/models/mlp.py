"""Small PyTorch MLP blocks for surrogate experiments."""

from __future__ import annotations


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def build_mlp(input_dim: int, output_dim: int, hidden: int = 64, depth: int = 3):
    import torch.nn as nn

    layers = []
    dim = input_dim
    for _ in range(depth):
        layers.extend([nn.Linear(dim, hidden), nn.Tanh()])
        dim = hidden
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)
