"""M04 neural representations and conservative put-boundary extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from models.mlp import build_mlp


class ScalarPutBoundaryHead(nn.Module):
    """A graph-valued put boundary constrained to the normalized strike interval."""

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    @classmethod
    def build(cls, input_dim: int = 4, hidden: int = 128, depth: int = 3) -> "ScalarPutBoundaryHead":
        return cls(nn.Sequential(build_mlp(input_dim, 1, hidden, depth), nn.Sigmoid()))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class ImplicitPutLevelSet(nn.Module):
    """A signed exercise/continuation representation whose zero set is the boundary."""

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    @classmethod
    def build(cls, input_dim: int = 5, hidden: int = 128, depth: int = 3) -> "ImplicitPutLevelSet":
        return cls(build_mlp(input_dim, 1, hidden, depth))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class PositivePremiumSurrogate(nn.Module):
    """Strictly non-negative premium surrogate used only by M04's baseline."""

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    @classmethod
    def build(cls, input_dim: int = 5, hidden: int = 128, depth: int = 3) -> "PositivePremiumSurrogate":
        return cls(nn.Sequential(build_mlp(input_dim, 1, hidden, depth), nn.Softplus()))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


@dataclass(frozen=True)
class RootExtraction:
    value: float | None
    status: str
    reason: str | None
    bracket: tuple[float, float] | None


def extract_put_zero_crossing(spots: Sequence[float], levels: Sequence[float], *, tolerance: float = 1e-12) -> RootExtraction:
    """Return one supported low-spot exercise to high-spot continuation crossing.

    A zero is accepted only at the contact transition. Any reversed or multiple
    crossing is reported rather than arbitrarily selecting a root.
    """
    x = np.asarray(spots, dtype=float)
    y = np.asarray(levels, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("spots and levels must be equally sized one-dimensional arrays")
    if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.all(np.diff(x) > 0.0):
        raise ValueError("root extraction requires finite, strictly increasing paired nodes")
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")

    signs = np.where(y > tolerance, 1, np.where(y < -tolerance, -1, 0))
    if np.all(signs >= 0):
        return RootExtraction(None, "no_exercise_region", "all levels are continuation", None)
    if np.all(signs <= 0):
        return RootExtraction(None, "no_continuation_region", "all levels are exercise", None)
    transitions: list[tuple[int, int]] = []
    for left in range(len(signs) - 1):
        if signs[left] < 0 and signs[left + 1] > 0:
            transitions.append((left, left + 1))
    zeros = np.flatnonzero(signs == 0)
    for index in zeros:
        left_negative = index > 0 and signs[index - 1] < 0
        right_positive = index + 1 < len(signs) and signs[index + 1] > 0
        if left_negative and right_positive:
            transitions.append((int(index), int(index)))
        elif (index > 0 and signs[index - 1] > 0) or (index + 1 < len(signs) and signs[index + 1] < 0):
            return RootExtraction(None, "non_monotone_profile", "zero lacks exercise-to-continuation contact", None)
    if len(transitions) != 1:
        return RootExtraction(None, "non_monotone_profile", "profile has reversed or multiple crossings", None)
    left, right = transitions[0]
    if left == right:
        return RootExtraction(float(x[left]), "defined", None, (float(x[left]), float(x[left])))
    value = float(x[left] - y[left] * (x[right] - x[left]) / (y[right] - y[left]))
    return RootExtraction(value, "defined", None, (float(x[left]), float(x[right])))
