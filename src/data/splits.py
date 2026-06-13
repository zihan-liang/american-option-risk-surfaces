"""Regime-level train/test split utilities."""

from __future__ import annotations

import random
from typing import Iterable


def regime_split(regimes: list[dict], test_fraction: float = 0.25, seed: int = 7) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = list(regimes)
    rng.shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * test_fraction))
    return shuffled[n_test:], shuffled[:n_test]


def regime_key(regime: dict) -> str:
    return "|".join(f"{k}={regime[k]}" for k in sorted(regime))
