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


def three_way_regime_split(
    regimes: list[dict],
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 7,
) -> dict[str, list[dict]]:
    if not regimes:
        return {"train": [], "validation": [], "test": []}
    if train_fraction <= 0 or validation_fraction < 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("fractions must satisfy train > 0, validation >= 0, train + validation < 1")
    rng = random.Random(seed)
    shuffled = list(regimes)
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_train = min(n_total, round(n_total * train_fraction))
    n_validation = round(n_total * validation_fraction)
    if n_total >= 3:
        n_train = max(1, min(n_train, n_total - 2))
        n_validation = max(1, min(n_validation, n_total - n_train - 1))
    else:
        n_validation = min(n_validation, max(0, n_total - n_train))
    return {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train : n_train + n_validation],
        "test": shuffled[n_train + n_validation :],
    }


def regime_key(regime: dict) -> str:
    return "|".join(f"{k}={regime[k]}" for k in sorted(regime))
