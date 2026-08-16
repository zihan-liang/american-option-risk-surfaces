from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def build_seeded_mlp(
    builder: Callable[..., Any],
    *,
    seed: int,
    input_dim: int,
    output_dim: int,
    hidden: int,
    depth: int,
) -> Any:
    import torch

    torch.manual_seed(int(seed))
    return builder(input_dim, output_dim, hidden=hidden, depth=depth)


def selected_test_keys(
    selected_by_seed: Mapping[int, Mapping[str, float | None]],
) -> list[tuple[int, str, float]]:
    keys: list[tuple[int, str, float]] = []
    for seed in sorted(int(x) for x in selected_by_seed):
        chosen = selected_by_seed[seed]
        for arm in ("fine_only", "coarse_plus_few_fine"):
            fraction = chosen.get(arm)
            if fraction is None:
                raise ValueError("matched-accuracy selection incomplete")
            keys.append((seed, arm, float(fraction)))
    return keys
