"""Helpers shared by stage experiment scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common import ensure_dir
from solvers.cn_psor import SolverConfig


def solver_config_from_dict(data: dict) -> SolverConfig:
    allowed = set(SolverConfig.__dataclass_fields__)
    cleaned = {k: v for k, v in data.items() if k in allowed}
    return SolverConfig(**cleaned)


def save_npz(path: str | Path, **arrays) -> None:
    ensure_dir(Path(path).parent)
    np.savez_compressed(path, **arrays)


def dataset_exists(path: str | Path) -> bool:
    return Path(path).exists()
