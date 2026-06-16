"""Grid/domain convergence comparison helpers."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from solvers.boundaries import BoundaryResult


def surface_to_reference_grid(
    source_spots: np.ndarray,
    source_taus: np.ndarray,
    source_surface: np.ndarray,
    reference_spots: np.ndarray,
    reference_taus: np.ndarray,
) -> np.ndarray:
    """Interpolate a time-by-spot surface onto a reference tensor-product grid."""
    spot_interp = np.vstack(
        [np.interp(reference_spots, source_spots, row) for row in source_surface]
    )
    time_interp = np.vstack(
        [
            np.interp(reference_taus, source_taus, spot_interp[:, j])
            for j in range(spot_interp.shape[1])
        ]
    ).T
    return time_interp


def masked_error(candidate: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    active = mask & np.isfinite(candidate) & np.isfinite(reference)
    samples = int(np.count_nonzero(active))
    if samples == 0:
        return {"samples": 0, "rmse": float("nan"), "mae": float("nan"), "max": float("nan")}
    diff = candidate[active] - reference[active]
    return {
        "samples": samples,
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
        "max": float(np.max(np.abs(diff))),
    }


def boundary_error(
    candidate: Iterable[BoundaryResult],
    reference: Iterable[BoundaryResult],
) -> dict[str, float | int]:
    diffs: list[float] = []
    for cand, ref in zip(candidate, reference):
        if cand.status == "boundary_found" and ref.status == "boundary_found":
            if math.isfinite(cand.value) and math.isfinite(ref.value):
                diffs.append(cand.value - ref.value)
    if not diffs:
        return {"samples": 0, "rmse": float("nan"), "mae": float("nan"), "max": float("nan")}
    arr = np.asarray(diffs, dtype=float)
    return {
        "samples": int(arr.size),
        "rmse": float(np.sqrt(np.mean(arr**2))),
        "mae": float(np.mean(np.abs(arr))),
        "max": float(np.max(np.abs(arr))),
    }


def psor_iteration_summary(iterations: list[int]) -> dict[str, float | int]:
    if not iterations:
        return {"max_psor_iterations": 0, "mean_psor_iterations": 0.0}
    arr = np.asarray(iterations, dtype=float)
    return {
        "max_psor_iterations": int(np.max(arr)),
        "mean_psor_iterations": float(np.mean(arr)),
    }
