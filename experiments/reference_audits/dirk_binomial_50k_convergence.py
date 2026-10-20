"""Matched American-put convergence audit for the production DIRK reference.

The numerical method is *not* reimplemented here.  The module imports Ruixi's
frozen ``american_dirk_policy_price`` and ``sinh_spot_grid`` implementation
from a checkout pinned by CI.  Only M and N are varied.

The independent benchmark is the uploaded 50,000-step CRR American-put
implementation.  It is treated as a convergence-checked numerical benchmark,
not an analytic exact solution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from american_risk_surfaces.solvers.american_lcp import AmericanLCPConfig
from american_risk_surfaces.solvers.greek_integrators import american_dirk_policy_price
from american_risk_surfaces.solvers.grid import sinh_spot_grid

from experiments.reference_audits.crr50000_core import (
    SOURCE_ARCHIVE_SHA256,
    SOURCE_PRICE_SLICES_SHA256,
    SOURCE_SCRIPT_SHA256,
    crr_american_put,
)


RUIXI_SOURCE_COMMIT = "5f63fa7b12f704c3577c72f9d7b7e5146e78c1ca"
RUIXI_EXPORT_COMMIT = "63ce7cf6ffe04a778dc2f3b0ce691a26a1aca5c6"
REGIME_ID = "put_T100_s060_r005_q010"
REFERENCE_SOURCE = "DIRK_policy_sinh_M480_N960"

CONTRACT = {
    "option_type": "put",
    "K": 1.0,
    "T": 1.0,
    "r": 0.05,
    "q": 0.10,
    "sigma": 0.60,
    "Smax": 4.0,
}

BENCHMARK_STEPS = 50_000
SPOT_POINTS = 401
SLICE_TAUS = (1.0, 0.5, 0.1)
SPATIAL_LEVELS = (120, 240, 480, 960)
SPATIAL_FIXED_N = 960
TEMPORAL_LEVELS = (60, 120, 240, 480, 960, 1920)
TEMPORAL_FIXED_M = 480
LCP_TOLERANCE = 1e-12
OBSTACLE_TOLERANCE = 1e-12
DAMPING_STEPS = 2
REPRODUCTION_MAX_ABS_TOL = 1e-9
BENCHMARK_ANCHOR_TOL = 5e-12
ORDER_METRICS = ("mae", "rmse", "relative_l2", "linf", "p95_abs")


def common_spot_grid() -> np.ndarray:
    return np.linspace(0.5, 1.5, SPOT_POINTS, dtype=float)


def compute_error_metrics(approximate: np.ndarray, benchmark: np.ndarray) -> dict[str, float]:
    approximate = np.asarray(approximate, dtype=float)
    benchmark = np.asarray(benchmark, dtype=float)
    if approximate.shape != benchmark.shape or approximate.size == 0:
        raise ValueError("approximate and benchmark must have the same non-empty shape")
    if not np.all(np.isfinite(approximate)) or not np.all(np.isfinite(benchmark)):
        raise ValueError("error metrics require finite arrays")
    error = approximate - benchmark
    abs_error = np.abs(error)
    denominator = float(np.linalg.norm(benchmark.ravel(), ord=2))
    if denominator <= 0.0:
        raise ValueError("relative L2 denominator must be positive")
    return {
        "bias": float(np.mean(error)),
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "relative_l2": float(np.linalg.norm(error.ravel(), ord=2) / denominator),
        "linf": float(np.max(abs_error)),
        "p95_abs": float(np.quantile(abs_error, 0.95)),
    }


def observed_order(
    coarse_error: float,
    fine_error: float,
    *,
    refinement_ratio: float = 2.0,
) -> float:
    coarse = float(coarse_error)
    fine = float(fine_error)
    ratio = float(refinement_ratio)
    if coarse <= 0.0 or fine <= 0.0:
        raise ValueError("observed order requires positive errors")
    if ratio <= 1.0:
        raise ValueError("refinement_ratio must exceed one")
    return float(math.log(coarse / fine) / math.log(ratio))


def interpolate_surface_like_ruixi(
    source_m: np.ndarray,
    source_tau: np.ndarray,
    surface: np.ndarray,
    query_m: np.ndarray,
    query_tau: np.ndarray,
) -> np.ndarray:
    """Match Ruixi's space-then-time linear reference interpolation exactly."""
    source_m = np.asarray(source_m, dtype=float)
    source_tau = np.asarray(source_tau, dtype=float)
    surface = np.asarray(surface, dtype=float)
    query_m = np.asarray(query_m, dtype=float)
    query_tau = np.asarray(query_tau, dtype=float)
    if source_m.ndim != 1 or source_tau.ndim != 1:
        raise ValueError("source grids must be one-dimensional")
    if surface.shape != (len(source_tau), len(source_m)):
        raise ValueError("surface shape must match source grids")
    if query_m.shape != query_tau.shape:
        raise ValueError("query arrays must have the same shape")
    if np.any(query_m < source_m[0]) or np.any(query_m > source_m[-1]):
        raise ValueError("spot extrapolation is forbidden")
    if np.any(query_tau < source_tau[0]) or np.any(query_tau > source_tau[-1]):
        raise ValueError("time extrapolation is forbidden")

    upper_index = np.searchsorted(source_tau, query_tau, side="right")
    upper_index = np.clip(upper_index, 1, len(source_tau) - 1)
    lower_index = upper_index - 1
    denominator = source_tau[upper_index] - source_tau[lower_index]
    weight = np.divide(
        query_tau - source_tau[lower_index],
        denominator,
        out=np.zeros_like(query_tau),
        where=denominator > 0.0,
    )
    output = np.empty(len(query_m), dtype=float)
    for index in range(len(query_m)):
        lower_value = np.interp(
            query_m[index], source_m, surface[lower_index[index]]
        )
        upper_value = np.interp(
            query_m[index], source_m, surface[upper_index[index]]
        )
        output[index] = lower_value + weight[index] * (upper_value - lower_value)
    return output


def solve_dirk_price_slices(
    *,
    M: int,
    N: int,
    query_spots: np.ndarray | None = None,
    query_taus: Iterable[float] = SLICE_TAUS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run Ruixi's frozen DIRK/Policy/sinh solver and interpolate price slices."""
    config = AmericanLCPConfig(
        option_type=CONTRACT["option_type"],
        K=CONTRACT["K"],
        T=CONTRACT["T"],
        r=CONTRACT["r"],
        q=CONTRACT["q"],
        sigma=CONTRACT["sigma"],
        Smax=CONTRACT["Smax"],
        M=int(M),
        N=int(N),
        tolerance=LCP_TOLERANCE,
        obstacle_tolerance=OBSTACLE_TOLERANCE,
    )
    spot_grid = sinh_spot_grid(config.Smax, config.K, config.M)
    result = american_dirk_policy_price(
        config,
        quadratic_time=True,
        damping_steps=DAMPING_STEPS,
        spot_grid=spot_grid,
    )
    if not result.converged:
        raise RuntimeError(f"DIRK solve failed to converge for M={M}, N={N}")
    if result.max_obstacle_violation > max(OBSTACLE_TOLERANCE, 1e-14):
        raise RuntimeError(
            f"obstacle violation {result.max_obstacle_violation} for M={M}, N={N}"
        )

    spots = common_spot_grid() if query_spots is None else np.asarray(query_spots, dtype=float)
    taus = np.asarray(tuple(query_taus), dtype=float)
    spot_mesh, tau_mesh = np.meshgrid(spots, taus)
    flat = interpolate_surface_like_ruixi(
        result.spot_grid / config.K,
        result.tau_grid / config.T,
        result.value_grid / config.K,
        spot_mesh.reshape(-1),
        tau_mesh.reshape(-1),
    )
    values = flat.reshape(len(taus), len(spots))
    metadata = {
        "method": result.method,
        "M": int(M),
        "N": int(N),
        "runtime_seconds": float(result.total_seconds),
        "converged": bool(result.converged),
        "max_obstacle_violation": float(result.max_obstacle_violation),
        "time_grid": result.time_grid,
        "spot_grid": "sinh_strike_concentrated",
        "sinh_default_concentration_width": 0.1 * config.K,
        "damping_steps": DAMPING_STEPS,
        "lcp_tolerance": config.tolerance,
        "obstacle_tolerance": config.obstacle_tolerance,
    }
    return values, metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()


def verify_ruixi_source(repo_root: Path) -> dict[str, Any]:
    actual = _git_head(repo_root)
    passed = actual == RUIXI_SOURCE_COMMIT
    return {
        "expected_commit": RUIXI_SOURCE_COMMIT,
        "actual_commit": actual,
        "passed": passed,
    }


def _reference_slice_file(reference_dir: Path, tau: float) -> Path:
    mapping = {
        1.0: "price_slice_tau_1.csv",
        0.5: "price_slice_tau_05.csv",
        0.1: "price_slice_tau_01.csv",
    }
    return reference_dir / mapping[float(tau)]


def load_stored_reference_slices(reference_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    spots: np.ndarray | None = None
    values: list[np.ndarray] = []
    for tau in SLICE_TAUS:
        path = _reference_slice_file(reference_dir, tau)
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        current_spots = np.asarray([float(row["S_over_K"]) for row in rows], dtype=float)
        current_values = np.asarray([float(row["V_over_K"]) for row in rows], dtype=float)
        if spots is None:
            spots = current_spots
        elif not np.array_equal(spots, current_spots):
            raise RuntimeError("stored reference slices do not share the same spot grid")
        values.append(current_values)
    assert spots is not None
    if len(spots) != SPOT_POINTS or not np.allclose(spots, common_spot_grid(), atol=0.0, rtol=0.0):
        raise RuntimeError("stored reference spot grid does not match frozen common grid")
    return spots, np.vstack(values)


def verify_canonical_reproduction(reference_dir: Path) -> dict[str, Any]:
    spots, stored = load_stored_reference_slices(reference_dir)
    rerun, metadata = solve_dirk_price_slices(
        M=480,
        N=960,
        query_spots=spots,
        query_taus=SLICE_TAUS,
    )
    difference = rerun - stored
    max_abs = float(np.max(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(difference**2)))
    return {
        "reference_source": REFERENCE_SOURCE,
        "ruixi_source_commit": RUIXI_SOURCE_COMMIT,
        "ruixi_export_commit": RUIXI_EXPORT_COMMIT,
        "M": 480,
        "N": 960,
        "points": int(difference.size),
        "max_abs_difference": max_abs,
        "rmse_difference": rmse,
        "tolerance": REPRODUCTION_MAX_ABS_TOL,
        "passed": bool(max_abs <= REPRODUCTION_MAX_ABS_TOL),
        "solver_metadata": metadata,
        "stored_slice_sha256": {
            str(tau): _sha256(_reference_slice_file(reference_dir, tau))
            for tau in SLICE_TAUS
        },
    }


def _crr_price_job(job: tuple[int, float, float]) -> tuple[int, float]:
    index, spot, tau = job
    return index, crr_american_put(spot, tau, steps=BENCHMARK_STEPS)


def generate_crr_price_grid(*, workers: int) -> np.ndarray:
    spots = common_spot_grid()
    jobs: list[tuple[int, float, float]] = []
    for tau_index, tau in enumerate(SLICE_TAUS):
        for spot_index, spot in enumerate(spots):
            flat_index = tau_index * len(spots) + spot_index
            jobs.append((flat_index, float(spot), float(tau)))
    values = np.empty(len(jobs), dtype=float)
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = [executor.submit(_crr_price_job, job) for job in jobs]
        for future in as_completed(futures):
            index, value = future.result()
            values[index] = value
    return values.reshape(len(SLICE_TAUS), len(spots))


def write_crr_cache(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    spots = common_spot_grid()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("S_over_K", "tau_over_T", "steps", "V_over_K"),
            lineterminator="\n",
        )
        writer.writeheader()
        for tau_index, tau in enumerate(SLICE_TAUS):
            for spot_index, spot in enumerate(spots):
                writer.writerow(
                    {
                        "S_over_K": format(float(spot), ".17g"),
                        "tau_over_T": format(float(tau), ".17g"),
                        "steps": str(BENCHMARK_STEPS),
                        "V_over_K": format(float(values[tau_index, spot_index]), ".17g"),
                    }
                )


def load_crr_cache(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(SLICE_TAUS) * SPOT_POINTS:
        raise RuntimeError("CRR cache does not contain 1203 rows")
    spots = common_spot_grid()
    values = np.empty((len(SLICE_TAUS), len(spots)), dtype=float)
    expected = [(tau, spot) for tau in SLICE_TAUS for spot in spots]
    for row, (expected_tau, expected_spot) in zip(rows, expected):
        if int(row["steps"]) != BENCHMARK_STEPS:
            raise RuntimeError("CRR cache has wrong tree step count")
        if not math.isclose(float(row["tau_over_T"]), float(expected_tau), rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError("CRR cache tau grid mismatch")
        if not math.isclose(float(row["S_over_K"]), float(expected_spot), rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError("CRR cache spot grid mismatch")
        tau_index = SLICE_TAUS.index(expected_tau)
        spot_index = int(round((expected_spot - 0.5) / 0.0025))
        values[tau_index, spot_index] = float(row["V_over_K"])
    if not np.all(np.isfinite(values)):
        raise RuntimeError("CRR cache contains non-finite values")
    return values


def load_or_generate_crr(path: Path, *, workers: int) -> tuple[np.ndarray, bool]:
    if path.exists():
        return load_crr_cache(path), True
    values = generate_crr_price_grid(workers=workers)
    write_crr_cache(path, values)
    return values, False


def verify_crr_anchors(values: np.ndarray, anchor_path: Path) -> dict[str, Any]:
    spots = common_spot_grid()
    with anchor_path.open(newline="", encoding="utf-8") as handle:
        anchors = list(csv.DictReader(handle))
    differences = []
    for row in anchors:
        if int(row["steps"]) != BENCHMARK_STEPS:
            raise RuntimeError("anchor table does not use 50,000 steps")
        tau = float(row["tau_over_T"])
        spot = float(row["S_over_K"])
        try:
            tau_index = SLICE_TAUS.index(tau)
        except ValueError as exc:
            raise RuntimeError(f"unexpected anchor tau {tau}") from exc
        spot_index = int(np.argmin(np.abs(spots - spot)))
        if abs(float(spots[spot_index]) - spot) > 1e-12:
            raise RuntimeError(f"anchor spot {spot} is not on common grid")
        differences.append(values[tau_index, spot_index] - float(row["V_over_K"]))
    difference = np.asarray(differences, dtype=float)
    max_abs = float(np.max(np.abs(difference)))
    return {
        "anchor_file": str(anchor_path),
        "anchor_sha256": _sha256(anchor_path),
        "anchor_count": len(anchors),
        "max_abs_difference": max_abs,
        "tolerance": BENCHMARK_ANCHOR_TOL,
        "passed": bool(max_abs <= BENCHMARK_ANCHOR_TOL),
        "uploaded_source_script_sha256": SOURCE_SCRIPT_SHA256,
        "uploaded_source_price_slices_sha256": SOURCE_PRICE_SLICES_SHA256,
        "uploaded_source_archive_sha256": SOURCE_ARCHIVE_SHA256,
    }


def _run_pairs() -> list[tuple[int, int]]:
    pairs = {(m, SPATIAL_FIXED_N) for m in SPATIAL_LEVELS}
    pairs.update((TEMPORAL_FIXED_M, n) for n in TEMPORAL_LEVELS)
    return sorted(pairs)


def _metric_rows(
    solved: dict[tuple[int, int], tuple[np.ndarray, dict[str, Any]]],
    benchmark: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (M, N), (values, metadata) in sorted(solved.items()):
        for tau_index, tau in enumerate(SLICE_TAUS):
            metrics = compute_error_metrics(values[tau_index], benchmark[tau_index])
            rows.append(
                {
                    "run_id": f"M{M}_N{N}",
                    "M": M,
                    "N": N,
                    "scope": f"tau_{tau:g}",
                    "tau_over_T": tau,
                    "points": SPOT_POINTS,
                    **metrics,
                    "runtime_seconds": metadata["runtime_seconds"],
                }
            )
        metrics = compute_error_metrics(values.reshape(-1), benchmark.reshape(-1))
        rows.append(
            {
                "run_id": f"M{M}_N{N}",
                "M": M,
                "N": N,
                "scope": "overall",
                "tau_over_T": "",
                "points": int(values.size),
                **metrics,
                "runtime_seconds": metadata["runtime_seconds"],
            }
        )
    return rows


def _orders(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (int(row["M"]), int(row["N"]), str(row["scope"])): row for row in metric_rows
    }
    rows: list[dict[str, Any]] = []
    axes = (
        ("spatial", SPATIAL_LEVELS, lambda level: (level, SPATIAL_FIXED_N)),
        ("temporal", TEMPORAL_LEVELS, lambda level: (TEMPORAL_FIXED_M, level)),
    )
    scopes = ("overall",) + tuple(f"tau_{tau:g}" for tau in SLICE_TAUS)
    for axis, levels, pair_for in axes:
        for coarse_level, fine_level in zip(levels, levels[1:]):
            coarse_pair = pair_for(coarse_level)
            fine_pair = pair_for(fine_level)
            ratio = fine_level / coarse_level
            for scope in scopes:
                coarse = lookup[(*coarse_pair, scope)]
                fine = lookup[(*fine_pair, scope)]
                for metric in ORDER_METRICS:
                    rows.append(
                        {
                            "axis": axis,
                            "scope": scope,
                            "metric": metric,
                            "coarse_level": coarse_level,
                            "fine_level": fine_level,
                            "coarse_M": coarse_pair[0],
                            "coarse_N": coarse_pair[1],
                            "fine_M": fine_pair[0],
                            "fine_N": fine_pair[1],
                            "refinement_ratio": ratio,
                            "coarse_error": coarse[metric],
                            "fine_error": fine[metric],
                            "error_ratio": float(coarse[metric]) / float(fine[metric]),
                            "observed_order": observed_order(
                                float(coarse[metric]),
                                float(fine[metric]),
                                refinement_ratio=ratio,
                            ),
                        }
                    )
    return rows


def _pointwise_rows(
    solved: dict[tuple[int, int], tuple[np.ndarray, dict[str, Any]]],
    benchmark: np.ndarray,
) -> list[dict[str, Any]]:
    spots = common_spot_grid()
    rows: list[dict[str, Any]] = []
    for (M, N), (values, _metadata) in sorted(solved.items()):
        for tau_index, tau in enumerate(SLICE_TAUS):
            for spot_index, spot in enumerate(spots):
                error = float(values[tau_index, spot_index] - benchmark[tau_index, spot_index])
                rows.append(
                    {
                        "run_id": f"M{M}_N{N}",
                        "M": M,
                        "N": N,
                        "S_over_K": float(spot),
                        "tau_over_T": float(tau),
                        "dirk_V_over_K": float(values[tau_index, spot_index]),
                        "crr50000_V_over_K": float(benchmark[tau_index, spot_index]),
                        "signed_error": error,
                        "abs_error": abs(error),
                    }
                )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _axis_summary(metric_rows: list[dict[str, Any]], order_rows: list[dict[str, Any]], axis: str) -> list[dict[str, Any]]:
    levels = SPATIAL_LEVELS if axis == "spatial" else TEMPORAL_LEVELS
    pair_for = (
        (lambda level: (level, SPATIAL_FIXED_N))
        if axis == "spatial"
        else (lambda level: (TEMPORAL_FIXED_M, level))
    )
    overall = {
        (int(row["M"]), int(row["N"])): row
        for row in metric_rows
        if row["scope"] == "overall"
    }
    order_lookup = {
        (int(row["fine_level"]), str(row["metric"])): float(row["observed_order"])
        for row in order_rows
        if row["axis"] == axis and row["scope"] == "overall"
    }
    output = []
    for level in levels:
        M, N = pair_for(level)
        metric = overall[(M, N)]
        row = {
            "M": M,
            "N": N,
            "bias": metric["bias"],
            "mae": metric["mae"],
            "rmse": metric["rmse"],
            "relative_l2": metric["relative_l2"],
            "linf": metric["linf"],
            "p95_abs": metric["p95_abs"],
            "runtime_seconds": metric["runtime_seconds"],
        }
        for name in ORDER_METRICS:
            row[f"p_{name}"] = order_lookup.get((level, name), "")
        output.append(row)
    return output


def run_audit(
    *,
    output_dir: Path,
    reference_dir: Path,
    ruixi_repo_root: Path,
    anchor_path: Path,
    workers: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    source_gate = verify_ruixi_source(ruixi_repo_root)
    _write_json(output_dir / "ruixi_source_gate.json", source_gate)
    if not source_gate["passed"]:
        raise RuntimeError("Ruixi source checkout does not match frozen commit")

    reproduction = verify_canonical_reproduction(reference_dir)
    _write_json(output_dir / "reproduction_gate.json", reproduction)
    if not reproduction["passed"]:
        raise RuntimeError(
            "canonical M480/N960 reproduction gate failed; refusing convergence sweep"
        )

    benchmark_path = output_dir / "crr50000_price_benchmark.csv"
    benchmark, reused_cache = load_or_generate_crr(benchmark_path, workers=workers)
    benchmark_gate = verify_crr_anchors(benchmark, anchor_path)
    benchmark_gate["cache_reused"] = reused_cache
    benchmark_gate["generated_benchmark_sha256"] = _sha256(benchmark_path)
    _write_json(output_dir / "benchmark_reproduction_gate.json", benchmark_gate)
    if not benchmark_gate["passed"]:
        raise RuntimeError("CRR50000 anchor reproduction gate failed")

    solved: dict[tuple[int, int], tuple[np.ndarray, dict[str, Any]]] = {}
    for M, N in _run_pairs():
        solved[(M, N)] = solve_dirk_price_slices(M=M, N=N)

    metric_rows = _metric_rows(solved, benchmark)
    order_rows = _orders(metric_rows)
    pointwise_rows = _pointwise_rows(solved, benchmark)
    _write_csv(output_dir / "grid_metrics.csv", metric_rows)
    _write_csv(output_dir / "observed_orders.csv", order_rows)
    _write_csv(output_dir / "pointwise_price_errors.csv", pointwise_rows)
    _write_csv(
        output_dir / "spatial_convergence.csv",
        _axis_summary(metric_rows, order_rows, "spatial"),
    )
    _write_csv(
        output_dir / "temporal_convergence.csv",
        _axis_summary(metric_rows, order_rows, "temporal"),
    )

    runs = []
    for (M, N), (_values, metadata) in sorted(solved.items()):
        runs.append({"run_id": f"M{M}_N{N}", **metadata})
    _write_csv(output_dir / "run_metrics.csv", runs)

    manifest = {
        "schema_version": 1,
        "experiment_id": "dirk_binomial_50k_convergence_audit",
        "regime_id": REGIME_ID,
        "contract": CONTRACT,
        "method": {
            "solver": "Ruixi american_dirk_policy_price",
            "time_integrator": "second_order_Cash_DIRK",
            "policy_iteration": True,
            "spot_grid": "sinh_strike_concentrated",
            "sinh_concentration_width": "default 0.1*K",
            "time_grid": "quadratic",
            "damping_steps": DAMPING_STEPS,
            "lcp_tolerance": LCP_TOLERANCE,
            "obstacle_tolerance": OBSTACLE_TOLERANCE,
            "ruixi_source_commit": RUIXI_SOURCE_COMMIT,
            "ruixi_export_commit": RUIXI_EXPORT_COMMIT,
        },
        "benchmark": {
            "method": "CRR American put",
            "steps": BENCHMARK_STEPS,
            "role": "convergence_checked_independent_numerical_benchmark",
            "analytic_ground_truth": False,
            "uploaded_source_script_sha256": SOURCE_SCRIPT_SHA256,
            "uploaded_source_price_slices_sha256": SOURCE_PRICE_SLICES_SHA256,
            "uploaded_source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "generated_price_benchmark_sha256": _sha256(benchmark_path),
        },
        "evaluation": {
            "S_over_K_min": 0.5,
            "S_over_K_max": 1.5,
            "S_points": SPOT_POINTS,
            "tau_over_T": list(SLICE_TAUS),
            "overall_points": SPOT_POINTS * len(SLICE_TAUS),
            "metrics": ["bias", "mae", "rmse", "relative_l2", "linf", "p95_abs"],
            "primary_order_metric": "relative_l2",
            "co_primary_order_metric": "rmse",
            "strict_secondary_order_metric": "linf",
            "order_formula": "log(E_coarse/E_fine)/log(refinement_ratio)",
        },
        "spatial_refinement": {"fixed_N": SPATIAL_FIXED_N, "M": list(SPATIAL_LEVELS)},
        "temporal_refinement": {"fixed_M": TEMPORAL_FIXED_M, "N": list(TEMPORAL_LEVELS)},
        "reproduction_gate": reproduction,
        "benchmark_reproduction_gate": benchmark_gate,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "crr_workers": workers,
        },
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dirk_binomial_50k_convergence"),
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path(
            "_vendor/ruixi-export/results/visualization_exports/american_put_matched/"
            f"{REGIME_ID}/reference"
        ),
    )
    parser.add_argument("--ruixi-repo-root", type=Path, default=Path("_vendor/ruixi-code"))
    parser.add_argument(
        "--anchor-path",
        type=Path,
        default=Path("results/consolidated_csv/crr50000_price_anchor_points.csv"),
    )
    parser.add_argument("--workers", type=int, default=max(1, min(2, os.cpu_count() or 1)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    manifest = run_audit(
        output_dir=args.output_dir,
        reference_dir=args.reference_dir,
        ruixi_repo_root=args.ruixi_repo_root,
        anchor_path=args.anchor_path,
        workers=args.workers,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
