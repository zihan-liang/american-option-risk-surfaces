#!/usr/bin/env python3
"""M05: classical Schrödingerization proxy for two American-option inner solves."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import numpy as np

from common import load_config
from solvers.cn_psor import (
    SolverConfig,
    cn_matrices,
    lcp_natural_residual,
    solve_option,
    time_boundary_values,
)


def dissipative_recovery(
    matrix: np.ndarray,
    rhs: np.ndarray,
    initial: np.ndarray,
    *,
    decay_tolerance: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Classically recover finite-time state of du/ds=A^T(b-Au).

    This is a deterministic proxy for the dissipative evolution that a
    Schrödingerization would embed. It is not a quantum circuit simulation.
    """
    matrix = np.asarray(matrix, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    initial = np.asarray(initial, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if rhs.shape != (matrix.shape[0],) or initial.shape != rhs.shape:
        raise ValueError("rhs and initial must match matrix dimension")
    if not 0.0 < decay_tolerance < 1.0:
        raise ValueError("decay_tolerance must lie strictly between zero and one")
    if not (np.isfinite(matrix).all() and np.isfinite(rhs).all() and np.isfinite(initial).all()):
        raise ValueError("all inputs must be finite")

    normal = matrix.T @ matrix
    eigenvalues, eigenvectors = np.linalg.eigh(normal)
    if eigenvalues[0] <= np.finfo(float).eps * max(1.0, eigenvalues[-1]):
        raise ValueError("matrix is singular or numerically rank deficient")
    horizon = -np.log(decay_tolerance) / eigenvalues[0]
    transformed_rhs = eigenvectors.T @ (matrix.T @ rhs)
    steady_coefficients = transformed_rhs / eigenvalues
    initial_coefficients = eigenvectors.T @ initial
    decay = np.exp(-eigenvalues * horizon)
    recovered = eigenvectors @ (
        steady_coefficients + decay * (initial_coefficients - steady_coefficients)
    )
    return recovered, {
        "horizon": float(horizon),
        "normal_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "slow_mode_decay": float(decay.max()),
    }


def _boundary_source(
    cfg: SolverConfig,
    tau: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    source = np.zeros(cfg.m - 1, dtype=float)
    left, right = time_boundary_values(cfg, tau)
    source[0] = lower[0] * left
    source[-1] = upper[-1] * right
    return source


def _cn_problem(
    cfg: SolverConfig,
    values: np.ndarray,
    step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, lower, upper, matrix, rhs_matrix = cn_matrices(cfg)
    dt = cfg.t / cfg.n
    tau_old = (step - 1) * dt
    tau_new = step * dt
    previous = values[step - 1, 1:-1].copy()
    rhs = rhs_matrix @ previous
    rhs += 0.5 * dt * (
        _boundary_source(cfg, tau_old, lower, upper)
        + _boundary_source(cfg, tau_new, lower, upper)
    )
    obstacle = values[0, 1:-1].copy()
    return matrix, rhs, obstacle, previous


def _policy_system(
    matrix: np.ndarray,
    rhs: np.ndarray,
    obstacle: np.ndarray,
    exercise: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    system = matrix.copy()
    policy_rhs = rhs.copy()
    system[exercise, :] = 0.0
    indices = np.flatnonzero(exercise)
    system[indices, indices] = 1.0
    policy_rhs[exercise] = obstacle[exercise]
    return system, policy_rhs


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _solver_config(config: Mapping[str, Any], *, multiplier: int = 1) -> SolverConfig:
    return SolverConfig(
        option_type="put",
        k=float(config["k"]),
        t=float(config["t"]),
        r=float(config["r"]),
        q=float(config["q"]),
        sigma=float(config["sigma"]),
        s_max=float(config["s_max"]),
        m=int(config["medium_m"]) * multiplier,
        n=int(config["medium_n"]) * multiplier,
        omega=1.2,
        tol=float(config["tol"]),
        max_iter=int(config["max_iter"]),
        american=True,
        time_scheme="cn",
        lcp_solver="psor",
    )


def run_screen(
    config: Mapping[str, Any],
    *,
    provenance: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if config.get("method_id") != "M05" or config.get("phase") != "screen":
        raise ValueError("M05 standalone runner requires method_id=M05 and phase=screen")
    started = time.perf_counter()
    solver = dict(config["solver"])
    experiment = dict(config["experiment"])
    gates = dict(config["gates"])
    multiplier = int(solver["fine_multiplier"])
    if multiplier < 2:
        raise ValueError("fine_multiplier must be at least two")

    medium_cfg = _solver_config(solver)
    fine_cfg = _solver_config(solver, multiplier=multiplier)
    oracle_started = time.perf_counter()
    medium = solve_option(medium_cfg)
    fine = solve_option(fine_cfg)
    oracle_seconds = time.perf_counter() - oracle_started
    if not all(medium.lcp_converged) or not all(fine.lcp_converged):
        raise RuntimeError("CN/PSOR oracle did not converge")

    step = min(medium_cfg.n, max(1, round(float(experiment["time_fraction"]) * medium_cfg.n)))
    fine_step = step * multiplier
    fine_on_medium = np.interp(medium.spots, fine.spots, fine.values[fine_step])
    discretization_floor = float(
        np.sqrt(np.mean((medium.values[step, 1:-1] - fine_on_medium[1:-1]) ** 2))
        / medium_cfg.k
    )
    if not np.isfinite(discretization_floor) or discretization_floor <= 0.0:
        raise RuntimeError("classical discretization floor must be positive and finite")

    matrix, rhs, obstacle, previous = _cn_problem(medium_cfg, medium.values, step)
    target = medium.values[step, 1:-1]
    contact_tolerance = max(10.0 * medium_cfg.tol, 1.0e-9)
    frozen_exercise = target - obstacle <= contact_tolerance
    previous_residual = matrix @ previous - rhs
    live_exercise = previous - obstacle <= previous_residual

    rows: list[dict[str, Any]] = []
    for variant, exercise in (
        ("frozen_continuation", frozen_exercise),
        ("one_live_obstacle_update", live_exercise),
    ):
        policy_matrix, policy_rhs = _policy_system(matrix, rhs, obstacle, exercise)
        direct_started = time.perf_counter()
        direct = np.linalg.solve(policy_matrix, policy_rhs)
        direct_seconds = time.perf_counter() - direct_started
        proxy_started = time.perf_counter()
        candidate, diagnostics = dissipative_recovery(
            policy_matrix,
            policy_rhs,
            previous,
            decay_tolerance=float(experiment["decay_tolerance"]),
        )
        proxy_seconds = time.perf_counter() - proxy_started
        if variant == "one_live_obstacle_update":
            direct = np.maximum(direct, obstacle)
            candidate = np.maximum(candidate, obstacle)
        inner_rmse = float(np.sqrt(np.mean((candidate - direct) ** 2)) / medium_cfg.k)
        obstacle_violation = float(np.max(np.maximum(obstacle - candidate, 0.0)) / medium_cfg.k)
        candidate_active = candidate - obstacle <= contact_tolerance
        direct_active = direct - obstacle <= contact_tolerance
        mask_mismatch = float(np.mean(candidate_active != direct_active))
        numerical_gate = (
            inner_rmse <= discretization_floor
            and obstacle_violation <= float(gates["obstacle_tolerance"])
        )
        rows.append(
            {
                "variant": variant,
                "grid_m": medium_cfg.m,
                "time_step": step,
                "inner_rmse": inner_rmse,
                "classical_discretization_floor": discretization_floor,
                "error_to_floor_ratio": inner_rmse / discretization_floor,
                "obstacle_violation": obstacle_violation,
                "lcp_natural_residual": lcp_natural_residual(matrix, candidate, rhs, obstacle),
                "active_set_mismatch_fraction": mask_mismatch,
                "policy_exercise_nodes": int(np.sum(exercise)),
                "horizon": diagnostics["horizon"],
                "normal_condition_number": diagnostics["normal_condition_number"],
                "slow_mode_decay": diagnostics["slow_mode_decay"],
                "direct_solve_seconds": direct_seconds,
                "proxy_recovery_seconds": proxy_seconds,
                "numerical_gate_passed": numerical_gate,
            }
        )

    gate_passed = all(bool(row["numerical_gate_passed"]) for row in rows)
    provenance = dict(
        provenance
        or {
            "branch": _git_value("branch", "--show-current") or "detached",
            "commit": _git_value("rev-parse", "HEAD"),
        }
    )
    output = Path(config["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    resolved_config = json.loads(json.dumps(dict(config)))
    resolved_config["blind_holdout_accessed"] = False
    manifest = {
        "method_id": "M05",
        "method_name": "Schrodingerization",
        "method_scope": "classical dissipative-recovery proxy; no quantum circuit or speedup claim",
        "config": resolved_config,
        "provenance": provenance,
        "blind_holdout_accessed": False,
    }
    decision = {
        "method_id": "M05",
        "status": "diagnostic_only",
        "claim_status": "diagnostic_only",
        "numerical_gate_passed": gate_passed,
        "reason": (
            "Both bounded inner-solve diagnostics are below the classical discretization floor."
            if gate_passed
            else "At least one bounded inner-solve diagnostic exceeds its numerical gate."
        ),
        "scientific_limit": "No American obstacle Schrödingerization or quantum advantage is established.",
        "blind_holdout_accessed": False,
        "total_seconds": time.perf_counter() - started,
        "oracle_seconds": oracle_seconds,
    }
    _write_json(output / "config_manifest.json", manifest)
    _write_csv(output / "metrics.csv", rows)
    _write_json(output / "decision.json", decision)
    files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _write_json(output / "artifact_hashes.json", {"algorithm": "sha256", "files": files})
    return {**decision, "metrics": rows, "output_root": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output_root:
        config["output_root"] = args.output_root
    print(json.dumps(run_screen(config), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
