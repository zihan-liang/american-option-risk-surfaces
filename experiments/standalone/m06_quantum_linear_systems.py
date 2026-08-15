#!/usr/bin/env python3
"""M06: exact-statevector VQLS and QSVT resource audit on real CN systems."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import numpy as np

from common import load_config
from solvers.cn_psor import SolverConfig, cn_matrices, solve_option, time_boundary_values


def _apply_ry(state: np.ndarray, angle: float, qubit: int) -> None:
    stride = 1 << qubit
    cosine = np.cos(0.5 * angle)
    sine = np.sin(0.5 * angle)
    for start in range(0, len(state), 2 * stride):
        for offset in range(stride):
            zero = start + offset
            one = zero + stride
            first, second = state[zero], state[one]
            state[zero] = cosine * first - sine * second
            state[one] = sine * first + cosine * second


def hardware_efficient_state(
    parameters: np.ndarray,
    *,
    qubits: int,
    layers: int,
) -> np.ndarray:
    parameters = np.asarray(parameters, dtype=float)
    if qubits < 1 or layers < 1 or parameters.shape != (qubits * layers,):
        raise ValueError("parameters must have qubits*layers entries")
    state = np.zeros(1 << qubits, dtype=float)
    state[0] = 1.0
    for layer in range(layers):
        for qubit in range(qubits):
            _apply_ry(state, parameters[layer * qubits + qubit], qubit)
        if qubits > 1:
            edges = [(qubit, qubit + 1) for qubit in range(qubits - 1)]
            if qubits > 2:
                edges.append((qubits - 1, 0))
            for left, right in edges:
                for index in range(len(state)):
                    if ((index >> left) & 1) and ((index >> right) & 1):
                        state[index] *= -1.0
    return state


def _vqls_objective(
    parameters: np.ndarray,
    matrix: np.ndarray,
    normalized_rhs: np.ndarray,
    *,
    qubits: int,
    layers: int,
) -> float:
    state = hardware_efficient_state(parameters, qubits=qubits, layers=layers)
    transformed = matrix @ state
    denominator = float(transformed @ transformed)
    if denominator <= np.finfo(float).tiny:
        return 1.0
    overlap = float(normalized_rhs @ transformed)
    return float(max(0.0, min(1.0, 1.0 - overlap * overlap / denominator)))


def vqls_solve(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    seed: int,
    layers: int,
    iterations: int,
    learning_rate: float,
    finite_difference_step: float,
    cost_tolerance: float,
) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    dimension = len(rhs)
    qubits = int(round(np.log2(dimension)))
    if matrix.shape != (dimension, dimension) or (1 << qubits) != dimension:
        raise ValueError("VQLS requires a square power-of-two system")
    if not np.isfinite(matrix).all() or not np.isfinite(rhs).all() or np.linalg.norm(rhs) == 0.0:
        raise ValueError("matrix and nonzero rhs must be finite")

    normalized_rhs = rhs / np.linalg.norm(rhs)
    rng = np.random.default_rng(seed)
    parameters = rng.uniform(-0.25, 0.25, size=qubits * layers)
    first_moment = np.zeros_like(parameters)
    second_moment = np.zeros_like(parameters)
    best_parameters = parameters.copy()
    best_cost = _vqls_objective(
        parameters, matrix, normalized_rhs, qubits=qubits, layers=layers
    )
    evaluations = 1
    completed_iterations = 0

    for iteration in range(1, iterations + 1):
        gradient = np.zeros_like(parameters)
        for index in range(len(parameters)):
            plus = parameters.copy()
            minus = parameters.copy()
            plus[index] += finite_difference_step
            minus[index] -= finite_difference_step
            gradient[index] = (
                _vqls_objective(plus, matrix, normalized_rhs, qubits=qubits, layers=layers)
                - _vqls_objective(minus, matrix, normalized_rhs, qubits=qubits, layers=layers)
            ) / (2.0 * finite_difference_step)
        evaluations += 2 * len(parameters)
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient * gradient
        corrected_first = first_moment / (1.0 - 0.9**iteration)
        corrected_second = second_moment / (1.0 - 0.999**iteration)
        parameters -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1.0e-8)
        cost = _vqls_objective(
            parameters, matrix, normalized_rhs, qubits=qubits, layers=layers
        )
        evaluations += 1
        completed_iterations = iteration
        if cost < best_cost:
            best_cost = cost
            best_parameters = parameters.copy()
        if best_cost <= cost_tolerance:
            break

    state = hardware_efficient_state(best_parameters, qubits=qubits, layers=layers)
    transformed = matrix @ state
    scale = float((transformed @ rhs) / (transformed @ transformed))
    solution = scale * state
    exact = np.linalg.solve(matrix, rhs)
    exact_norm = float(np.linalg.norm(exact))
    normalized_exact = exact / exact_norm
    return {
        "solution": solution,
        "final_cost": float(best_cost),
        "normalized_residual": float(np.linalg.norm(matrix @ solution - rhs) / np.linalg.norm(rhs)),
        "relative_l2_error": float(np.linalg.norm(solution - exact) / exact_norm),
        "relative_linf_error": float(np.max(np.abs(solution - exact)) / np.max(np.abs(exact))),
        "solution_rmse": float(np.sqrt(np.mean((solution - exact) ** 2))),
        "state_fidelity": float(abs(normalized_exact @ state) ** 2),
        "optimizer_iterations": completed_iterations,
        "objective_evaluations": evaluations,
        "ansatz_parameters": len(best_parameters),
        "recovered_scale": scale,
    }


def pauli_encoding_audit(matrix: np.ndarray, *, truncation: float) -> dict[str, float | int]:
    matrix = np.asarray(matrix, dtype=complex)
    dimension = matrix.shape[0]
    qubits = int(round(np.log2(dimension)))
    if matrix.shape != (dimension, dimension) or (1 << qubits) != dimension:
        raise ValueError("Pauli audit requires a square power-of-two matrix")
    paulis = (
        np.array([[1, 0], [0, 1]], dtype=complex),
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    )
    reconstructed = np.zeros_like(matrix)
    retained = 0
    coefficient_l1 = 0.0
    for labels in itertools.product(range(4), repeat=qubits):
        operator = np.array([[1.0 + 0.0j]])
        for label in labels:
            operator = np.kron(operator, paulis[label])
        coefficient = np.trace(operator.conj().T @ matrix) / dimension
        if abs(coefficient) > truncation:
            reconstructed += coefficient * operator
            retained += 1
            coefficient_l1 += abs(coefficient)
    error = float(np.linalg.norm(matrix - reconstructed, 2) / np.linalg.norm(matrix, 2))
    return {
        "pauli_basis_terms": 4**qubits,
        "retained_terms": retained,
        "coefficient_l1": float(coefficient_l1),
        "relative_spectral_error": error,
    }


def _boundary_source(cfg: SolverConfig, tau: float, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    source = np.zeros(cfg.m - 1)
    left, right = time_boundary_values(cfg, tau)
    source[0] = lower[0] * left
    source[-1] = upper[-1] * right
    return source


def _frozen_policy_problem(
    dimension: int,
    solver: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, float, int]:
    cfg = SolverConfig(
        option_type="put",
        k=float(solver["k"]),
        t=float(solver["t"]),
        r=float(solver["r"]),
        q=float(solver["q"]),
        sigma=float(solver["sigma"]),
        s_max=float(solver["s_max"]),
        m=dimension + 1,
        n=int(solver["time_steps"]),
        omega=1.2,
        tol=float(solver["tol"]),
        max_iter=int(solver["max_iter"]),
        american=True,
        time_scheme="cn",
        lcp_solver="psor",
    )
    result = solve_option(cfg)
    if not all(result.lcp_converged):
        raise RuntimeError(f"CN/PSOR oracle failed for N={dimension}")
    step = min(cfg.n, max(1, round(float(solver["time_fraction"]) * cfg.n)))
    _, lower, upper, matrix, rhs_matrix = cn_matrices(cfg)
    dt = cfg.t / cfg.n
    rhs = rhs_matrix @ result.values[step - 1, 1:-1]
    rhs += 0.5 * dt * (
        _boundary_source(cfg, (step - 1) * dt, lower, upper)
        + _boundary_source(cfg, step * dt, lower, upper)
    )
    obstacle = result.values[0, 1:-1]
    target = result.values[step, 1:-1]
    exercise = target - obstacle <= max(10.0 * cfg.tol, 1.0e-9)
    policy = matrix.copy()
    policy_rhs = rhs.copy()
    policy[exercise, :] = 0.0
    indices = np.flatnonzero(exercise)
    policy[indices, indices] = 1.0
    policy_rhs[exercise] = obstacle[exercise]

    fine_cfg = SolverConfig(**{**cfg.__dict__, "m": 2 * cfg.m, "n": 2 * cfg.n})
    fine = solve_option(fine_cfg)
    fine_on_coarse = np.interp(result.spots, fine.spots, fine.values[2 * step])
    floor = float(np.sqrt(np.mean((target - fine_on_coarse[1:-1]) ** 2)) / cfg.k)
    return policy, policy_rhs, floor, int(np.sum(exercise))


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


def run_screen(
    config: Mapping[str, Any],
    *,
    provenance: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if config.get("method_id") != "M06" or config.get("phase") != "screen":
        raise ValueError("M06 runner requires method_id=M06 and phase=screen")
    started = time.perf_counter()
    vqls = dict(config["vqls"])
    resources = dict(config["resources"])
    dimensions = [int(value) for value in vqls["dimensions"]]
    seeds = [int(value) for value in vqls["seeds"]]
    if any(value < 2 or value & (value - 1) for value in dimensions):
        raise ValueError("every VQLS dimension must be a power of two")
    if not seeds:
        raise ValueError("at least one seed is required")

    metric_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    dimension_passes: dict[str, bool] = {}
    for dimension in dimensions:
        matrix, rhs, cn_floor, exercise_nodes = _frozen_policy_problem(dimension, config["solver"])
        condition = float(np.linalg.cond(matrix, 2))
        encoding_started = time.perf_counter()
        audit = pauli_encoding_audit(matrix, truncation=float(resources["pauli_truncation"]))
        encoding_seconds = time.perf_counter() - encoding_started
        qsvt_degree = int(np.ceil(condition * np.log(1.0 / float(resources["target_inverse_error"]))))
        projected_shots = int(
            audit["retained_terms"] ** 2 * int(resources["projected_shots_per_expectation"])
        )
        resource_rows.append(
            {
                "dimension": dimension,
                "solution_qubits": int(np.log2(dimension)),
                "condition_number_2": condition,
                "policy_exercise_nodes": exercise_nodes,
                "pauli_basis_terms": audit["pauli_basis_terms"],
                "retained_pauli_terms": audit["retained_terms"],
                "pauli_coefficient_l1": audit["coefficient_l1"],
                "pauli_relative_spectral_error": audit["relative_spectral_error"],
                "encoding_seconds": encoding_seconds,
                "qsvt_degree_proxy": qsvt_degree,
                "qsvt_proxy_kind": "kappa_log_inverse_error_not_circuit_count",
                "state_loading_amplitudes": dimension,
                "full_readout_values": dimension,
                "projected_shots_per_expectation": int(resources["projected_shots_per_expectation"]),
                "projected_lcu_shot_executions": projected_shots,
                "classical_comparator": "tridiagonal_banded_O_N_not_dense_timing",
            }
        )
        dimension_rows = []
        for seed in seeds:
            run_started = time.perf_counter()
            result = vqls_solve(
                matrix,
                rhs,
                seed=seed,
                layers=int(vqls["layers"]),
                iterations=int(vqls["iterations"]),
                learning_rate=float(vqls["learning_rate"]),
                finite_difference_step=float(vqls["finite_difference_step"]),
                cost_tolerance=float(vqls["cost_tolerance"]),
            )
            row = {
                "dimension": dimension,
                "seed": seed,
                "normalized_linear_residual": result["normalized_residual"],
                "solution_rmse": result["solution_rmse"] / float(config["solver"]["k"]),
                "relative_l2_error": result["relative_l2_error"],
                "relative_linf_error": result["relative_linf_error"],
                "state_fidelity": result["state_fidelity"],
                "final_cost": result["final_cost"],
                "optimizer_iterations": result["optimizer_iterations"],
                "objective_evaluations": result["objective_evaluations"],
                "ansatz_parameters": result["ansatz_parameters"],
                "executed_shots": 0,
                "expectation_mode": "exact_statevector",
                "cn_discretization_floor": cn_floor,
                "run_seconds": time.perf_counter() - run_started,
            }
            metric_rows.append(row)
            dimension_rows.append(row)
        median_error = float(np.median([row["solution_rmse"] for row in dimension_rows]))
        dimension_passes[str(dimension)] = median_error <= (
            float(config["gates"]["quantum_error_fraction_of_cn_floor"]) * cn_floor
        )

    gate_passed = all(dimension_passes.values())
    provenance = dict(
        provenance
        or {
            "branch": _git_value("branch", "--show-current") or "detached",
            "commit": _git_value("rev-parse", "HEAD"),
        }
    )
    output = Path(config["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method_id": "M06",
        "method_name": "Quantum linear systems",
        "method_scope": "exact-statevector VQLS diagnostic and QSVT resource proxy; no speed claim",
        "config": json.loads(json.dumps(dict(config))),
        "provenance": provenance,
        "blind_holdout_accessed": False,
    }
    decision = {
        "method_id": "M06",
        "status": "diagnostic_only",
        "claim_status": "diagnostic_only",
        "numerical_gate_passed": gate_passed,
        "dimension_gate_passed": dimension_passes,
        "reason": "VQLS and QSVT outputs are diagnostic microbenchmarks, not end-to-end quantum evidence.",
        "scientific_limit": "No QPU, finite-shot optimization, obstacle iteration, tomography, or speed comparison was executed.",
        "blind_holdout_accessed": False,
        "total_seconds": time.perf_counter() - started,
    }
    _write_json(output / "config_manifest.json", manifest)
    _write_csv(output / "metrics.csv", metric_rows)
    _write_csv(output / "resource_accounting.csv", resource_rows)
    _write_json(output / "decision.json", decision)
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _write_json(output / "artifact_hashes.json", {"algorithm": "sha256", "files": hashes})
    return {
        **decision,
        "metrics": metric_rows,
        "resources": resource_rows,
        "output_root": str(output),
    }


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
