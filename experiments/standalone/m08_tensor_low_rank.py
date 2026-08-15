#!/usr/bin/env python3
"""M08 tensor/low-rank oracle-compressibility and completion screen."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from data.splits import three_way_regime_split
from solvers.boundaries import boundary_curve
from solvers.cn_psor import SolverConfig, fd_operator_matrix, solve_option, time_boundary_values
from solvers.greeks import delta_gamma

from m08_tensor_core import (
    TTCompletionRepresentation,
    classify_status,
    configuration_passes,
    fit_tt_completion,
    pod_compress,
    pod_reconstruct,
    primal_dual_greedy_basis,
    primal_dual_reconstruct,
    select_observed_regimes,
    tt_rank_vector,
    tt_reconstruct,
    tt_svd,
    tt_storage,
    validate_config,
)

ORDINARY_METRICS = (
    "price_rmse",
    "delta_rmse",
    "gamma_rmse",
    "boundary_rmse",
    "obstacle_violation",
)
RESIDUAL_METRICS = (
    "dual_violation",
    "complementarity",
    "continuation_residual",
)
ALL_GATE_METRICS = ORDINARY_METRICS + RESIDUAL_METRICS


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("metrics rows cannot be empty")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _linspace(spec: Mapping[str, Any]) -> np.ndarray:
    return np.linspace(float(spec["min"]), float(spec["max"]), int(spec["count"]), dtype=float)


def build_regimes(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    grid = config["parameter_grid"]
    sigmas, rates, dividends = (_linspace(grid[name]) for name in ("sigma", "r", "q"))
    regimes: list[dict[str, Any]] = []
    flat = 0
    for i_sigma, sigma in enumerate(sigmas):
        for i_r, rate in enumerate(rates):
            for i_q, dividend in enumerate(dividends):
                regimes.append(
                    {
                        "id": f"regime_{flat:04d}",
                        "flat_index": flat,
                        "i_sigma": i_sigma,
                        "i_r": i_r,
                        "i_q": i_q,
                        "sigma": float(sigma),
                        "r": float(rate),
                        "q": float(dividend),
                    }
                )
                flat += 1
    return regimes, (sigmas, rates, dividends)


def solver_config(base: Mapping[str, Any], regime: Mapping[str, Any]) -> SolverConfig:
    kwargs = dict(base)
    kwargs.update({"sigma": float(regime["sigma"]), "r": float(regime["r"]), "q": float(regime["q"])})
    return SolverConfig(**kwargs)


def build_value_tensor(
    base: Mapping[str, Any],
    regimes: list[dict[str, Any]],
    parameter_shape: tuple[int, int, int],
) -> tuple[np.ndarray, dict[int, Any], float]:
    started = time.perf_counter()
    surfaces: dict[int, Any] = {}
    tensor: np.ndarray | None = None
    for regime in regimes:
        result = solve_option(solver_config(base, regime))
        if not np.isfinite(result.values).all() or not all(result.lcp_converged):
            raise RuntimeError(f"CN/PSOR failed for {regime['id']}")
        if tensor is None:
            tensor = np.empty(result.values.shape + parameter_shape, dtype=float)
        tensor[:, :, regime["i_sigma"], regime["i_r"], regime["i_q"]] = result.values
        surfaces[int(regime["flat_index"])] = result
    assert tensor is not None
    return tensor, surfaces, float(time.perf_counter() - started)


def _operator_coefficients(cfg: SolverConfig, spots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ds = float(spots[1] - spots[0])
    interior = spots[1:-1]
    diffusion = 0.5 * cfg.sigma**2 * interior**2
    drift = (cfg.r - cfg.q) * interior
    lower = diffusion / ds**2 - drift / (2.0 * ds)
    upper = diffusion / ds**2 + drift / (2.0 * ds)
    return lower, upper


def cn_residual_fields(values: np.ndarray, cfg: SolverConfig, spots: np.ndarray, taus: np.ndarray, payoff: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate the exact repository CN obstacle stencil on a reconstructed surface."""
    if values.shape != (cfg.n + 1, cfg.m + 1):
        raise ValueError("candidate surface shape does not match solver config")
    l_mat = fd_operator_matrix(cfg, spots)
    lower, upper = _operator_coefficients(cfg, spots)
    eye = np.eye(cfg.m - 1)
    obstacle = payoff[1:-1]
    residual = np.empty((cfg.n, cfg.m - 1), dtype=float)
    gap = values[1:, 1:-1] - obstacle[None, :]

    def boundary_source(tau: float) -> np.ndarray:
        source = np.zeros(cfg.m - 1, dtype=float)
        left, right = time_boundary_values(cfg, tau)
        source[0] = lower[0] * left
        source[-1] = upper[-1] * right
        return source

    for step in range(cfg.n):
        dt = float(taus[step + 1] - taus[step])
        a = eye - 0.5 * dt * l_mat
        b = eye + 0.5 * dt * l_mat
        rhs = b @ values[step, 1:-1]
        rhs += 0.5 * dt * (boundary_source(float(taus[step])) + boundary_source(float(taus[step + 1])))
        residual[step] = a @ values[step + 1, 1:-1] - rhs
    reference_continuation = gap > 1e-8
    return {"residual": residual, "gap": gap, "continuation": reference_continuation}


def surface_diagnostics(
    candidate: np.ndarray,
    reference_base: np.ndarray,
    reference_refined: np.ndarray,
    base_result: Any,
    refined_result: Any,
    cfg: SolverConfig,
    *,
    contact_tol: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return candidate and base numerical errors against the same refined oracle."""
    refined_on_base = interpolate_surface(refined_result.values, refined_result.spots, refined_result.taus, base_result.spots, base_result.taus)
    base_delta, base_gamma = delta_gamma(reference_base, base_result.spots)
    candidate_delta, candidate_gamma = delta_gamma(candidate, base_result.spots)
    refined_delta, refined_gamma = delta_gamma(refined_on_base, base_result.spots)
    interior = (slice(None), slice(1, -1))

    payoff = base_result.payoff
    base_boundary = boundary_curve(base_result.spots, reference_base, payoff, "put", tol=contact_tol)
    candidate_boundary = boundary_curve(base_result.spots, candidate, payoff, "put", tol=contact_tol)
    refined_boundary_native = boundary_curve(refined_result.spots, refined_result.values, refined_result.payoff, "put", tol=contact_tol)
    refined_boundary = np.interp(base_result.taus, refined_result.taus, refined_boundary_native)
    boundary_mask = np.isfinite(base_boundary) & np.isfinite(candidate_boundary) & np.isfinite(refined_boundary) & (base_result.taus > 0)

    base_fields = cn_residual_fields(reference_base, cfg, base_result.spots, base_result.taus, payoff)
    candidate_fields = cn_residual_fields(candidate, cfg, base_result.spots, base_result.taus, payoff)
    reference_continuation = base_fields["gap"] > contact_tol

    def rmse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

    def lcp_metrics(fields: dict[str, np.ndarray]) -> tuple[float, float, float, float]:
        gap = fields["gap"]
        residual = fields["residual"]
        obstacle = float(np.max(np.maximum(-gap, 0.0)))
        dual = float(np.max(np.maximum(-residual, 0.0)))
        complementarity = float(np.max(np.abs(gap * residual)))
        continuation = float(np.sqrt(np.mean(residual[reference_continuation] ** 2))) if np.any(reference_continuation) else 0.0
        return obstacle, dual, complementarity, continuation

    base_lcp = lcp_metrics(base_fields)
    candidate_lcp = lcp_metrics(candidate_fields)
    if np.any(boundary_mask):
        base_boundary_rmse = rmse(base_boundary[boundary_mask], refined_boundary[boundary_mask])
        candidate_boundary_rmse = rmse(candidate_boundary[boundary_mask], refined_boundary[boundary_mask])
    else:
        base_boundary_rmse = candidate_boundary_rmse = 0.0

    baseline = {
        "price_rmse": rmse(reference_base, refined_on_base),
        "delta_rmse": rmse(base_delta[interior], refined_delta[interior]),
        "gamma_rmse": rmse(base_gamma[interior], refined_gamma[interior]),
        "boundary_rmse": base_boundary_rmse,
        "obstacle_violation": base_lcp[0],
        "dual_violation": base_lcp[1],
        "complementarity": base_lcp[2],
        "continuation_residual": base_lcp[3],
    }
    metrics = {
        "price_rmse": rmse(candidate, refined_on_base),
        "delta_rmse": rmse(candidate_delta[interior], refined_delta[interior]),
        "gamma_rmse": rmse(candidate_gamma[interior], refined_gamma[interior]),
        "boundary_rmse": candidate_boundary_rmse,
        "obstacle_violation": candidate_lcp[0],
        "dual_violation": candidate_lcp[1],
        "complementarity": candidate_lcp[2],
        "continuation_residual": candidate_lcp[3],
    }
    return metrics, baseline


def interpolate_surface(values: np.ndarray, spots: np.ndarray, taus: np.ndarray, target_spots: np.ndarray, target_taus: np.ndarray) -> np.ndarray:
    spot_interp = np.vstack([np.interp(target_spots, spots, row) for row in values])
    out = np.empty((len(target_taus), len(target_spots)), dtype=float)
    for column in range(len(target_spots)):
        out[:, column] = np.interp(target_taus, taus, spot_interp[:, column])
    return out


def _bootstrap_upper(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    floor: float,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    candidate = np.asarray(candidate, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if candidate.shape != baseline.shape or candidate.ndim != 1 or candidate.size == 0:
        raise ValueError("bootstrap inputs must be paired nonempty regime vectors")
    denom = np.maximum(np.abs(baseline), float(floor))
    paired = 100.0 * (candidate - baseline) / denom
    point = float(np.mean(paired))
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(repetitions), dtype=float)
    for index in range(int(repetitions)):
        sample = rng.integers(0, paired.size, size=paired.size)
        means[index] = float(np.mean(paired[sample]))
    return point, float(np.quantile(means, 0.975))


def _fit_representation(config_item: Mapping[str, Any], tensor: np.ndarray, train_indices: np.ndarray, seed: int, completion_cfg: Mapping[str, Any]) -> tuple[Any, np.ndarray, dict[str, Any]]:
    family = str(config_item["family"])
    rank = int(config_item["rank"])
    started = time.perf_counter()
    metadata: dict[str, Any] = {"family": family, "rank": rank}
    if family == "tt_svd":
        rep = tt_svd(tensor, rank)
        reconstructed = tt_reconstruct(rep)
        stored = tt_storage(rep)
        metadata.update({"tt_ranks": tt_rank_vector(rep), "oracle_complete_tensor_access": True})
    elif family == "pod":
        rep = pod_compress(tensor, state_modes=2, rank=rank)
        reconstructed = pod_reconstruct(rep)
        stored = rep.stored_scalars
        metadata.update({"oracle_complete_tensor_access": True})
    elif family == "primal_dual_rb":
        matrix = tensor.reshape(tensor.shape[0] * tensor.shape[1], -1)
        # Dual/contact proxy: positive continuation residual is built from the gap support.
        payoff = np.maximum(1.0 - np.linspace(0.0, 4.0, tensor.shape[1]), 0.0)
        gap = tensor - payoff.reshape(1, -1, 1, 1, 1)
        dual = np.maximum(-gap, 0.0).reshape(matrix.shape)
        rep = primal_dual_greedy_basis(matrix, dual, rank=rank)
        primal, _ = primal_dual_reconstruct(rep)
        reconstructed = primal.reshape(tensor.shape)
        stored = rep.stored_scalars
        metadata.update({"selected_regimes": list(rep.selected_regimes), "oracle_complete_tensor_access": True, "claim_scope": "primal_dual_compression_baseline_not_reduced_vi_solver"})
    elif family == "tt_completion":
        observed = select_observed_regimes(train_indices, fraction=float(config_item["observed_fraction"]), seed=seed)
        rep, history = fit_tt_completion(
            tensor,
            state_modes=2,
            rank=rank,
            observed_regime_indices=observed,
            seed=seed,
            steps=int(completion_cfg["steps"]),
            batch_size=int(completion_cfg["batch_size"]),
            learning_rate=float(completion_cfg["learning_rate"]),
        )
        assert isinstance(rep, TTCompletionRepresentation)
        reconstructed = tt_reconstruct(rep.cores)
        stored = rep.stored_scalars
        metadata.update({"tt_ranks": tt_rank_vector(rep.cores), "observed_regime_count": len(rep.observed_regime_indices), "observed_fraction": float(config_item["observed_fraction"]), "audit_history": history, "oracle_complete_tensor_access": False})
    else:
        raise ValueError(f"unknown M08 family: {family}")
    metadata.update(
        {
            "stored_scalars": int(stored),
            "compression_ratio": float(tensor.size / max(int(stored), 1)),
            "reconstruction_seconds": float(time.perf_counter() - started),
        }
    )
    return rep, reconstructed, metadata


def _candidate_surface(tensor: np.ndarray, regime: Mapping[str, Any]) -> np.ndarray:
    return tensor[:, :, int(regime["i_sigma"]), int(regime["i_r"]), int(regime["i_q"])]


def _refined_result(base_cfg: Mapping[str, Any], refined_cfg: Mapping[str, Any], regime: Mapping[str, Any]) -> Any:
    merged = dict(base_cfg)
    merged.update(refined_cfg)
    return solve_option(solver_config(merged, regime))


def evaluate_partition(
    reconstructed: np.ndarray,
    regimes: list[dict[str, Any]],
    base_surfaces: Mapping[int, Any],
    base_cfg: Mapping[str, Any],
    refined_cfg: Mapping[str, Any],
    diagnostics_cfg: Mapping[str, Any],
    bootstrap_cfg: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, list[float]]]:
    candidate_by_metric = {name: [] for name in ALL_GATE_METRICS}
    baseline_by_metric = {name: [] for name in ALL_GATE_METRICS}
    for regime in regimes:
        base = base_surfaces[int(regime["flat_index"])]
        refined = _refined_result(base_cfg, refined_cfg, regime)
        candidate = _candidate_surface(reconstructed, regime)
        metrics, baseline = surface_diagnostics(
            candidate,
            base.values,
            refined.values,
            base,
            refined,
            solver_config(base_cfg, regime),
            contact_tol=float(diagnostics_cfg["contact_tol"]),
        )
        for name in ALL_GATE_METRICS:
            candidate_by_metric[name].append(float(metrics[name]))
            baseline_by_metric[name].append(float(baseline[name]))

    summary: dict[str, float] = {}
    floors = {
        "price_rmse": float(diagnostics_cfg["ordinary_floor"]),
        "delta_rmse": float(diagnostics_cfg["ordinary_floor"]),
        "gamma_rmse": float(diagnostics_cfg["ordinary_floor"]),
        "boundary_rmse": float(diagnostics_cfg["ordinary_floor"]),
        "obstacle_violation": float(diagnostics_cfg["obstacle_floor"]),
        "dual_violation": float(diagnostics_cfg["residual_floor"]),
        "complementarity": float(diagnostics_cfg["residual_floor"]),
        "continuation_residual": float(diagnostics_cfg["residual_floor"]),
    }
    for offset, name in enumerate(ALL_GATE_METRICS):
        candidate = np.asarray(candidate_by_metric[name])
        baseline = np.asarray(baseline_by_metric[name])
        point, upper = _bootstrap_upper(
            candidate,
            baseline,
            floor=floors[name],
            repetitions=int(bootstrap_cfg["repetitions"]),
            seed=int(bootstrap_cfg["seed"]) + offset,
        )
        summary[name] = float(np.mean(candidate))
        summary[f"baseline_{name}"] = float(np.mean(baseline))
        summary[f"degradation_pct_{name}"] = point
        summary[f"degradation_upper_pct_{name}"] = upper
    return summary, {name: candidate_by_metric[name] for name in ALL_GATE_METRICS}


def _gate_upper(summary: Mapping[str, float]) -> dict[str, float]:
    return {name: float(summary[f"degradation_upper_pct_{name}"]) for name in ALL_GATE_METRICS}


def _gate_ratio(summary: Mapping[str, float], compression_ratio: float, gates: Mapping[str, Any]) -> float:
    ratios = [float(gates["compression_ratio_min"]) / max(float(compression_ratio), 1e-15)]
    for name in ORDINARY_METRICS:
        ratios.append(max(0.0, float(summary[f"degradation_upper_pct_{name}"])) / float(gates["ordinary_degradation_pct_max"]))
    for name in RESIDUAL_METRICS:
        ratios.append(max(0.0, float(summary[f"degradation_upper_pct_{name}"])) / float(gates["residual_degradation_pct_max"]))
    return float(max(ratios))


def _smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    smoke = json.loads(json.dumps(config))
    smoke["evidence_mode"] = "smoke"
    smoke["base_solver"].update({"m": 24, "n": 12, "t": 1.0})
    smoke["refined_solver"] = {"m": 36, "n": 18}
    smoke["parameter_grid"] = {
        "sigma": {"min": 0.15, "max": 0.35, "count": 3},
        "r": {"min": 0.02, "max": 0.08, "count": 3},
        "q": {"min": 0.00, "max": 0.04, "count": 2},
    }
    smoke["completion"].update({"steps": 20, "batch_size": 128})
    smoke["bootstrap"]["repetitions"] = 50
    return smoke


def run(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    started = time.perf_counter()
    regimes, axes = build_regimes(config)
    split = three_way_regime_split(
        regimes,
        train_fraction=float(config["split"]["train_fraction"]),
        validation_fraction=float(config["split"]["validation_fraction"]),
        seed=int(config["split"]["seed"]),
    )
    train_indices = np.asarray([regime["flat_index"] for regime in split["train"]], dtype=int)
    tensor, base_surfaces, label_seconds = build_value_tensor(
        config["base_solver"], regimes, tuple(len(axis) for axis in axes)
    )
    if time.perf_counter() - started > float(config["budget_seconds"]):
        raise TimeoutError("M08 screen budget exceeded during label generation")

    fitted: dict[str, tuple[Any, np.ndarray, dict[str, Any]]] = {}
    validation_rows: list[dict[str, Any]] = []
    seeds = list(config["tt_completion_seeds"])
    for config_index, item in enumerate(config["configurations"]):
        seed = int(seeds[config_index % len(seeds)])
        rep, reconstructed, metadata = _fit_representation(item, tensor, train_indices, seed, config["completion"])
        validation, _ = evaluate_partition(
            reconstructed,
            split["validation"],
            base_surfaces,
            config["base_solver"],
            config["refined_solver"],
            config["diagnostics"],
            config["bootstrap"],
        )
        ratio = _gate_ratio(validation, metadata["compression_ratio"], config["gates"])
        row = {
            "partition": "validation",
            "config_id": item["id"],
            "family": item["family"],
            "rank": item["rank"],
            "stored_scalars": metadata["stored_scalars"],
            "compression_ratio": metadata["compression_ratio"],
            "reconstruction_seconds": metadata["reconstruction_seconds"],
            "gate_ratio": ratio,
            "oracle_complete_tensor_access": metadata["oracle_complete_tensor_access"],
            **validation,
        }
        validation_rows.append(row)
        fitted[str(item["id"])] = (rep, reconstructed, metadata)
        if time.perf_counter() - started > float(config["budget_seconds"]):
            raise TimeoutError("M08 screen budget exceeded during validation")

    frozen: dict[str, str] = {}
    for family in ("tt_svd", "tt_completion", "pod", "primal_dual_rb"):
        family_rows = [row for row in validation_rows if row["family"] == family]
        winner = min(family_rows, key=lambda row: (float(row["gate_ratio"]), -float(row["compression_ratio"]), str(row["config_id"])))
        frozen[family] = str(winner["config_id"])

    test_rows: list[dict[str, Any]] = []
    passing: list[str] = []
    for family, config_id in frozen.items():
        item = next(item for item in config["configurations"] if item["id"] == config_id)
        _, reconstructed, metadata = fitted[config_id]
        test_summary, _ = evaluate_partition(
            reconstructed,
            split["test"],
            base_surfaces,
            config["base_solver"],
            config["refined_solver"],
            config["diagnostics"],
            config["bootstrap"],
        )
        passed = configuration_passes(
            float(metadata["compression_ratio"]),
            _gate_upper(test_summary),
            ordinary_limit_pct=float(config["gates"]["ordinary_degradation_pct_max"]),
            residual_limit_pct=float(config["gates"]["residual_degradation_pct_max"]),
        )
        if passed:
            passing.append(config_id)
        test_rows.append(
            {
                "partition": "test",
                "config_id": config_id,
                "family": family,
                "rank": item["rank"],
                "stored_scalars": metadata["stored_scalars"],
                "compression_ratio": metadata["compression_ratio"],
                "reconstruction_seconds": metadata["reconstruction_seconds"],
                "gate_ratio": _gate_ratio(test_summary, metadata["compression_ratio"], config["gates"]),
                "gate_passed": passed,
                "oracle_complete_tensor_access": metadata["oracle_complete_tensor_access"],
                **test_summary,
            }
        )

    elapsed = float(time.perf_counter() - started)
    if elapsed > float(config["budget_seconds"]):
        raise TimeoutError("M08 canonical screen exceeded one-hour budget")
    status = classify_status(str(config["evidence_mode"]), any_configuration_passed=bool(passing))
    return {
        "tensor": tensor,
        "rows": validation_rows + test_rows,
        "frozen": frozen,
        "passing": passing,
        "status": status,
        "total_seconds": elapsed,
        "label_seconds": label_seconds,
        "split_ids": {name: [regime["id"] for regime in values] for name, values in split.items()},
        "axes": {"sigma": axes[0].tolist(), "r": axes[1].tolist(), "q": axes[2].tolist()},
        "metadata": {config_id: fitted[config_id][2] for config_id in fitted},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    if args.smoke:
        config = _smoke_config(config)
    validate_config(config)
    output = Path(config["output_root"])
    output.mkdir(parents=True, exist_ok=True)

    try:
        result = run(config)
        status = result["status"]
        reason = (
            "At least one frozen low-rank configuration passed the 10x/2%/5% gate."
            if result["passing"]
            else "No frozen low-rank configuration passed compression and all shared diagnostic gates."
        )
    except TimeoutError as exc:
        result = {"rows": [], "frozen": {}, "passing": [], "total_seconds": float(config["budget_seconds"]), "label_seconds": 0.0, "split_ids": {}, "axes": {}, "metadata": {}}
        status = "resource_blocked"
        reason = str(exc)

    implementation_sha = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    manifest = {
        "method_id": "M08",
        "phase": "screen",
        "evidence_mode": config["evidence_mode"],
        "resolved_config": config,
        "source_config": str(config_path),
        "branch": branch,
        "implementation_sha": implementation_sha,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "seeds": config["tt_completion_seeds"],
        "splits": result["split_ids"],
        "parameter_axes": result["axes"],
        "blind_holdout_accessed": False,
        "scope_limits": [
            "Classical tensor/model-reduction experiment only; no quantum hardware or QTT PDE solver.",
            "TT-SVD, POD, and primal-dual RB are oracle compressibility baselines with complete-tensor access.",
            "tt_completion alone restricts observed fitting data to complete training regimes.",
            "primal_dual_rb is a dual-aware compression baseline, not a reduced variational-inequality solver.",
            "No post-reconstruction obstacle repair is applied before diagnostics.",
        ],
    }
    decision = {
        "method_id": "M08",
        "status": status,
        "claim_status": "diagnostic_only" if status != "candidate_supported" else "candidate_supported",
        "reason": reason,
        "frozen_validation_winners": result["frozen"],
        "passing_test_configurations": result["passing"],
        "numerical_gate_passed": bool(result["passing"]),
        "blind_holdout_accessed": False,
        "total_seconds": result["total_seconds"],
        "label_generation_seconds": result["label_seconds"],
        "limitations": manifest["scope_limits"],
    }
    _json(output / "config_manifest.json", manifest)
    if result["rows"]:
        _csv(output / "metrics.csv", result["rows"])
    else:
        _csv(output / "metrics.csv", [{"partition": "none", "status": status}])
    _json(output / "decision.json", decision)
    hashes = {
        name: _sha256(output / name)
        for name in ("config_manifest.json", "metrics.csv", "decision.json")
    }
    _json(output / "artifact_hashes.json", hashes)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
