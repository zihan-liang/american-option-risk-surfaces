"""Independent M03 learned feasible warm-start screening campaign."""

from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from campaign.io import write_method_outputs
from campaign.records import MethodRunRecord, RunPhase, RunStatus
from campaign.registry import get_method_spec
from campaign.statistics import measure_timing, summarize_paired_scenarios
from models.algorithm_unrolling import (
    FeatureNormalizer,
    SurfaceSolveResult,
    UnrolledOmegaResult,
    build_warm_start_mlp,
    extract_lcp_training_data,
    solve_cn_lcp_surface,
    surface_training_examples,
    train_unrolled_omegas,
)
from solvers.boundaries import boundary_curve
from solvers.cn_psor import SolverConfig, cn_matrices, time_boundary_values
from solvers.greeks import delta_gamma, greek_diagnostic_masks


REQUIRED_SEEDS = (101, 203, 307, 409, 503)


DEFAULT_CONFIG: dict[str, Any] = {
    "method_id": "M03",
    "run_mode": "canonical_screen",
    "sampling": {
        "generation_seed": 1701,
        "train_regimes": 20,
        "validation_regimes": 10,
        "parameter_ranges": {
            "t": [0.25, 2.0],
            "r": [0.0, 0.10],
            "q": [0.0, 0.06],
            "sigma": [0.10, 0.60],
        },
    },
    "solver": {
        "m": 76,
        "n": 100,
        "k": 1.0,
        "s_max": 4.0,
        "residual_tolerance": 1.0e-10,
        "label_residual_tolerance": 1.0e-12,
        "max_iter": 20_000,
        "omega_candidates": [1.0, 1.2, 1.4, 1.6],
    },
    "network": {
        "hidden": 128,
        "depth": 3,
        "epochs": 30,
        "batch_size": 8192,
        "patience": 8,
        "learning_rate": 0.001,
        "dtype": "float64",
        "device": "cpu",
        "blend_candidates": [0.5, 1.0],
    },
    "unrolling": {
        "K_candidates": [3, 5],
        "learning_rate": 0.01,
        "epochs": 100,
    },
    "timing": {"warmups": 5, "repetitions": 30},
    "statistics": {"bootstrap_iterations": 10_000, "bootstrap_seed": 1703},
    "budget_seconds": 3_600,
    "seeds": list(REQUIRED_SEEDS),
}


SMOKE_OVERRIDES: dict[str, Any] = {
    "run_mode": "smoke_diagnostic",
    "sampling": {"train_regimes": 2, "validation_regimes": 2},
    "solver": {
        "m": 8,
        "n": 3,
        "residual_tolerance": 1.0e-9,
        "label_residual_tolerance": 1.0e-11,
        "max_iter": 2_000,
        "omega_candidates": [1.0, 1.2],
    },
    "network": {
        "hidden": 8,
        "depth": 1,
        "epochs": 1,
        "batch_size": 64,
        "patience": 1,
        "blend_candidates": [0.5, 1.0],
    },
    "unrolling": {
        "K_candidates": [2, 3],
        "learning_rate": 0.01,
        "epochs": 2,
    },
    "timing": {"warmups": 0, "repetitions": 1},
    "statistics": {"bootstrap_iterations": 200},
}


@dataclass(frozen=True)
class RegimeSplits:
    train: tuple[Mapping[str, Any], ...]
    validation: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class FixedConfigurationSelection:
    config_id: str
    seeds: tuple[int, ...]
    selected_config_by_seed: Mapping[int, str]
    mean_cleanup_iterations: float


@dataclass(frozen=True)
class TrainedWarmStart:
    seed: int
    model: Any
    predictor: Any
    training_seconds: float
    model_bytes: int
    history: tuple[Mapping[str, Any], ...]


def build_regime_splits(
    *,
    train_count: int,
    validation_count: int,
    generation_seed: int,
    parameter_ranges: Mapping[str, Sequence[float]] | None = None,
) -> RegimeSplits:
    """Generate and split whole put regimes before expanding any grid nodes."""
    if train_count < 1 or validation_count < 2:
        raise ValueError("M03 requires at least one train and two validation regimes")
    rng = np.random.default_rng(generation_seed)
    samples = rng.random((train_count + validation_count, 4))
    ranges = parameter_ranges or DEFAULT_CONFIG["sampling"]["parameter_ranges"]
    names = ("t", "r", "q", "sigma")
    regimes: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        regime: dict[str, Any] = {
            "scenario_id": f"m03-regime-{index:03d}",
            "option_type": "put",
        }
        for column, name in enumerate(names):
            lower, upper = ranges[name]
            regime[name] = float(lower + sample[column] * (upper - lower))
        regimes.append(regime)
    return RegimeSplits(
        train=tuple(regimes[:train_count]),
        validation=tuple(regimes[train_count:]),
    )


def select_fixed_configuration(
    rows: Iterable[Mapping[str, Any]], *, required_seeds: Sequence[int]
) -> FixedConfigurationSelection:
    """Select one validation configuration and apply it to every training seed."""
    expected_seeds = tuple(int(seed) for seed in required_seeds)
    if not expected_seeds or len(expected_seeds) != len(set(expected_seeds)):
        raise ValueError("required_seeds must be non-empty and unique")
    grouped: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        config_id = str(row["config_id"])
        seed = int(row["seed"])
        iterations = float(row["cleanup_iterations"])
        if not config_id or not math.isfinite(iterations) or iterations < 0.0:
            raise ValueError("selection rows require finite non-negative cleanup iterations")
        grouped.setdefault(config_id, {}).setdefault(seed, []).append(iterations)
    if not grouped:
        raise ValueError("selection rows must not be empty")

    scores: list[tuple[float, str]] = []
    for config_id, seed_values in grouped.items():
        if set(seed_values) != set(expected_seeds):
            raise ValueError(f"configuration {config_id} does not cover every required seed")
        seed_means = [float(np.mean(seed_values[seed])) for seed in expected_seeds]
        scores.append((float(np.mean(seed_means)), config_id))
    score, selected = min(scores, key=lambda item: (item[0], item[1]))
    return FixedConfigurationSelection(
        config_id=selected,
        seeds=expected_seeds,
        selected_config_by_seed={seed: selected for seed in expected_seeds},
        mean_cleanup_iterations=score,
    )


def classify_screen_decision(
    *,
    smoke: bool,
    budget_exhausted: bool,
    speed_improved: bool,
    diagnostics_pass: bool,
    required_available: bool,
) -> tuple[str, str, str, bool]:
    """Return status, claim status, reason, and promotion eligibility."""
    if smoke:
        return ("diagnostic_only", "diagnostic_only", "smoke_non_evidence", False)
    if budget_exhausted:
        return ("resource_blocked", "diagnostic_only", "screen_budget_exhausted", False)
    promotion_eligible = speed_improved and diagnostics_pass and required_available
    if promotion_eligible:
        return ("completed", "pilot_candidate", "all_promotion_gates_passed", True)
    if not diagnostics_pass or not required_available:
        return ("failed_metric_gate", "failed_metric_gate", "diagnostic_gate_failed", False)
    return ("diagnostic_only", "diagnostic_only", "speed_improvement_not_established", False)


def run_m03_screen(
    *,
    output_dir: str | Path,
    smoke: bool,
    config_overrides: Mapping[str, Any] | None = None,
    provenance: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run one M03 screen without exposing legacy-test or blind-holdout APIs."""
    started_at = perf_counter()
    config = deepcopy(DEFAULT_CONFIG)
    if smoke:
        _deep_update(config, SMOKE_OVERRIDES)
    if config_overrides:
        _deep_update(config, config_overrides)
    _validate_config(config)
    resolved_provenance = dict(provenance or _git_provenance())

    sampling = config["sampling"]
    solver_cfg = config["solver"]
    network_cfg = config["network"]
    timing_cfg = config["timing"]
    statistics_cfg = config["statistics"]
    seeds = tuple(int(seed) for seed in config["seeds"])
    splits = build_regime_splits(
        train_count=int(sampling["train_regimes"]),
        validation_count=int(sampling["validation_regimes"]),
        generation_seed=int(sampling["generation_seed"]),
        parameter_ranges=sampling["parameter_ranges"],
    )

    selection_rows: list[dict[str, Any]] = []
    omega_results: dict[tuple[float, str], SurfaceSolveResult] = {}
    baseline_selection_started = perf_counter()
    for omega in solver_cfg["omega_candidates"]:
        for regime in splits.validation:
            result = _solve_regime(
                regime,
                solver_cfg,
                tolerance=float(solver_cfg["residual_tolerance"]),
                omega=float(omega),
            )
            omega_results[(float(omega), str(regime["scenario_id"]))] = result
            selection_rows.append(
                {
                    "selection_stage": "baseline_omega",
                    "config_id": _omega_config_id(float(omega)),
                    "seed": None,
                    "scenario_id": regime["scenario_id"],
                    "cleanup_iterations": result.cleanup_iterations_total,
                    "total_seconds": result.timings_seconds["total"],
                    "natural_residual": max(result.natural_residuals),
                    "fallback_count": result.fallback_count,
                }
            )
    omega_scores = {
        float(omega): float(
            np.mean(
                [
                    omega_results[(float(omega), str(regime["scenario_id"]))].cleanup_iterations_total
                    for regime in splits.validation
                ]
            )
        )
        for omega in solver_cfg["omega_candidates"]
    }
    selected_omega = min(omega_scores, key=lambda value: (omega_scores[value], value))
    baseline_results = {
        str(regime["scenario_id"]): omega_results[(selected_omega, str(regime["scenario_id"]))]
        for regime in splits.validation
    }
    baseline_selection_seconds = float(perf_counter() - baseline_selection_started)

    label_started = perf_counter()
    train_oracles = {
        str(regime["scenario_id"]): _solve_regime(
            regime,
            solver_cfg,
            tolerance=float(solver_cfg["label_residual_tolerance"]),
            omega=selected_omega,
        )
        for regime in splits.train
    }
    validation_oracles = {
        str(regime["scenario_id"]): _solve_regime(
            regime,
            solver_cfg,
            tolerance=float(solver_cfg["label_residual_tolerance"]),
            omega=selected_omega,
        )
        for regime in splits.validation
    }
    label_generation_seconds = float(perf_counter() - label_started)

    preprocessing_started = perf_counter()
    train_features, train_targets = _collect_examples(train_oracles.values())
    validation_features, validation_targets = _collect_examples(validation_oracles.values())
    normalizer = FeatureNormalizer.fit(train_features)
    normalized_train = normalizer.transform(train_features)
    normalized_validation = normalizer.transform(validation_features)
    preprocessing_seconds = float(perf_counter() - preprocessing_started)

    trained_models: dict[int, TrainedWarmStart] = {}
    training_history: list[dict[str, Any]] = []
    for seed in seeds:
        trained = _train_model(
            seed,
            normalized_train,
            train_targets,
            normalized_validation,
            validation_targets,
            network_cfg,
            normalizer,
        )
        trained_models[seed] = trained
        training_history.extend(dict(row) for row in trained.history)

    candidate_selection_started = perf_counter()
    candidate_results: dict[tuple[int, str, str], SurfaceSolveResult] = {}
    candidate_selection_rows: list[dict[str, Any]] = []
    for blend in network_cfg["blend_candidates"]:
        config_id = _blend_config_id(float(blend))
        for seed, trained in trained_models.items():
            for regime in splits.validation:
                scenario_id = str(regime["scenario_id"])
                result = _solve_regime(
                    regime,
                    solver_cfg,
                    tolerance=float(solver_cfg["residual_tolerance"]),
                    omega=selected_omega,
                    predictor=trained.predictor,
                    blend=float(blend),
                )
                candidate_results[(seed, config_id, scenario_id)] = result
                row = {
                    "selection_stage": "candidate_blend",
                    "config_id": config_id,
                    "seed": seed,
                    "scenario_id": scenario_id,
                    "cleanup_iterations": result.cleanup_iterations_total,
                    "total_seconds": result.timings_seconds["total"],
                    "natural_residual": max(result.natural_residuals),
                    "fallback_count": result.fallback_count,
                }
                selection_rows.append(row)
                candidate_selection_rows.append(row)

    # --- Unrolled projected SOR candidate path ---
    unroll_cfg = config["unrolling"]
    unrolled_trained: dict[int, dict[int, UnrolledOmegaResult]] = {seed: {} for seed in seeds}
    train_lcp_data: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for oracle in train_oracles.values():
        train_lcp_data.extend(extract_lcp_training_data(oracle))
    unrolled_training_started = perf_counter()
    for k_value in unroll_cfg["K_candidates"]:
        for seed in seeds:
            trained_unrolled = train_unrolled_omegas(
                train_lcp_data,
                K=int(k_value),
                learning_rate=float(unroll_cfg["learning_rate"]),
                epochs=int(unroll_cfg["epochs"]),
                seed=seed,
            )
            unrolled_trained[seed][int(k_value)] = trained_unrolled
    unrolled_training_seconds = float(perf_counter() - unrolled_training_started)

    for k_value in unroll_cfg["K_candidates"]:
        config_id = f"unrolled_projected_sor_k{k_value}"
        for seed in seeds:
            omegas = unrolled_trained[seed][int(k_value)].omegas
            for regime in splits.validation:
                scenario_id = str(regime["scenario_id"])
                result = _solve_regime(
                    regime,
                    solver_cfg,
                    tolerance=float(solver_cfg["residual_tolerance"]),
                    omega=selected_omega,
                    unrolled_omegas=omegas,
                )
                candidate_results[(seed, config_id, scenario_id)] = result
                row = {
                    "selection_stage": "candidate_unrolled",
                    "config_id": config_id,
                    "seed": seed,
                    "scenario_id": scenario_id,
                    "cleanup_iterations": result.cleanup_iterations_total,
                    "total_seconds": result.timings_seconds["total"],
                    "natural_residual": max(result.natural_residuals),
                    "fallback_count": result.fallback_count,
                }
                selection_rows.append(row)
                candidate_selection_rows.append(row)

    fixed_selection = select_fixed_configuration(candidate_selection_rows, required_seeds=seeds)
    selected_is_unrolled = fixed_selection.config_id.startswith("unrolled_")
    selected_blend = 1.0 if selected_is_unrolled else _blend_from_config_id(fixed_selection.config_id)
    candidate_selection_seconds = float(perf_counter() - candidate_selection_started)
    selection_seconds = baseline_selection_seconds + candidate_selection_seconds

    scenario_rows: list[dict[str, Any]] = []
    baseline_diagnostics: dict[str, dict[str, Any]] = {}
    candidate_diagnostics: dict[int, dict[str, dict[str, Any]]] = {seed: {} for seed in seeds}
    for regime in splits.validation:
        scenario_id = str(regime["scenario_id"])
        oracle = validation_oracles[scenario_id]
        baseline_diag = _surface_diagnostics(baseline_results[scenario_id], oracle)
        baseline_diagnostics[scenario_id] = baseline_diag
        scenario_rows.append(
            _scenario_row("tuned_psor", scenario_id, None, baseline_results[scenario_id], baseline_diag)
        )
        for seed in seeds:
            candidate = candidate_results[(seed, fixed_selection.config_id, scenario_id)]
            candidate_diag = _surface_diagnostics(candidate, oracle)
            candidate_diagnostics[seed][scenario_id] = candidate_diag
            scenario_rows.append(
                _scenario_row(fixed_selection.config_id, scenario_id, seed, candidate, candidate_diag)
            )

    timing_samples: list[dict[str, Any]] = []
    baseline_times: dict[str, list[float]] = {}
    candidate_times: dict[str, list[float]] = {str(regime["scenario_id"]): [] for regime in splits.validation}
    candidate_components: dict[int, list[Mapping[str, float]]] = {seed: [] for seed in seeds}
    for regime in splits.validation:
        scenario_id = str(regime["scenario_id"])
        baseline_measurement = _measure_surface(
            lambda regime=regime: _solve_regime(
                regime,
                solver_cfg,
                tolerance=float(solver_cfg["residual_tolerance"]),
                omega=selected_omega,
            ),
            warmups=int(timing_cfg["warmups"]),
            repetitions=int(timing_cfg["repetitions"]),
        )
        baseline_times[scenario_id] = list(baseline_measurement["samples"])
        timing_samples.extend(
            _timing_rows("tuned_psor", scenario_id, None, baseline_measurement)
        )
        for seed, trained in trained_models.items():
            if selected_is_unrolled:
                k_value = int(fixed_selection.config_id.rsplit("k", 1)[-1])
                omegas = unrolled_trained[seed][k_value].omegas
                candidate_measurement = _measure_surface(
                    lambda regime=regime, omegas=omegas: _solve_regime(
                        regime,
                        solver_cfg,
                        tolerance=float(solver_cfg["residual_tolerance"]),
                        omega=selected_omega,
                        unrolled_omegas=omegas,
                    ),
                    warmups=int(timing_cfg["warmups"]),
                    repetitions=int(timing_cfg["repetitions"]),
                )
            else:
                candidate_measurement = _measure_surface(
                    lambda regime=regime, predictor=trained.predictor: _solve_regime(
                        regime,
                        solver_cfg,
                        tolerance=float(solver_cfg["residual_tolerance"]),
                        omega=selected_omega,
                        predictor=predictor,
                        blend=selected_blend,
                    ),
                    warmups=int(timing_cfg["warmups"]),
                    repetitions=int(timing_cfg["repetitions"]),
                )
            candidate_times[scenario_id].extend(candidate_measurement["samples"])
            candidate_components[seed].append(candidate_measurement["component_medians"])
            timing_samples.extend(
                _timing_rows(fixed_selection.config_id, scenario_id, seed, candidate_measurement)
            )

    paired = summarize_paired_scenarios(
        baseline_times,
        candidate_times,
        lower_is_better=True,
        bootstrap_iterations=int(statistics_cfg["bootstrap_iterations"]),
        random_seed=int(statistics_cfg["bootstrap_seed"]),
    )
    baseline_aggregate = _aggregate_diagnostics(baseline_diagnostics.values())
    candidate_seed_aggregates = {
        seed: _aggregate_diagnostics(candidate_diagnostics[seed].values()) for seed in seeds
    }
    candidate_aggregate = _aggregate_diagnostics(candidate_seed_aggregates.values())
    candidate_execution = {
        seed: [
            candidate_results[(seed, fixed_selection.config_id, str(regime["scenario_id"]))]
            for regime in splits.validation
        ]
        for seed in seeds
    }
    gates = _diagnostic_gates(
        baseline_aggregate,
        candidate_aggregate,
        residual_tolerance=float(solver_cfg["residual_tolerance"]),
    )
    speed_improved = paired.relative_percent_change >= 2.0 and paired.confidence_interval[1] < 0.0
    diagnostics_pass = all(bool(value["passed"]) for value in gates.values())
    required_available = all(bool(value["available"]) for value in gates.values())
    elapsed_before_decision = float(perf_counter() - started_at)
    budget_exhausted = (not smoke) and elapsed_before_decision > float(config["budget_seconds"])
    status_value, claim_status, decision_reason, promotion_eligible = classify_screen_decision(
        smoke=smoke,
        budget_exhausted=budget_exhausted,
        speed_improved=speed_improved,
        diagnostics_pass=diagnostics_pass,
        required_available=required_available,
    )
    status = RunStatus(status_value)

    process_memory_mb = _process_memory_mb()
    records: list[MethodRunRecord] = []
    baseline_primary = float(np.mean([value for samples in baseline_times.values() for value in samples]))
    for seed in seeds:
        seed_samples = [
            float(row["total_seconds"])
            for row in timing_samples
            if row["method"] == fixed_selection.config_id and row["seed"] == seed
        ]
        seed_diag = candidate_seed_aggregates[seed]
        component_summary = _aggregate_components(candidate_components[seed])
        records.append(
            MethodRunRecord(
                method_id="M03",
                phase=RunPhase.SCREEN,
                status=status,
                config_id=fixed_selection.config_id,
                primary_metric_value=float(np.mean(seed_samples)),
                baseline_metric_value=baseline_primary,
                optional_metrics=_record_metrics(
                    seed_diag,
                    paired,
                    selected_omega,
                    selected_blend,
                    selected_is_unrolled=selected_is_unrolled,
                    selected_k=int(fixed_selection.config_id.rsplit("k", 1)[-1]) if selected_is_unrolled else None,
                    residual_tolerance=float(solver_cfg["residual_tolerance"]),
                    cleanup_iterations_mean=float(
                        np.mean([result.cleanup_iterations_total for result in candidate_execution[seed]])
                    ),
                    fallback_count=float(sum(result.fallback_count for result in candidate_execution[seed])),
                ),
                scenario_count=len(splits.validation),
                seed=seed,
                split="validation",
                branch=resolved_provenance["branch"],
                commit=resolved_provenance["commit"],
                timings_seconds={
                    "label_generation": label_generation_seconds,
                    "preprocessing": preprocessing_seconds,
                    "training": (
                        unrolled_trained[seed][int(fixed_selection.config_id.rsplit("k", 1)[-1])].training_seconds
                        if selected_is_unrolled
                        else trained_models[seed].training_seconds
                    ),
                    "unrolled_training_total": unrolled_training_seconds if selected_is_unrolled else 0.0,
                    "selection": selection_seconds,
                    **component_summary,
                },
                memory_mb=process_memory_mb,
                claim_status=claim_status,
                notes="Smoke non-evidence." if smoke else decision_reason,
            )
        )

    aggregate_primary = float(np.mean([value for samples in candidate_times.values() for value in samples]))
    aggregate_record = MethodRunRecord(
        method_id="M03",
        phase=RunPhase.SCREEN,
        status=status,
        config_id=fixed_selection.config_id,
        primary_metric_value=aggregate_primary,
        baseline_metric_value=baseline_primary,
        optional_metrics=_record_metrics(
            candidate_aggregate,
            paired,
            selected_omega,
            selected_blend,
            selected_is_unrolled=selected_is_unrolled,
            selected_k=int(fixed_selection.config_id.rsplit("k", 1)[-1]) if selected_is_unrolled else None,
            residual_tolerance=float(solver_cfg["residual_tolerance"]),
            cleanup_iterations_mean=float(
                np.mean(
                    [
                        result.cleanup_iterations_total
                        for results in candidate_execution.values()
                        for result in results
                    ]
                )
            ),
            fallback_count=float(
                sum(
                    result.fallback_count
                    for results in candidate_execution.values()
                    for result in results
                )
            ),
        ),
        scenario_count=len(splits.validation),
        seed=None,
        split="validation",
        branch=resolved_provenance["branch"],
        commit=resolved_provenance["commit"],
        timings_seconds={
            "elapsed": float(perf_counter() - started_at),
            "label_generation": label_generation_seconds,
            "preprocessing": preprocessing_seconds,
            "training": (
                unrolled_training_seconds
                if selected_is_unrolled
                else float(sum(model.training_seconds for model in trained_models.values()))
            ),
            "selection": selection_seconds,
        },
        memory_mb=process_memory_mb,
        claim_status=claim_status,
        notes="Smoke non-evidence; promotion is disabled." if smoke else decision_reason,
    )
    records.append(aggregate_record)

    actual_counts = {
        "train_regimes": len(splits.train),
        "validation_regimes": len(splits.validation),
        "train_samples": (
            int(len(train_lcp_data)) if selected_is_unrolled else int(len(train_features))
        ),
        "validation_samples": (
            int(sum(len(extract_lcp_training_data(o)) for o in validation_oracles.values()))
            if selected_is_unrolled
            else int(len(validation_features))
        ),
        "training_seeds": len(seeds),
    }
    effective_config = deepcopy(config)
    effective_config.update(
        {
            "actual_counts": actual_counts,
            "blind_holdout_accessed": False,
            "evidence": not smoke,
            "selected_omega": selected_omega,
            "selected_candidate_config": fixed_selection.config_id,
            "predeclared_configuration_count": len(solver_cfg["omega_candidates"])
            + len(network_cfg["blend_candidates"]),
            "residual_tolerance": float(solver_cfg["residual_tolerance"]),
            "split_scenario_ids": {
                "train": [str(regime["scenario_id"]) for regime in splits.train],
                "validation": [str(regime["scenario_id"]) for regime in splits.validation],
            },
        }
    )
    summary = {
        "method_id": "M03",
        "phase": "screen",
        "run_mode": config["run_mode"],
        "status": status.value,
        "evidence": not smoke,
        "promotion_eligible": promotion_eligible,
        "blind_holdout_accessed": False,
        "actual_counts": actual_counts,
        "residual_tolerance": float(solver_cfg["residual_tolerance"]),
        "selection": {
            "baseline_omega": selected_omega,
            "baseline_omega_mean_iterations": omega_scores[selected_omega],
            "candidate_config_id": fixed_selection.config_id,
            "candidate_is_unrolled": selected_is_unrolled,
            "candidate_blend": None if selected_is_unrolled else selected_blend,
            "candidate_k": int(fixed_selection.config_id.rsplit("k", 1)[-1]) if selected_is_unrolled else None,
            "candidate_mean_cleanup_iterations": fixed_selection.mean_cleanup_iterations,
            "fixed_across_seeds": True,
            "seeds": list(seeds),
        },
        "paired_total_time": {
            "baseline_mean_seconds": baseline_primary,
            "candidate_mean_seconds": aggregate_primary,
            "mean_difference_seconds": paired.mean_difference,
            "sample_standard_deviation_seconds": paired.sample_standard_deviation,
            "confidence_interval_seconds": list(paired.confidence_interval),
            "paired_p_value": paired.paired_p_value,
            "relative_percent_improvement": paired.relative_percent_change,
            "sampling_unit": "complete_validation_regime",
        },
        "costs_seconds": {
            "baseline_selection": baseline_selection_seconds,
            "candidate_selection": candidate_selection_seconds,
            "label_generation": label_generation_seconds,
            "preprocessing": preprocessing_seconds,
            "training_total": (
                unrolled_training_seconds
                if selected_is_unrolled
                else float(sum(model.training_seconds for model in trained_models.values()))
            ),
            "selection": selection_seconds,
            "elapsed_before_writes": float(perf_counter() - started_at),
        },
        "budget": {
            "screen_budget_seconds": float(config["budget_seconds"]),
            "elapsed_before_decision_seconds": elapsed_before_decision,
            "exhausted": budget_exhausted,
        },
        "memory": {
            "process_rss_mb": process_memory_mb,
            "model_bytes_per_seed": {
                str(seed): trained_models[seed].model_bytes for seed in seeds
            },
        },
        "fallback": {
            "count": int(
                sum(
                    candidate_results[(seed, fixed_selection.config_id, str(regime["scenario_id"]))].fallback_count
                    for seed in seeds
                    for regime in splits.validation
                )
            ),
            "behavior": "project previous solution and run ordinary PSOR cleanup",
        },
        "diagnostics": {
            "baseline": baseline_aggregate,
            "candidate": candidate_aggregate,
            "gates": gates,
        },
        "decision": {
            "status": status.value,
            "claim_status": claim_status,
            "reason": decision_reason,
            "promotion_eligible": promotion_eligible,
            "speed_gate_passed": speed_improved,
            "diagnostic_gates_passed": diagnostics_pass,
            "required_diagnostics_available": required_available,
        },
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_csv(output_path / "scenario_metrics.csv", scenario_rows)
    _write_csv(output_path / "selection_metrics.csv", selection_rows)
    _write_csv(output_path / "timing_samples.csv", timing_samples)
    _write_json(output_path / "screen_summary.json", summary)
    standard_paths = write_method_outputs(
        output_path,
        get_method_spec("M03"),
        records,
        config=effective_config,
        decision_record=aggregate_record,
        training_history=training_history,
    )
    decision_payload = json.loads(standard_paths["decision"].read_text(encoding="utf-8"))
    decision_payload.update(
        {
            "blind_holdout_accessed": False,
            "evidence": not smoke,
            "promotion_eligible": promotion_eligible,
            "reason": decision_reason,
        }
    )
    _write_json(standard_paths["decision"], decision_payload)
    _write_complete_hash_manifest(output_path)
    files = {path.name: str(path) for path in sorted(output_path.iterdir())}
    return {
        "method_id": "M03",
        "output_dir": str(output_path),
        "status": status.value,
        "evidence": not smoke,
        "promotion_eligible": promotion_eligible,
        "blind_holdout_accessed": False,
        "files": files,
        "standard_files": {name: str(path) for name, path in standard_paths.items()},
    }


def _train_model(
    seed: int,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    config: Mapping[str, Any],
    normalizer: FeatureNormalizer,
) -> TrainedWarmStart:
    import torch
    import torch.nn.functional as functional

    training_started = perf_counter()
    torch.manual_seed(seed)
    model = build_warm_start_mlp(
        train_features.shape[1], hidden=int(config["hidden"]), depth=int(config["depth"])
    ).to(dtype=torch.float64, device="cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    train_x = torch.as_tensor(train_features, dtype=torch.float64)
    train_y = torch.as_tensor(train_targets[:, None], dtype=torch.float64)
    validation_x = torch.as_tensor(validation_features, dtype=torch.float64)
    validation_y = torch.as_tensor(validation_targets[:, None], dtype=torch.float64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    best_loss = float("inf")
    best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(config["epochs"]) + 1):
        epoch_started = perf_counter()
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        loss_sum = 0.0
        for start in range(0, len(order), int(config["batch_size"])):
            indices = order[start : start + int(config["batch_size"])]
            prediction = model(train_x[indices])
            loss = functional.mse_loss(prediction, train_y[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        model.eval()
        with torch.no_grad():
            validation_loss = float(functional.mse_loss(model(validation_x), validation_y))
        history.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_loss": float(loss_sum / len(train_x)),
                "validation_loss": validation_loss,
                "epoch_seconds": float(perf_counter() - epoch_started),
                "train_samples": len(train_x),
                "validation_samples": len(validation_x),
                "dtype": "float64",
                "device": "cpu",
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            stale = 0
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(config["patience"]):
                break
    model.load_state_dict(best_state)
    model.eval()
    training_seconds = float(perf_counter() - training_started)

    def predictor(features: np.ndarray) -> np.ndarray:
        normalized = normalizer.transform(features)
        with torch.no_grad():
            values = model(torch.as_tensor(normalized, dtype=torch.float64)).detach().cpu().numpy()
        return values.reshape(-1)

    model_bytes = int(sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()))
    return TrainedWarmStart(
        seed=seed,
        model=model,
        predictor=predictor,
        training_seconds=training_seconds,
        model_bytes=model_bytes,
        history=tuple(history),
    )


def _solve_regime(
    regime: Mapping[str, Any],
    solver_config: Mapping[str, Any],
    *,
    tolerance: float,
    omega: float,
    predictor: Any | None = None,
    blend: float = 1.0,
    unrolled_omegas: Any | None = None,
) -> SurfaceSolveResult:
    config = SolverConfig(
        option_type="put",
        k=float(solver_config["k"]),
        t=float(regime["t"]),
        r=float(regime["r"]),
        q=float(regime["q"]),
        sigma=float(regime["sigma"]),
        s_max=float(solver_config["s_max"]),
        m=int(solver_config["m"]),
        n=int(solver_config["n"]),
        omega=omega,
        tol=tolerance,
        max_iter=int(solver_config["max_iter"]),
        american=True,
    )
    return solve_cn_lcp_surface(
        config,
        residual_tolerance=tolerance,
        omega=omega,
        max_iter=config.max_iter,
        warm_start_predictor=predictor,
        blend=blend,
        unrolled_omegas=unrolled_omegas,
    )


def _collect_examples(results: Iterable[SurfaceSolveResult]) -> tuple[np.ndarray, np.ndarray]:
    feature_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    for result in results:
        features, targets = surface_training_examples(result)
        feature_batches.append(features)
        target_batches.append(targets)
    return np.vstack(feature_batches), np.concatenate(target_batches)


def _measure_surface(operation: Any, *, warmups: int, repetitions: int) -> dict[str, Any]:
    captured: list[SurfaceSolveResult] = []

    def measured_operation() -> None:
        captured.append(operation())

    timing = measure_timing(measured_operation, warmups=warmups, repetitions=repetitions)
    measured_results = captured[warmups:]
    component_medians = {
        name: float(np.median([result.timings_seconds[name] for result in measured_results]))
        for name in (
            "assembly",
            "feature_construction",
            "inference",
            "projection",
            "fallback",
            "cleanup",
            "total",
        )
    }
    return {
        "samples": list(timing.samples),
        "median_seconds": timing.median_seconds,
        "mad_seconds": timing.median_absolute_deviation_seconds,
        "p95_seconds": timing.p95_seconds,
        "component_medians": component_medians,
        "results": measured_results,
    }


def _surface_diagnostics(
    result: SurfaceSolveResult, oracle: SurfaceSolveResult
) -> dict[str, Any]:
    difference = result.values - oracle.values
    price_rmse = float(np.sqrt(np.mean(difference**2)))
    result_delta, result_gamma = delta_gamma(result.values, result.spots)
    oracle_delta, oracle_gamma = delta_gamma(oracle.values, oracle.spots)
    oracle_boundary = boundary_curve(oracle.spots, oracle.values, oracle.payoff, "put")
    result_boundary = boundary_curve(result.spots, result.values, result.payoff, "put")
    strict_mask = greek_diagnostic_masks(oracle.spots, oracle_boundary, oracle.config.k)["strict"]
    delta_rmse, delta_reason = _masked_rmse(result_delta, oracle_delta, strict_mask, "no_strict_delta_nodes")
    gamma_rmse, gamma_reason = _masked_rmse(
        result.config.k * result_gamma,
        oracle.config.k * oracle_gamma,
        strict_mask,
        "no_strict_gamma_nodes",
    )
    finite_boundary = np.isfinite(result_boundary) & np.isfinite(oracle_boundary)
    boundary_rmse, boundary_reason = _masked_rmse(
        result_boundary,
        oracle_boundary,
        finite_boundary,
        "no_common_finite_boundary",
    )
    continuation = oracle.values - oracle.payoff[None, :] > 10.0 * oracle.residual_tolerance
    continuation_rmse, continuation_reason = _masked_rmse(
        result.values,
        oracle.values,
        continuation,
        "no_oracle_continuation_nodes",
    )
    obstacle_violation = float(max(0.0, np.max(result.payoff[None, :] - result.values)))
    complementarity = _complementarity_error(result)
    return {
        "price_rmse": price_rmse,
        "price_rmse_reason": None,
        "delta_rmse": delta_rmse,
        "delta_rmse_reason": delta_reason,
        "scaled_gamma_rmse": gamma_rmse,
        "scaled_gamma_rmse_reason": gamma_reason,
        "boundary_rmse": boundary_rmse,
        "boundary_rmse_reason": boundary_reason,
        "continuation_rmse": continuation_rmse,
        "continuation_rmse_reason": continuation_reason,
        "complementarity_error": complementarity,
        "complementarity_error_reason": None,
        "natural_residual": float(max(result.natural_residuals)),
        "natural_residual_reason": None,
        "obstacle_violation": obstacle_violation,
        "obstacle_violation_reason": None,
    }


def _complementarity_error(result: SurfaceSolveResult) -> float:
    config = result.config
    _, lower, upper, matrix, rhs_matrix = cn_matrices(config)
    dt = config.t / config.n
    maximum = 0.0
    for step in range(config.n):
        tau_old = float(result.taus[step])
        tau_new = float(result.taus[step + 1])
        left_old, right_old = time_boundary_values(config, tau_old)
        left_new, right_new = time_boundary_values(config, tau_new)
        rhs = rhs_matrix @ result.values[step, 1:-1]
        rhs[0] += 0.5 * dt * lower[0] * (left_old + left_new)
        rhs[-1] += 0.5 * dt * upper[-1] * (right_old + right_new)
        residual = matrix @ result.values[step + 1, 1:-1] - rhs
        gap = result.values[step + 1, 1:-1] - result.payoff[1:-1]
        maximum = max(maximum, float(np.max(np.abs(gap * residual))))
    return maximum


def _masked_rmse(
    candidate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    reason: str,
) -> tuple[float | None, str | None]:
    selected = np.asarray(mask, dtype=bool)
    if not np.any(selected):
        return None, reason
    errors = np.asarray(candidate, dtype=float)[selected] - np.asarray(reference, dtype=float)[selected]
    return float(np.sqrt(np.mean(errors**2))), None


def _aggregate_diagnostics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = tuple(rows)
    if not materialized:
        raise ValueError("diagnostic rows must not be empty")
    metrics = (
        "price_rmse",
        "delta_rmse",
        "scaled_gamma_rmse",
        "boundary_rmse",
        "continuation_rmse",
        "complementarity_error",
        "natural_residual",
        "obstacle_violation",
    )
    maximum_metrics = {"complementarity_error", "natural_residual", "obstacle_violation"}
    aggregate: dict[str, Any] = {}
    for metric in metrics:
        values = [float(row[metric]) for row in materialized if row.get(metric) is not None]
        if len(values) != len(materialized):
            aggregate[metric] = None
            reasons = sorted(
                {str(row.get(f"{metric}_reason")) for row in materialized if row.get(metric) is None}
            )
            aggregate[f"{metric}_reason"] = ";".join(reasons)
        else:
            aggregate[metric] = float(max(values) if metric in maximum_metrics else np.mean(values))
            aggregate[f"{metric}_reason"] = None
    return aggregate


def _diagnostic_gates(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    residual_tolerance: float,
) -> dict[str, dict[str, Any]]:
    limits = {
        "price_rmse": 2.0,
        "delta_rmse": 2.0,
        "scaled_gamma_rmse": 2.0,
        "boundary_rmse": 2.0,
        "continuation_rmse": 2.0,
        "complementarity_error": 2.0,
        "natural_residual": 5.0,
    }
    gates: dict[str, dict[str, Any]] = {}
    for metric, limit_percent in limits.items():
        baseline_value = baseline.get(metric)
        candidate_value = candidate.get(metric)
        available = baseline_value is not None and candidate_value is not None
        floor = residual_tolerance * 1.0e-3 if metric in {"natural_residual", "complementarity_error"} else 1.0e-14
        threshold = None if not available else float(baseline_value) * (1.0 + limit_percent / 100.0) + floor
        gates[metric] = {
            "available": available,
            "passed": bool(available and float(candidate_value) <= float(threshold)),
            "baseline": baseline_value,
            "candidate": candidate_value,
            "limit_percent": limit_percent,
            "threshold": threshold,
            "reason": None if available else candidate.get(f"{metric}_reason") or baseline.get(f"{metric}_reason"),
        }
    obstacle = candidate.get("obstacle_violation")
    available = obstacle is not None
    gates["obstacle_violation"] = {
        "available": available,
        "passed": bool(available and float(obstacle) <= 1.0e-10),
        "baseline": baseline.get("obstacle_violation"),
        "candidate": obstacle,
        "limit": 1.0e-10,
        "reason": None if available else candidate.get("obstacle_violation_reason"),
    }
    return gates


def _scenario_row(
    method: str,
    scenario_id: str,
    seed: int | None,
    result: SurfaceSolveResult,
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "method": method,
        "scenario_id": scenario_id,
        "seed": seed,
        "residual_tolerance": result.residual_tolerance,
        "cleanup_iterations": result.cleanup_iterations_total,
        "cleanup_seconds": result.timings_seconds["cleanup"],
        "feature_construction_seconds": result.timings_seconds["feature_construction"],
        "inference_seconds": result.timings_seconds["inference"],
        "projection_seconds": result.timings_seconds["projection"],
        "fallback_seconds": result.timings_seconds["fallback"],
        "total_seconds": result.timings_seconds["total"],
        "fallback_count": result.fallback_count,
        **dict(diagnostics),
    }


def _timing_rows(
    method: str,
    scenario_id: str,
    seed: int | None,
    measurement: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repetition, (sample, result) in enumerate(
        zip(measurement["samples"], measurement["results"]), start=1
    ):
        rows.append(
            {
                "method": method,
                "scenario_id": scenario_id,
                "seed": seed,
                "repetition": repetition,
                "total_seconds": float(sample),
                "assembly_seconds": result.timings_seconds["assembly"],
                "feature_construction_seconds": result.timings_seconds["feature_construction"],
                "inference_seconds": result.timings_seconds["inference"],
                "projection_seconds": result.timings_seconds["projection"],
                "fallback_seconds": result.timings_seconds["fallback"],
                "cleanup_seconds": result.timings_seconds["cleanup"],
                "cleanup_iterations": result.cleanup_iterations_total,
                "fallback_count": result.fallback_count,
                "natural_residual": max(result.natural_residuals),
            }
        )
    return rows


def _record_metrics(
    diagnostics: Mapping[str, Any],
    paired: Any,
    selected_omega: float,
    selected_blend: float,
    *,
    selected_is_unrolled: bool = False,
    selected_k: int | None = None,
    residual_tolerance: float,
    cleanup_iterations_mean: float,
    fallback_count: float,
) -> dict[str, float | None]:
    values: dict[str, float | None] = {
        metric: diagnostics.get(metric)
        for metric in (
            "price_rmse",
            "delta_rmse",
            "scaled_gamma_rmse",
            "boundary_rmse",
            "continuation_rmse",
            "complementarity_error",
            "natural_residual",
            "obstacle_violation",
        )
    }
    values.update(
        {
            "relative_percent_improvement": paired.relative_percent_change,
            "time_ci_low_seconds": paired.confidence_interval[0],
            "time_ci_high_seconds": paired.confidence_interval[1],
            "paired_p_value": paired.paired_p_value,
            "selected_omega": selected_omega,
            "selected_blend": selected_blend,
            "selected_is_unrolled": 1.0 if selected_is_unrolled else 0.0,
            "selected_k": float(selected_k) if selected_k is not None else None,
            "residual_tolerance": residual_tolerance,
            "cleanup_iterations_mean": cleanup_iterations_mean,
            "fallback_count": fallback_count,
            "blind_holdout_accessed": 0.0,
        }
    )
    return values


def _aggregate_components(rows: Iterable[Mapping[str, float]]) -> dict[str, float]:
    materialized = tuple(rows)
    return {
        name: float(np.mean([row[name] for row in materialized]))
        for name in (
            "assembly",
            "feature_construction",
            "inference",
            "projection",
            "fallback",
            "cleanup",
            "total",
        )
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    seeds = tuple(int(seed) for seed in config["seeds"])
    if seeds != REQUIRED_SEEDS:
        raise ValueError(f"M03 stochastic seeds must be {list(REQUIRED_SEEDS)}")
    solver = config["solver"]
    network = config["network"]
    tolerance = float(solver["residual_tolerance"])
    label_tolerance = float(solver["label_residual_tolerance"])
    if not 0.0 < label_tolerance < tolerance:
        raise ValueError("label_residual_tolerance must be positive and tighter than residual_tolerance")
    unroll_cfg = config.get("unrolling", {})
    configuration_count = (
        len(solver["omega_candidates"])
        + len(network["blend_candidates"])
        + len(unroll_cfg.get("K_candidates", []))
    )
    if not 1 <= configuration_count <= 8:
        raise ValueError("M03 must predeclare between one and eight configurations")
    if str(network["dtype"]) != "float64" or str(network["device"]) != "cpu":
        raise ValueError("M03 canonical learning uses CPU float64")
    if int(config["sampling"]["validation_regimes"]) < 2:
        raise ValueError("paired statistics require at least two validation regimes")


def _deep_update(target: dict[str, Any], updates: Mapping[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)


def _omega_config_id(omega: float) -> str:
    return f"tuned_psor_omega_{omega:.2f}".replace(".", "p")


def _blend_config_id(blend: float) -> str:
    return f"learned_feasible_blend_{blend:.2f}".replace(".", "p")


def _blend_from_config_id(config_id: str) -> float:
    return float(config_id.rsplit("_", 1)[-1].replace("p", "."))


def _process_memory_mb() -> float:
    import psutil

    return float(psutil.Process().memory_info().rss / (1024.0**2))


def _git_provenance() -> dict[str, str]:
    branch = subprocess.run(
        ["git", "branch", "--show-current"], text=True, capture_output=True, check=True
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True
    ).stdout.strip()
    return {"branch": branch, "commit": commit}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty detail file: {path.name}")
    fieldnames = tuple(sorted({key for row in rows for key in row}))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_complete_hash_manifest(output_dir: Path) -> None:
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_dir.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _write_json(output_dir / "artifact_hashes.json", {"algorithm": "sha256", "files": hashes})
