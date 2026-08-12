"""Stage 24: classical oracle and one-factor time/LCP solver comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from campaign.io import write_method_outputs
from campaign.records import MethodRunRecord, RunPhase, RunStatus
from campaign.registry import get_method_spec
from campaign.statistics import holm_adjust, measure_timing, summarize_paired_scenarios
from common import ensure_dir, load_config, write_csv, write_json
from diagnostics.convergence import masked_error, surface_to_reference_grid
from diagnostics.solver_checks import closed_form_error_in_region, solver_summary
from solvers.black_scholes import black_scholes_price, payoff
from solvers.boundaries import boundary_curve, boundary_curve_details
from solvers.cn_psor import (
    SolverConfig,
    brennan_schwartz_lcp,
    cn_matrices,
    policy_iteration_lcp,
    solve_option,
    time_boundary_values,
)
from solvers.greeks import delta_gamma, greek_diagnostic_masks
from solvers.integral_boundary import american_put_boundary_integral


@dataclass(frozen=True)
class SolvedCase:
    result: Any
    boundary: np.ndarray
    boundary_details: list[Any]
    delta: np.ndarray
    gamma: np.ndarray
    strict_mask: np.ndarray


def run(
    config_path: str,
    *,
    output_root: str | Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    run_started = perf_counter()
    cfg = load_config(config_path)
    if cfg.get("method_id") != "M01":
        raise ValueError("M01 experiment requires method_id: M01")
    if cfg.get("phase") != "screen":
        raise ValueError("M01 experiment requires phase: screen")
    _validate_methods(cfg.get("methods", []))
    output = ensure_dir(output_root or cfg["output_root"])
    effective = _effective_config(cfg, smoke=smoke)
    branch, commit = _git_provenance()
    methods = effective["methods"]
    regimes = effective["regimes"]

    comparison_rows: list[dict[str, Any]] = []
    analytic_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    references: dict[str, SolvedCase] = {}
    control_generation_elapsed = 0.0
    comparison_refinement_elapsed = 0.0
    deterministic_timing_elapsed = 0.0
    reference_hard_lcp_converged = True

    for regime in regimes:
        control_started = perf_counter()
        scenario_id = regime["scenario_id"]
        reference_cfg = _solver_config(
            effective["common"], regime, effective["reference"]
        )
        reference = _solve(reference_cfg)
        references[scenario_id] = reference
        reference_hard_lcp_converged &= all(reference.result.lcp_converged)

        if reference_cfg.r <= 0.0:
            boundary_rows.append(
                {
                    "scenario_id": scenario_id,
                    "reference_config": "rannacher_cn_policy_iteration",
                    "integral_steps": 0,
                    "boundary_rmse": None,
                    "boundary_mae": None,
                    "maturity_boundary_difference": None,
                    "applicable": False,
                    "control_status": "not_applicable_negative_rate",
                    "applicability_reason": "integral_boundary_requires_positive_rate",
                    "passed": None,
                }
            )
        else:
            integral_taus, integral_boundary = american_put_boundary_integral(
                k=reference_cfg.k,
                t=reference_cfg.t,
                r=reference_cfg.r,
                q=reference_cfg.q,
                sigma=reference_cfg.sigma,
                steps=int(effective["integral_boundary"]["steps"]),
            )
            reference_boundary_at_integral = _interpolate_finite_boundary(
                reference.result.taus,
                reference.boundary,
                integral_taus,
                valid=np.array(
                    [
                        detail.status == "boundary_found"
                        for detail in reference.boundary_details
                    ]
                ),
            )
            boundary_diff = reference_boundary_at_integral - integral_boundary
            boundary_rmse = _finite_rmse(boundary_diff)
            boundary_passed = boundary_rmse <= 0.05
            boundary_rows.append(
                {
                    "scenario_id": scenario_id,
                    "reference_config": "rannacher_cn_policy_iteration",
                    "integral_steps": len(integral_taus) - 1,
                    "boundary_rmse": boundary_rmse,
                    "boundary_mae": _finite_mae(boundary_diff),
                    "maturity_boundary_difference": float(boundary_diff[-1]),
                    "applicable": True,
                    "control_status": (
                        "applicable_passed" if boundary_passed else "applicable_failed"
                    ),
                    "applicability_reason": "positive_rate_integral_control",
                    "passed": boundary_passed,
                }
            )
        analytic_rows.extend(_analytic_control_rows(effective, regime, methods))
        brennan_row = _brennan_control_row(effective, regime)
        analytic_rows.append(brennan_row)
        control_generation_elapsed += perf_counter() - control_started

        for method in methods:
            comparison_started = perf_counter()
            candidate_cfg = _solver_config(effective["common"], regime, method)
            candidate = _solve(candidate_cfg)
            row = _comparison_row(
                scenario_id,
                method,
                candidate,
                reference,
                effective["region"],
            )
            comparison_refinement_elapsed += perf_counter() - comparison_started
            timing_cfg = _solver_config(
                effective["common"],
                regime,
                {
                    **method,
                    "m": effective["timing"]["m"],
                    "n": effective["timing"]["n"],
                },
            )
            timing_started = perf_counter()
            timing = measure_timing(
                lambda current=timing_cfg: solve_option(current),
                warmups=int(effective["timing"]["warmups"]),
                repetitions=int(effective["timing"]["repetitions"]),
            )
            deterministic_timing_elapsed += perf_counter() - timing_started
            row.update(
                {
                    "timing_median_seconds": timing.median_seconds,
                    "timing_mad_seconds": timing.median_absolute_deviation_seconds,
                    "timing_p95_seconds": timing.p95_seconds,
                }
            )
            comparison_rows.append(row)
            timing_rows.extend(
                {
                    "scenario_id": scenario_id,
                    "config_id": method["config_id"],
                    "repetition": index,
                    "seconds": seconds,
                }
                for index, seconds in enumerate(timing.samples, start=1)
            )

            for grid in effective["grid_refinement"]["grids"]:
                refinement_started = perf_counter()
                grid_cfg = _solver_config(
                    effective["common"], regime, {**method, "m": grid, "n": grid}
                )
                grid_case = _solve(grid_cfg)
                grid_metric = _comparison_row(
                    scenario_id,
                    method,
                    grid_case,
                    reference,
                    effective["region"],
                )
                grid_rows.append(
                    {
                        **grid_metric,
                        "grid": grid,
                        "grid_spacing": grid_cfg.s_max / grid_cfg.m,
                        "reference_grid": reference_cfg.m,
                        "refinement_check": "diagnostic_only_no_order_claim",
                    }
                )
                comparison_refinement_elapsed += perf_counter() - refinement_started

    method_spec = get_method_spec("M01")
    screen_elapsed = perf_counter() - run_started
    resource_accounting = {
        "screen_budget_seconds": method_spec.screen_budget_seconds,
        "total_screen_elapsed_seconds": screen_elapsed,
        "deterministic_timing_elapsed_seconds": deterministic_timing_elapsed,
        "control_generation_elapsed_seconds": control_generation_elapsed,
        "comparison_refinement_elapsed_seconds": comparison_refinement_elapsed,
        "training_elapsed_seconds": 0.0,
        "training_cost_status": "not_applicable_no_trainable_model",
        "timing_warmups": int(effective["timing"]["warmups"]),
        "timing_repetitions": int(effective["timing"]["repetitions"]),
        "memory_mb": _process_peak_rss_mb(),
        "memory_measure": "process_peak_rss",
        "budget_exceeded": screen_elapsed > method_spec.screen_budget_seconds,
    }
    records, selected_id, method_decisions, supported_config_ids = _build_records(
        effective,
        comparison_rows,
        analytic_rows,
        grid_rows,
        boundary_rows,
        reference_hard_lcp_converged=reference_hard_lcp_converged,
        resource_accounting=resource_accounting,
        branch=branch,
        commit=commit,
        smoke=smoke,
    )
    selected_record = next(record for record in records if record.config_id == selected_id)

    write_csv(output / "solver_comparison.csv", comparison_rows)
    write_csv(output / "analytic_controls.csv", analytic_rows)
    write_csv(output / "grid_refinement.csv", grid_rows)
    write_csv(output / "boundary_crosscheck.csv", boundary_rows)
    write_csv(output / "timing_samples.csv", timing_rows)
    summary = {
        "method_id": "M01",
        "run_mode": "smoke_non_evidence" if smoke else "canonical_screen",
        "evidence_status": _evidence_status(
            smoke=smoke,
            supported_config_ids=supported_config_ids,
            budget_exceeded=bool(resource_accounting["budget_exceeded"]),
        ),
        "phase": "screen",
        "scenario_count": len(regimes),
        "selected_config_id": selected_id,
        "method_decisions": method_decisions,
        "blind_holdout_accessed": False,
        "reference": effective["reference"],
        "resource_accounting": resource_accounting,
        "grid_refinement_claim": (
            "diagnostic consistency check only; no convergence order is estimated"
        ),
        "claim_limit": (
            "M01 compares discrete numerical methods on committed validation controls; "
            "it does not prove a continuous-PDE theorem or justify a universal fastest solver."
        ),
    }
    write_json(output / "m01_summary.json", summary)
    manifest_config = {
        **effective,
        "config_path": str(config_path),
        "run_mode": summary["run_mode"],
        "git_branch": branch,
        "git_commit": commit,
        "blind_holdout_accessed": False,
        "evidence_status": summary["evidence_status"],
        "resource_accounting": resource_accounting,
    }
    write_method_outputs(
        output,
        method_spec,
        records,
        config=manifest_config,
        decision_record=selected_record,
    )
    _rewrite_artifact_hashes(output)
    return summary


def _effective_config(cfg: dict[str, Any], *, smoke: bool) -> dict[str, Any]:
    effective = json.loads(json.dumps(cfg))
    if not smoke:
        return effective
    smoke_cfg = effective["smoke"]
    common_grid = int(smoke_cfg["common_grid"])
    effective["common"].update({"m": common_grid, "n": common_grid})
    reference_grid = int(smoke_cfg["reference_grid"])
    effective["reference"].update({"m": reference_grid, "n": reference_grid})
    timing_grid = int(smoke_cfg["timing_grid"])
    effective["timing"].update(
        {
            "m": timing_grid,
            "n": timing_grid,
            "warmups": int(smoke_cfg["timing_warmups"]),
            "repetitions": int(smoke_cfg["timing_repetitions"]),
        }
    )
    effective["integral_boundary"]["steps"] = int(smoke_cfg["integral_steps"])
    effective["bootstrap"]["iterations"] = int(smoke_cfg["bootstrap_iterations"])
    effective["regimes"] = effective["regimes"][: int(smoke_cfg["regime_count"])]
    effective["grid_refinement"]["grids"] = [
        int(grid) for grid in smoke_cfg["grid_refinement_grids"]
    ]
    return effective


def _validate_methods(methods: list[dict[str, Any]]) -> None:
    expected = {
        "cn_psor": ("cn", 0, "psor"),
        "rannacher_cn_psor": ("rannacher_cn", 2, "psor"),
        "dirk_psor": ("sdirk2", 0, "psor"),
        "cn_policy_iteration": ("cn", 0, "policy_iteration"),
        "cn_penalty_newton": ("cn", 0, "penalty_newton"),
    }
    by_id = {method.get("config_id"): method for method in methods}
    if len(methods) != len(expected) or set(by_id) != set(expected):
        raise ValueError("M01 requires exactly the five approved one-factor configurations")
    for config_id, signature in expected.items():
        method = by_id[config_id]
        actual = (
            method.get("time_scheme"),
            int(method.get("rannacher_steps", 0)),
            method.get("lcp_solver"),
        )
        if actual != signature:
            raise ValueError(f"M01 configuration {config_id!r} does not match the approved plan")


def _solver_config(base: dict[str, Any], regime: dict[str, Any], method: dict[str, Any]) -> SolverConfig:
    allowed = set(SolverConfig.__dataclass_fields__)
    merged = {**base, **regime, **method}
    return SolverConfig(**{key: value for key, value in merged.items() if key in allowed})


def _solve(cfg: SolverConfig) -> SolvedCase:
    result = solve_option(cfg)
    if not np.isfinite(result.values).all():
        raise FloatingPointError("solver produced non-finite values")
    boundary = boundary_curve(result.spots, result.values, result.payoff, cfg.option_type)
    details = boundary_curve_details(result.spots, result.values, result.payoff, cfg.option_type)
    delta, gamma = delta_gamma(result.values, result.spots)
    masks = greek_diagnostic_masks(result.spots, boundary, cfg.k)
    return SolvedCase(result, boundary, details, delta, gamma, masks["strict"])


def _comparison_factor(config_id: str) -> str:
    if config_id == "cn_psor":
        return "baseline"
    if config_id in {"rannacher_cn_psor", "dirk_psor"}:
        return "time_scheme"
    return "lcp_solver"


def _comparison_row(
    scenario_id: str,
    method: dict[str, Any],
    candidate: SolvedCase,
    reference: SolvedCase,
    region: dict[str, float],
) -> dict[str, Any]:
    ref = reference.result
    values_on_ref = surface_to_reference_grid(
        candidate.result.spots,
        candidate.result.taus,
        candidate.result.values,
        ref.spots,
        ref.taus,
    )
    delta_on_ref = surface_to_reference_grid(
        candidate.result.spots,
        candidate.result.taus,
        candidate.delta,
        ref.spots,
        ref.taus,
    )
    gamma_on_ref = surface_to_reference_grid(
        candidate.result.spots,
        candidate.result.taus,
        candidate.gamma,
        ref.spots,
        ref.taus,
    )
    price_mask = (ref.spots >= region["s_min_over_k"] * ref.config.k) & (
        ref.spots <= region["s_max_over_k"] * ref.config.k
    )
    price = masked_error(values_on_ref[-1:], ref.values[-1:], price_mask[None, :])
    delta = masked_error(delta_on_ref, reference.delta, reference.strict_mask)
    gamma = masked_error(gamma_on_ref, reference.gamma, reference.strict_mask)
    reference_boundary = _interpolate_finite_boundary(
        ref.taus,
        reference.boundary,
        candidate.result.taus,
        valid=np.array(
            [detail.status == "boundary_found" for detail in reference.boundary_details]
        ),
    )
    candidate_boundary_valid = np.array(
        [detail.status == "boundary_found" for detail in candidate.boundary_details]
    )
    boundary_difference = np.where(
        candidate_boundary_valid, candidate.boundary - reference_boundary, np.nan
    )
    summary = solver_summary(candidate.result)
    config_id = method["config_id"]
    return {
        "scenario_id": scenario_id,
        "config_id": config_id,
        "comparison_factor": _comparison_factor(config_id),
        "time_scheme": candidate.result.config.time_scheme,
        "lcp_solver": candidate.result.config.lcp_solver,
        "price_rmse": price["rmse"],
        "price_mae": price["mae"],
        "delta_rmse": delta["rmse"],
        "delta_mae": delta["mae"],
        "gamma_rmse": gamma["rmse"],
        "gamma_mae": gamma["mae"],
        "boundary_rmse": _finite_rmse(boundary_difference),
        "boundary_mae": _finite_mae(boundary_difference),
        "max_obstacle_violation": max(
            summary["obstacle_violation"], summary["max_lcp_obstacle_violation"]
        ),
        "max_lcp_obstacle_violation": summary["max_lcp_obstacle_violation"],
        "max_complementarity_error": summary["complementarity_error"],
        "max_lcp_natural_residual": summary["max_lcp_natural_residual"],
        "max_lcp_continuation_violation": summary["max_lcp_continuation_violation"],
        "max_lcp_iterations": summary["max_lcp_iterations"],
        "all_lcp_converged": summary["all_lcp_converged"],
    }


def _analytic_control_rows(
    effective: dict[str, Any],
    regime: dict[str, Any],
    methods: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    control_grid = int(effective["analytic_controls"]["m"])
    if effective["smoke"]["common_grid"] == effective["common"]["m"]:
        control_grid = int(effective["common"]["m"])
    for method in methods:
        european_cfg = _solver_config(
            {**effective["common"], "american": False},
            regime,
            {**method, "m": control_grid, "n": control_grid},
        )
        european = solve_option(european_cfg)
        euro_error = closed_form_error_in_region(european, 0.4, 1.8)
        rows.append(
            {
                "scenario_id": regime["scenario_id"],
                "config_id": method["config_id"],
                "control": "european_put_analytic",
                "rmse": euro_error["region_rmse"],
                "max_error": euro_error["region_max_error"],
                "applicable": True,
                "control_status": (
                    "applicable_passed"
                    if euro_error["region_rmse"] <= 0.05
                    else "applicable_failed"
                ),
                "applicability_reason": "european_closed_form_available",
                "passed": bool(euro_error["region_rmse"] <= 0.05),
                "all_lcp_converged": all(european.lcp_converged),
            }
        )
        if float(regime["r"]) < 0.0:
            rows.append(
                {
                    "scenario_id": regime["scenario_id"],
                    "config_id": method["config_id"],
                    "control": "no_dividend_american_call_equivalence",
                    "rmse": None,
                    "max_error": None,
                    "applicable": False,
                    "control_status": "not_applicable_negative_rate",
                    "applicability_reason": (
                        "no_dividend_call_equivalence_requires_nonnegative_rate"
                    ),
                    "passed": None,
                    "all_lcp_converged": None,
                }
            )
            continue
        call_cfg = _solver_config(
            effective["common"],
            {**regime, "option_type": "call", "q": 0.0},
            {**method, "m": control_grid, "n": control_grid},
        )
        call = solve_option(call_cfg)
        exact = np.asarray(
            black_scholes_price(
                call.spots,
                call_cfg.k,
                call_cfg.t,
                call_cfg.r,
                call_cfg.q,
                call_cfg.sigma,
                "call",
            )
        )
        mask = (call.spots >= 0.4 * call_cfg.k) & (call.spots <= 1.8 * call_cfg.k)
        difference = call.terminal_values[mask] - exact[mask]
        call_rmse = float(np.sqrt(np.mean(difference**2)))
        rows.append(
            {
                "scenario_id": regime["scenario_id"],
                "config_id": method["config_id"],
                "control": "no_dividend_american_call_equivalence",
                "rmse": call_rmse,
                "max_error": float(np.max(np.abs(difference))),
                "applicable": True,
                "control_status": (
                    "applicable_passed" if call_rmse <= 0.05 else "applicable_failed"
                ),
                "applicability_reason": "nonnegative_rate_no_dividend_call_theorem",
                "passed": bool(call_rmse <= 0.05),
                "all_lcp_converged": all(call.lcp_converged),
            }
        )
    return rows


def _brennan_control_row(
    effective: dict[str, Any], regime: dict[str, Any]
) -> dict[str, Any]:
    cfg = _solver_config(effective["common"], regime, {"m": 40, "n": 40})
    spots, lower, upper, a, b_mat = cn_matrices(cfg)
    phi = payoff(spots, cfg.k, "put")
    tau_old = 0.0
    tau_new = cfg.t / cfg.n
    left_old, right_old = time_boundary_values(cfg, tau_old)
    left_new, right_new = time_boundary_values(cfg, tau_new)
    rhs = np.einsum("ij,j->i", b_mat, phi[1:-1], optimize=False)
    rhs[0] += 0.5 * tau_new * lower[0] * (left_old + left_new)
    rhs[-1] += 0.5 * tau_new * upper[-1] * (right_old + right_new)
    brennan, info = brennan_schwartz_lcp(
        a,
        rhs,
        phi[1:-1],
        tol=cfg.tol,
        exercise_direction="put",
        interest_rate=cfg.r,
    )
    if not bool(info["applicable"]):
        reason = str(info["applicability_reason"])
        return {
            "scenario_id": regime["scenario_id"],
            "config_id": "brennan_schwartz",
            "control": "brennan_schwartz_first_step",
            "rmse": None,
            "max_error": None,
            "natural_residual": None,
            "applicable": False,
            "exercise_direction": "put",
            "applicability_reason": reason,
            "control_status": (
                "not_applicable_negative_rate"
                if reason == "negative_interest_rate"
                else "not_applicable_structural_conditions"
            ),
            "comparison_tolerance": None,
            "passed": None,
            "all_lcp_converged": None,
        }
    policy, policy_info = policy_iteration_lcp(
        a, rhs, phi[1:-1], tol=cfg.tol, max_iter=cfg.max_iter, x0=phi[1:-1]
    )
    error = float(np.max(np.abs(policy - brennan)))
    grid_tolerance = 0.01 * float((spots[1] - spots[0]) ** 2)
    passed = bool(
        error <= grid_tolerance
        and float(info["natural_residual"]) <= cfg.tol
        and bool(policy_info["converged"])
    )
    return {
        "scenario_id": regime["scenario_id"],
        "config_id": "brennan_schwartz",
        "control": "brennan_schwartz_first_step",
        "rmse": float(np.sqrt(np.mean((policy - brennan) ** 2))),
        "max_error": error,
        "natural_residual": float(info["natural_residual"]),
        "applicable": True,
        "exercise_direction": info["exercise_direction"],
        "applicability_reason": info["applicability_reason"],
        "control_status": "applicable_passed" if passed else "applicable_failed",
        "comparison_tolerance": grid_tolerance,
        "passed": passed,
        "all_lcp_converged": bool(policy_info["converged"]),
    }


def _build_records(
    effective: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    analytic_rows: list[dict[str, Any]],
    grid_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
    *,
    reference_hard_lcp_converged: bool,
    resource_accounting: dict[str, Any],
    branch: str,
    commit: str,
    smoke: bool,
) -> tuple[list[MethodRunRecord], str, dict[str, Any], tuple[str, ...]]:
    methods = [method["config_id"] for method in effective["methods"]]
    by_method = {
        method: [row for row in comparison_rows if row["config_id"] == method]
        for method in methods
    }
    baseline_id = "cn_psor"
    baseline_rows = by_method[baseline_id]
    baseline_runtime = {
        row["scenario_id"]: [row["timing_median_seconds"]] for row in baseline_rows
    }
    baseline_runtime_mean = float(
        np.mean([row["timing_median_seconds"] for row in baseline_rows])
    )
    timing_summaries = {}
    for method in methods:
        runtime = {
            row["scenario_id"]: [row["timing_median_seconds"]]
            for row in by_method[method]
        }
        timing_summaries[method] = summarize_paired_scenarios(
            baseline_runtime,
            runtime,
            lower_is_better=True,
            bootstrap_iterations=int(effective["bootstrap"]["iterations"]),
            random_seed=int(effective["bootstrap"]["seed"]),
        )
    candidate_ids = [method for method in methods if method != baseline_id]
    holm = holm_adjust(
        [timing_summaries[method].paired_p_value for method in candidate_ids],
        alpha=float(effective["gates"]["familywise_alpha"]),
    )
    timing_inference = {
        method: {
            "raw_p_value": timing_summaries[method].paired_p_value,
            "adjusted_p_value": adjusted,
            "reject": reject,
        }
        for method, adjusted, reject in zip(
            candidate_ids, holm.adjusted_p_values, holm.reject
        )
    }
    records: list[MethodRunRecord] = []
    decisions: dict[str, Any] = {}
    supported: list[tuple[float, str]] = []
    budget_exceeded = bool(resource_accounting["budget_exceeded"])

    for method in methods:
        rows = by_method[method]
        timing_summary = timing_summaries[method]
        inference = timing_inference.get(
            method,
            {"raw_p_value": 1.0, "adjusted_p_value": 1.0, "reject": False},
        )
        metrics = {
            key: float(np.mean([float(row[key]) for row in rows]))
            for key in (
                "price_rmse",
                "delta_rmse",
                "gamma_rmse",
                "boundary_rmse",
                "max_complementarity_error",
                "max_lcp_natural_residual",
            )
        }
        metrics["max_obstacle_violation"] = float(
            max(float(row["max_obstacle_violation"]) for row in rows)
        )
        metrics = {key: (None if isinstance(value, float) and not np.isfinite(value) else value) for key, value in metrics.items()}
        ordinary_keys = (
            "price_rmse",
            "delta_rmse",
            "gamma_rmse",
            "boundary_rmse",
            "max_complementarity_error",
        )
        baseline_means = {
            key: float(np.mean([float(row[key]) for row in baseline_rows]))
            for key in (*ordinary_keys, "max_lcp_natural_residual")
        }
        ordinary_limit = (
            1.0 + float(effective["gates"]["ordinary_no_regression_percent"]) / 100.0
        )
        residual_limit = (
            1.0 + float(effective["gates"]["residual_no_regression_percent"]) / 100.0
        )
        residual_floor = float(effective.get("tol", effective.get("common", {}).get("tol", 1e-10)))
        no_regression = all(
            (metrics[key] is None or metrics[key] <= baseline_means[key] * ordinary_limit + 1e-12)
            for key in ordinary_keys
        ) and (
            metrics["max_lcp_natural_residual"] is None
            or metrics["max_lcp_natural_residual"] <= max(
                baseline_means["max_lcp_natural_residual"] * residual_limit + 1e-12,
                residual_floor,
            )
        )
        method_spec = next(item for item in effective["methods"] if item["config_id"] == method)
        hard_method = method_spec["lcp_solver"] != "penalty_newton"
        obstacle_ok = (not hard_method) or metrics["max_obstacle_violation"] <= float(
            effective["gates"]["hard_obstacle_tolerance"]
        )
        relevant_analytic = [
            row
            for row in analytic_rows
            if row["config_id"] in {method, "brennan_schwartz"}
        ]
        analytic_ok = _applicable_controls_pass(relevant_analytic)
        boundary_ok = _applicable_controls_pass(boundary_rows)
        global_hard_controls_converged = all(
            bool(row["all_lcp_converged"])
            for row in analytic_rows
            if row["config_id"] == "brennan_schwartz" and bool(row["applicable"])
        )
        method_hard_rows = rows + [
            row for row in grid_rows if row["config_id"] == method
        ]
        method_hard_controls = [
            row
            for row in analytic_rows
            if row["config_id"] == method
            and bool(row["applicable"])
            and row["all_lcp_converged"] is not None
        ]
        hard_lcp_converged = bool(
            reference_hard_lcp_converged
            and global_hard_controls_converged
            and (
                not hard_method
                or (
                    all(bool(row["all_lcp_converged"]) for row in method_hard_rows)
                    and all(
                        bool(row["all_lcp_converged"])
                        for row in method_hard_controls
                    )
                )
            )
        )
        practical = timing_summary.relative_percent_change >= float(
            effective["gates"]["practical_improvement_percent"]
        )
        ci_excludes_zero = timing_summary.confidence_interval[1] < 0.0
        status, claim_status = _candidate_run_status(
            smoke=smoke,
            baseline=method == baseline_id,
            no_regression=no_regression,
            obstacle_ok=obstacle_ok,
            analytic_ok=analytic_ok,
            boundary_ok=boundary_ok,
            practical=practical,
            ci_excludes_zero=ci_excludes_zero,
            holm_reject=bool(inference["reject"]),
            hard_lcp_converged=hard_lcp_converged,
            budget_exceeded=budget_exceeded,
        )
        if status is RunStatus.COMPLETED:
            supported.append((float(np.mean([row["timing_median_seconds"] for row in rows])), method))

        mean_runtime = float(np.mean([row["timing_median_seconds"] for row in rows]))
        optional_metrics = {
            **metrics,
            "timing_paired_mean_difference": timing_summary.mean_difference,
            "timing_paired_ci_low": timing_summary.confidence_interval[0],
            "timing_paired_ci_high": timing_summary.confidence_interval[1],
            "timing_raw_p_value": float(inference["raw_p_value"]),
            "timing_holm_adjusted_p_value": float(inference["adjusted_p_value"]),
            "timing_holm_reject": float(bool(inference["reject"])),
            "timing_relative_improvement_percent": timing_summary.relative_percent_change,
            "all_relevant_hard_lcp_converged": float(hard_lcp_converged),
        }
        records.append(
            MethodRunRecord(
                method_id="M01",
                phase=RunPhase.SCREEN,
                status=status,
                config_id=method,
                primary_metric_value=mean_runtime if status is RunStatus.COMPLETED else None,
                baseline_metric_value=baseline_runtime_mean if status is RunStatus.COMPLETED else None,
                optional_metrics=optional_metrics,
                scenario_count=len(rows),
                split="validation_controls",
                branch=branch,
                commit=commit,
                timings_seconds={
                    "median_solve": mean_runtime,
                    "total_screen": float(
                        resource_accounting["total_screen_elapsed_seconds"]
                    ),
                    "deterministic_timing": float(
                        resource_accounting["deterministic_timing_elapsed_seconds"]
                    ),
                    "preprocessing_controls": float(
                        resource_accounting["control_generation_elapsed_seconds"]
                    ),
                    "comparison_refinement": float(
                        resource_accounting["comparison_refinement_elapsed_seconds"]
                    ),
                    "training": 0.0,
                },
                memory_mb=float(resource_accounting["memory_mb"]),
                claim_status=claim_status,
                notes="One-factor M01 comparison; smoke runs are not campaign evidence." if smoke else "One-factor M01 screen.",
            )
        )
        decisions[method] = {
            "status": status.value,
            "claim_status": claim_status,
            "no_regression": no_regression,
            "obstacle_gate": obstacle_ok,
            "analytic_controls": analytic_ok,
            "integral_boundary_gate": boundary_ok,
            "timing_practical_improvement": practical,
            "timing_ci_excludes_zero": ci_excludes_zero,
            "timing_raw_p_value": float(inference["raw_p_value"]),
            "timing_holm_adjusted_p_value": float(inference["adjusted_p_value"]),
            "timing_holm_reject": bool(inference["reject"]),
            "all_relevant_hard_lcp_converged": hard_lcp_converged,
            "budget_exceeded": budget_exceeded,
        }

    selected_id = min(supported)[1] if supported else baseline_id
    return records, selected_id, decisions, tuple(method for _, method in supported)


def _applicable_controls_pass(rows: list[dict[str, Any]]) -> bool:
    return all(not bool(row["applicable"]) or bool(row["passed"]) for row in rows)


def _candidate_run_status(
    *,
    smoke: bool,
    baseline: bool,
    no_regression: bool,
    obstacle_ok: bool,
    analytic_ok: bool,
    boundary_ok: bool,
    practical: bool,
    ci_excludes_zero: bool,
    holm_reject: bool,
    hard_lcp_converged: bool,
    budget_exceeded: bool,
) -> tuple[RunStatus, str]:
    if budget_exceeded:
        return RunStatus.RESOURCE_BLOCKED, "resource_blocked"
    if smoke:
        return RunStatus.DIAGNOSTIC_ONLY, "smoke_non_evidence"
    if baseline:
        return RunStatus.DIAGNOSTIC_ONLY, "benchmark"
    quality_ok = no_regression and obstacle_ok and analytic_ok and boundary_ok
    inference_ok = practical and ci_excludes_zero and holm_reject
    if quality_ok and hard_lcp_converged and inference_ok:
        return RunStatus.COMPLETED, "candidate_supported"
    if quality_ok and hard_lcp_converged:
        return RunStatus.DIAGNOSTIC_ONLY, "primary_inference_not_supported"
    return RunStatus.FAILED_METRIC_GATE, "failed_metric_gate"


def _evidence_status(
    *,
    smoke: bool,
    supported_config_ids: tuple[str, ...],
    budget_exceeded: bool,
) -> str:
    if smoke:
        return "diagnostic_non_evidence"
    if budget_exceeded:
        return "resource_blocked"
    if supported_config_ids:
        return "candidate_supported"
    return "benchmark_only"


def _interpolate_finite_boundary(
    source_taus: np.ndarray,
    source_boundary: np.ndarray,
    target_taus: np.ndarray,
    *,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    finite = np.isfinite(source_boundary)
    if valid is not None:
        finite &= np.asarray(valid, dtype=bool)
    if np.count_nonzero(finite) < 2:
        return np.full_like(target_taus, np.nan, dtype=float)
    return np.interp(
        target_taus,
        source_taus[finite],
        source_boundary[finite],
        left=np.nan,
        right=np.nan,
    )


def _finite_rmse(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite**2)))


def _finite_mae(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(np.abs(finite)))

def _process_peak_rss_mb() -> float:
    """Return peak RSS in MiB, or NaN when unavailable."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return float(usage.ru_maxrss) / (1024.0 * 1024.0)
    except Exception:
        return float("nan")



def _git_provenance() -> tuple[str, str]:
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return branch, commit


def _rewrite_artifact_hashes(output: Path) -> None:
    manifest = output / "artifact_hashes.json"
    files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.iterdir(), key=lambda item: item.name)
        if path.is_file() and path != manifest
    }
    manifest.write_text(
        json.dumps({"algorithm": "sha256", "files": files}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/independent_m01_classical.yaml")
    parser.add_argument("--output-root")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(run(args.config, output_root=args.output_root, smoke=args.smoke))
