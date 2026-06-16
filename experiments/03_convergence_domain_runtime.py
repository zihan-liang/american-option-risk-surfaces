"""Stage 4: grid/domain sensitivity and runtime study."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common import ensure_dir, load_config, write_csv, write_json
from diagnostics.convergence import (
    boundary_error,
    masked_error,
    psor_iteration_summary,
    surface_to_reference_grid,
)
from diagnostics.shape import convexity_violation, monotonicity_violation, obstacle_violation
from diagnostics.solver_checks import complementarity_error
from experiment_utils import save_npz, solver_config_from_dict
from plotting.surfaces import save_curve
from solvers.boundaries import boundary_curve, boundary_curve_details
from solvers.cn_psor import SolverConfig, SolverResult, solve_option
from solvers.greeks import delta_gamma, greek_diagnostic_masks


@dataclass
class SolvedCase:
    name: str
    result: SolverResult
    runtime_seconds: float
    boundary: np.ndarray
    boundary_details: list
    delta: np.ndarray
    gamma: np.ndarray
    masks: dict[str, np.ndarray]


def run(
    config_path: str,
    *,
    output_root: str | Path = "results/experiment_03_convergence_domain_runtime",
    figure_root: str | Path = "results/figures/stage04_convergence_runtime",
    overrides: dict | None = None,
) -> dict:
    raw = load_config(config_path)
    if overrides:
        raw = {**raw, **overrides}
    base = raw["base"]
    cases = raw["cases"]
    grids = [int(g) for g in raw["grids"]]
    domains = [float(d) for d in raw["domains"]]
    region = raw.get("region", {"s_min": 0.4, "s_max": 1.8})
    out_dir = ensure_dir(output_root)
    fig_dir = ensure_dir(figure_root)

    grid_rows: list[dict] = []
    runtime_rows: list[dict] = []
    diagnostics_rows: list[dict] = []
    domain_rows: list[dict] = []
    reference_payload: dict[str, np.ndarray] = {}

    for case in cases:
        case_name = case["name"]
        solved_by_grid = {
            grid: _solve_case(case_name, _config_for(base, case, m=grid, n=grid))
            for grid in grids
        }
        reference = solved_by_grid[max(grids)]
        reference_payload[f"{case_name}_spots"] = reference.result.spots
        reference_payload[f"{case_name}_taus"] = reference.result.taus
        reference_payload[f"{case_name}_values"] = reference.result.values
        reference_payload[f"{case_name}_delta"] = reference.delta
        reference_payload[f"{case_name}_gamma"] = reference.gamma
        reference_payload[f"{case_name}_boundary"] = reference.boundary

        for grid in grids:
            solved = solved_by_grid[grid]
            grid_rows.append(_grid_row(case_name, grid, solved, reference, region))
            runtime_rows.append(_runtime_row(case_name, grid, solved))
            diagnostics_rows.append(_diagnostics_row(case_name, f"grid_{grid}", solved))

        for domain in domains:
            solved = _solve_case(
                case_name,
                _config_for(base, case, m=min(max(grids), 120), n=min(max(grids), 120), s_max=domain),
            )
            domain_rows.append(_domain_row(case_name, domain, solved, reference, region))
            diagnostics_rows.append(_diagnostics_row(case_name, f"domain_{domain}", solved))

        _save_case_figures(case_name, fig_dir / case_name, grid_rows, domain_rows, runtime_rows)

    write_csv(out_dir / "grid_convergence.csv", grid_rows)
    write_csv(out_dir / "runtime_table.csv", runtime_rows)
    write_csv(out_dir / "diagnostics_table.csv", diagnostics_rows)
    write_csv(out_dir / "domain_sensitivity.csv", domain_rows)
    save_npz(out_dir / "reference_surfaces.npz", **reference_payload)
    summary = {
        "cases": len(cases),
        "grids": grids,
        "domains": domains,
        "max_obstacle_violation": max(row["obstacle_violation"] for row in diagnostics_rows),
        "max_complementarity_error": max(row["complementarity_error"] for row in diagnostics_rows),
        "max_monotonicity_violation": max(row["monotonicity_violation"] for row in diagnostics_rows),
        "max_convexity_violation": max(row["convexity_violation"] for row in diagnostics_rows),
        "reference_grid": max(grids),
        "reference_domain": base["s_max"],
    }
    write_json(out_dir / "convergence_summary.json", summary)
    write_json(out_dir / "convergence_config_manifest.json", raw)
    return summary


def _config_for(base: dict, case: dict, **overrides) -> SolverConfig:
    data = {**base, **case, **overrides}
    data.pop("name", None)
    return solver_config_from_dict(data)


def _solve_case(name: str, cfg: SolverConfig) -> SolvedCase:
    start = time.perf_counter()
    result = solve_option(cfg)
    runtime = time.perf_counter() - start
    boundary = boundary_curve(result.spots, result.values, result.payoff, cfg.option_type)
    boundary_details = boundary_curve_details(result.spots, result.values, result.payoff, cfg.option_type)
    delta, gamma = delta_gamma(result.values, result.spots)
    masks = greek_diagnostic_masks(result.spots, boundary, cfg.k)
    return SolvedCase(name, result, runtime, boundary, boundary_details, delta, gamma, masks)


def _grid_row(case_name: str, grid: int, solved: SolvedCase, reference: SolvedCase, region: dict) -> dict:
    ref = reference.result
    cfg = solved.result.config
    values_on_ref = surface_to_reference_grid(
        solved.result.spots, solved.result.taus, solved.result.values, ref.spots, ref.taus
    )
    delta_on_ref = surface_to_reference_grid(
        solved.result.spots, solved.result.taus, solved.delta, ref.spots, ref.taus
    )
    gamma_on_ref = surface_to_reference_grid(
        solved.result.spots, solved.result.taus, solved.gamma, ref.spots, ref.taus
    )
    region_mask = (ref.spots >= region["s_min"] * cfg.k) & (ref.spots <= region["s_max"] * cfg.k)
    price_metrics = masked_error(values_on_ref[-1:], ref.values[-1:], region_mask[None, :])
    boundary_metrics = boundary_error(solved.boundary_details, reference.boundary_details)
    delta_metrics = masked_error(delta_on_ref, reference.delta, reference.masks["strict"])
    gamma_metrics = masked_error(gamma_on_ref, reference.gamma, reference.masks["strict"])
    return {
        "case": case_name,
        "grid": grid,
        "price_rmse": price_metrics["rmse"],
        "price_mae": price_metrics["mae"],
        "price_max": price_metrics["max"],
        "boundary_samples": boundary_metrics["samples"],
        "boundary_rmse": boundary_metrics["rmse"],
        "boundary_mae": boundary_metrics["mae"],
        "boundary_max": boundary_metrics["max"],
        "delta_strict_samples": delta_metrics["samples"],
        "delta_strict_rmse": delta_metrics["rmse"],
        "delta_strict_mae": delta_metrics["mae"],
        "delta_strict_max": delta_metrics["max"],
        "gamma_strict_samples": gamma_metrics["samples"],
        "gamma_strict_rmse": gamma_metrics["rmse"],
        "gamma_strict_mae": gamma_metrics["mae"],
        "gamma_strict_max": gamma_metrics["max"],
    }


def _runtime_row(case_name: str, grid: int, solved: SolvedCase) -> dict:
    return {
        "case": case_name,
        "grid": grid,
        "runtime_seconds": solved.runtime_seconds,
        **psor_iteration_summary(solved.result.psor_iterations),
    }


def _diagnostics_row(case_name: str, setting: str, solved: SolvedCase) -> dict:
    result = solved.result
    return {
        "case": case_name,
        "setting": setting,
        "obstacle_violation": obstacle_violation(result.values, result.payoff),
        "complementarity_error": complementarity_error(result),
        "monotonicity_violation": monotonicity_violation(result.values, result.config.option_type),
        "convexity_violation": convexity_violation(result.values),
        "max_psor_iterations": max(result.psor_iterations) if result.psor_iterations else 0,
        "mean_psor_iterations": float(np.mean(result.psor_iterations)) if result.psor_iterations else 0.0,
    }


def _domain_row(case_name: str, domain: float, solved: SolvedCase, reference: SolvedCase, region: dict) -> dict:
    ref = reference.result
    cfg = solved.result.config
    values_on_ref = surface_to_reference_grid(
        solved.result.spots, solved.result.taus, solved.result.values, ref.spots, ref.taus
    )
    region_mask = (ref.spots >= region["s_min"] * cfg.k) & (ref.spots <= region["s_max"] * cfg.k)
    price_metrics = masked_error(values_on_ref[-1:], ref.values[-1:], region_mask[None, :])
    boundary_metrics = boundary_error(solved.boundary_details, reference.boundary_details)
    return {
        "case": case_name,
        "s_max": domain,
        "runtime_seconds": solved.runtime_seconds,
        "price_rmse_region": price_metrics["rmse"],
        "price_max_region": price_metrics["max"],
        "boundary_samples": boundary_metrics["samples"],
        "boundary_mae": boundary_metrics["mae"],
        "boundary_max": boundary_metrics["max"],
        "initial_boundary": float(solved.boundary[-1]),
        "obstacle_violation": obstacle_violation(solved.result.values, solved.result.payoff),
        **psor_iteration_summary(solved.result.psor_iterations),
    }


def _save_case_figures(
    case_name: str,
    fig_dir: Path,
    grid_rows: list[dict],
    domain_rows: list[dict],
    runtime_rows: list[dict],
) -> None:
    ensure_dir(fig_dir)
    case_grid = [row for row in grid_rows if row["case"] == case_name]
    case_domain = [row for row in domain_rows if row["case"] == case_name]
    case_runtime = [row for row in runtime_rows if row["case"] == case_name]
    if case_grid:
        grids = np.array([row["grid"] for row in case_grid], dtype=float)
        save_curve(grids, np.array([row["price_rmse"] for row in case_grid]), fig_dir / "price_error_vs_grid.png", f"{case_name} price error", "RMSE")
        save_curve(grids, np.array([row["boundary_mae"] for row in case_grid]), fig_dir / "boundary_error_vs_grid.png", f"{case_name} boundary error", "MAE")
        save_curve(grids, np.array([row["delta_strict_rmse"] for row in case_grid]), fig_dir / "delta_error_vs_grid.png", f"{case_name} Delta strict error", "RMSE")
        save_curve(grids, np.array([row["gamma_strict_rmse"] for row in case_grid]), fig_dir / "gamma_error_vs_grid.png", f"{case_name} Gamma strict error", "RMSE")
    if case_runtime:
        grids = np.array([row["grid"] for row in case_runtime], dtype=float)
        save_curve(grids, np.array([row["runtime_seconds"] for row in case_runtime]), fig_dir / "runtime_vs_grid.png", f"{case_name} runtime", "seconds")
    if case_domain:
        domains = np.array([row["s_max"] for row in case_domain], dtype=float)
        save_curve(domains, np.array([row["initial_boundary"] for row in case_domain]), fig_dir / "domain_boundary_sensitivity.png", f"{case_name} domain boundary sensitivity", "B/K")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/convergence.yaml")
    args = parser.parse_args()
    print(run(args.config))
