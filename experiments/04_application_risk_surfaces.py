"""Stage 5: audited application risk-surface sweep."""

from __future__ import annotations

import argparse
import itertools
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from common import ensure_dir, load_config, write_csv, write_json
from diagnostics.risk_metrics import (
    boundary_status_counts,
    classify_regime,
    exercise_fraction,
    finite_summary,
    greek_strict_summary,
    regime_flags,
)
from diagnostics.shape import convexity_violation, monotonicity_violation, obstacle_violation
from diagnostics.solver_checks import complementarity_error
from experiment_utils import save_npz, solver_config_from_dict
from plotting.risk_maps import save_heatmap
from plotting.surfaces import save_curve, save_price_surface
from solvers.boundaries import boundary_curve, boundary_curve_details
from solvers.cn_psor import SolverConfig, solve_option
from solvers.greeks import delta_gamma, greek_diagnostic_masks


def run(
    config_path: str,
    *,
    output_root: str | Path = "results/experiment_04_application_risk_surfaces",
    figure_root: str | Path = "results/figures/stage05_application_risk_surfaces",
    overrides: dict | None = None,
) -> dict:
    raw = load_config(config_path)
    if overrides:
        raw = {**raw, **overrides}

    out_dir = ensure_dir(output_root)
    fig_dir = ensure_dir(figure_root)
    run_id = raw.get("run_id") or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    surface_dir = ensure_dir(out_dir / "surfaces" / run_id)
    representative_limit_per_regime = int(raw.get("representative_limit_per_regime", raw.get("representative_limit", 4)))

    risk_rows: list[dict] = []
    boundary_rows: list[dict] = []
    greek_rows: list[dict] = []
    flag_rows: list[dict] = []
    surface_rows: list[dict] = []
    representative_counts = {
        "put": 0,
        "dividend_call": 0,
        "no_dividend_call_control": 0,
    }

    for option_type, t_value, sigma, r_value, q_value in _iter_cases(raw):
        cfg = _solver_config(raw, option_type, t_value, sigma, r_value, q_value)
        case_id = _case_id(option_type, t_value, sigma, r_value, q_value)
        regime_class = classify_regime(option_type, q_value)
        start = time.perf_counter()
        result = solve_option(cfg)
        runtime = time.perf_counter() - start

        boundary = boundary_curve(result.spots, result.values, result.payoff, cfg.option_type)
        boundary_details = boundary_curve_details(result.spots, result.values, result.payoff, cfg.option_type)
        delta, gamma = delta_gamma(result.values, result.spots)
        masks = greek_diagnostic_masks(result.spots, boundary, cfg.k)

        obstacle = obstacle_violation(result.values, result.payoff)
        complementarity = complementarity_error(result)
        monotonicity = monotonicity_violation(result.values, cfg.option_type)
        convexity = convexity_violation(result.values)
        statuses = boundary_status_counts(boundary_details)
        genuine_boundaries = [
            detail.value for detail in boundary_details if detail.status == "boundary_found"
        ]
        boundary_summary = finite_summary(genuine_boundaries)
        greek_summary = greek_strict_summary(delta, gamma, masks["strict"])
        flags = regime_flags(
            option_type=cfg.option_type,
            regime_class=regime_class,
            q=cfg.q,
            boundary_found_count=statuses["boundary_found"],
            obstacle_violation=obstacle,
            complementarity_error=complementarity,
            monotonicity_violation=monotonicity,
            convexity_violation=convexity,
        )

        common = {
            "case_id": case_id,
            "regime_class": regime_class,
            "option_type": cfg.option_type,
            "T": cfg.t,
            "sigma": cfg.sigma,
            "r": cfg.r,
            "q": cfg.q,
        }
        risk_rows.append(
            {
                **common,
                "runtime_seconds": runtime,
                "max_psor_iterations": max(result.psor_iterations) if result.psor_iterations else 0,
                "mean_psor_iterations": float(np.mean(result.psor_iterations)) if result.psor_iterations else 0.0,
                "obstacle_violation": obstacle,
                "complementarity_error": complementarity,
                "monotonicity_violation": monotonicity,
                "convexity_violation": convexity,
                "exercise_fraction": exercise_fraction(result.values, result.payoff),
            }
        )
        boundary_rows.append(
            {
                **common,
                **statuses,
                "boundary_samples": boundary_summary["samples"],
                "initial_boundary": boundary_summary["initial"],
                "mean_boundary": boundary_summary["mean"],
                "min_boundary": boundary_summary["min"],
                "max_boundary": boundary_summary["max"],
            }
        )
        greek_rows.append({**common, **greek_summary})
        flag_rows.append({**common, **flags})

        should_save = (
            _is_representative(cfg, regime_class)
            and representative_counts[regime_class] < representative_limit_per_regime
        )
        if should_save:
            surface_path = surface_dir / f"{case_id}.npz"
            save_npz(
                surface_path,
                spots=result.spots,
                taus=result.taus,
                values=result.values,
                payoff=result.payoff,
                boundary=boundary,
                delta=delta,
                gamma=gamma,
                mask_strict=masks["strict"],
                mask_near_kink=masks["near_kink"],
                mask_near_boundary=masks["near_boundary"],
                mask_near_zero_spot=masks["near_zero_spot"],
            )
            _save_representative_figures(fig_dir / regime_class / case_id, cfg, result.spots, result.taus, result.values, boundary, delta, gamma)
            surface_rows.append({**common, "path": str(surface_path), "saved": True})
            representative_counts[regime_class] += 1
        else:
            surface_rows.append({**common, "path": "", "saved": False})

    write_csv(out_dir / "risk_metrics.csv", risk_rows)
    write_csv(out_dir / "boundary_summary.csv", boundary_rows)
    write_csv(out_dir / "greek_summary.csv", greek_rows)
    write_csv(out_dir / "regime_flags.csv", flag_rows)
    write_csv(out_dir / "surface_index.csv", surface_rows)
    write_json(out_dir / "application_config_manifest.json", {**raw, "surface_run_id": run_id})
    _save_heatmaps(fig_dir, boundary_rows, risk_rows, greek_rows)

    summary = _summary(risk_rows, flag_rows, surface_rows)
    write_json(out_dir / "application_summary.json", summary)
    return summary


def _iter_cases(raw: dict):
    for option_type, t_value, sigma, r_value, q_value in itertools.product(
        ["put", "call"],
        raw["t_values"],
        raw["sigma_values"],
        raw["r_values"],
        raw["q_values"],
    ):
        yield option_type, float(t_value), float(sigma), float(r_value), float(q_value)


def _solver_config(raw: dict, option_type: str, t_value: float, sigma: float, r_value: float, q_value: float) -> SolverConfig:
    return solver_config_from_dict(
        {
            "option_type": option_type,
            "k": raw["k"],
            "t": t_value,
            "r": r_value,
            "q": q_value,
            "sigma": sigma,
            "s_max": raw["s_max"],
            "m": raw["m"],
            "n": raw["n"],
            "omega": raw["omega"],
            "tol": raw["tol"],
            "max_iter": raw["max_iter"],
            "american": True,
        }
    )


def _case_id(option_type: str, t_value: float, sigma: float, r_value: float, q_value: float) -> str:
    return f"{option_type}_T{t_value:g}_sig{sigma:g}_r{r_value:g}_q{q_value:g}".replace(".", "p")


def _is_representative(cfg: SolverConfig, regime_class: str) -> bool:
    return (
        cfg.t in {0.25, 1.0, 2.0}
        and cfg.r == 0.05
        and cfg.sigma in {0.2, 0.6}
        and (cfg.q in {0.0, 0.06, 0.10} or regime_class == "dividend_call")
    )


def _save_representative_figures(
    fig_dir: Path,
    cfg: SolverConfig,
    spots: np.ndarray,
    taus: np.ndarray,
    values: np.ndarray,
    boundary: np.ndarray,
    delta: np.ndarray,
    gamma: np.ndarray,
) -> None:
    ensure_dir(fig_dir)
    save_price_surface(spots / cfg.k, taus, values / cfg.k, fig_dir / "price_surface.png", "price surface")
    save_price_surface(spots / cfg.k, taus, delta, fig_dir / "delta_surface.png", "Delta surface")
    save_price_surface(spots / cfg.k, taus, gamma, fig_dir / "gamma_surface.png", "Gamma surface")
    save_curve(taus, boundary / cfg.k, fig_dir / "boundary_curve.png", "boundary curve", "B/K")


def _save_heatmaps(fig_dir: Path, boundary_rows: list[dict], risk_rows: list[dict], greek_rows: list[dict]) -> None:
    for regime_class in ["put", "dividend_call", "no_dividend_call_control"]:
        _save_metric_heatmap(fig_dir / regime_class, boundary_rows, regime_class, "initial_boundary", "initial boundary")
        _save_metric_heatmap(fig_dir / regime_class, risk_rows, regime_class, "exercise_fraction", "exercise fraction")
        _save_metric_heatmap(fig_dir / regime_class, risk_rows, regime_class, "runtime_seconds", "runtime")
        _save_metric_heatmap(fig_dir / regime_class, greek_rows, regime_class, "gamma_strict_mean_abs", "strict Gamma mean abs")


def _save_metric_heatmap(fig_dir: Path, rows: list[dict], regime_class: str, metric: str, title: str) -> None:
    selected = [row for row in rows if row["regime_class"] == regime_class and float(row["T"]) == 1.0 and float(row["r"]) == 0.05]
    if not selected:
        return
    sigmas = sorted({float(row["sigma"]) for row in selected})
    qs = sorted({float(row["q"]) for row in selected})
    values = np.full((len(qs), len(sigmas)), np.nan)
    for row in selected:
        i = qs.index(float(row["q"]))
        j = sigmas.index(float(row["sigma"]))
        values[i, j] = float(row[metric])
    ensure_dir(fig_dir)
    save_heatmap(sigmas, qs, values, fig_dir / f"{metric}_heatmap.png", f"{regime_class} {title}")


def _summary(risk_rows: list[dict], flag_rows: list[dict], surface_rows: list[dict]) -> dict:
    regime_counts = {
        regime: sum(1 for row in risk_rows if row["regime_class"] == regime)
        for regime in ["put", "dividend_call", "no_dividend_call_control"]
    }
    return {
        "total_cases": len(risk_rows),
        "regime_counts": regime_counts,
        "saved_surfaces": sum(1 for row in surface_rows if row["saved"]),
        "surface_run_id": _surface_run_id(surface_rows),
        "max_obstacle_violation": max(row["obstacle_violation"] for row in risk_rows),
        "max_complementarity_error": max(row["complementarity_error"] for row in risk_rows),
        "max_monotonicity_violation": max(row["monotonicity_violation"] for row in risk_rows),
        "max_convexity_violation": max(row["convexity_violation"] for row in risk_rows),
        "delicate_regimes": sum(1 for row in flag_rows if row["delicate_regime"]),
        "diagnostic_failures": sum(1 for row in flag_rows if row["diagnostic_failure"]),
        "note": "No neural surrogate evidence is produced by this stage.",
    }


def _surface_run_id(surface_rows: list[dict]) -> str:
    for row in surface_rows:
        if row["saved"]:
            return Path(row["path"]).parent.name
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/application_sweep.yaml")
    args = parser.parse_args()
    print(run(args.config))
