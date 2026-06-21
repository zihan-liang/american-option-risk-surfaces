"""Stage 4: boundary, Greek, and shape diagnostics for one option config."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import ensure_dir, load_config, write_csv, write_json
from diagnostics.shape import convexity_violation, monotonicity_violation, obstacle_violation
from diagnostics.solver_checks import complementarity_error
from experiment_utils import save_npz, solver_config_from_dict
from plotting.surfaces import save_curve, save_price_surface
from solvers.boundaries import boundary_curve, boundary_curve_details
from solvers.cn_psor import solve_option
from solvers.greeks import delta_gamma, greek_diagnostic_masks


def run(
    config_path: str,
    *,
    output_root: str | Path | None = None,
    figure_root: str | Path | None = None,
    overrides: dict | None = None,
) -> dict:
    data = load_config(config_path)
    if overrides:
        data = {**data, **overrides}
    cfg = solver_config_from_dict(data)
    result = solve_option(cfg)
    boundary = boundary_curve(result.spots, result.values, result.payoff, cfg.option_type)
    boundary_details = boundary_curve_details(result.spots, result.values, result.payoff, cfg.option_type)
    delta, gamma = delta_gamma(result.values, result.spots)
    masks = greek_diagnostic_masks(result.spots, boundary, cfg.k)
    root = Path(output_root) if output_root is not None else Path("results/experiment_02_boundary_greeks")
    out_dir = ensure_dir(root / cfg.option_type)
    fig_root = ensure_dir((Path(figure_root) if figure_root is not None else Path("results/figures/stage03_boundary_greeks")) / cfg.option_type)
    metrics = {
        "option_type": cfg.option_type,
        "obstacle_violation": obstacle_violation(result.values, result.payoff),
        "complementarity_error": complementarity_error(result),
        "monotonicity_violation": monotonicity_violation(result.values, cfg.option_type),
        "convexity_violation": convexity_violation(result.values),
        "boundary_at_maturity": float(boundary[-1]),
        "boundary_status_at_maturity": boundary_details[-1].status,
        "strict_mask_fraction": float(np.mean(masks["strict"])),
        "near_kink_fraction": float(np.mean(masks["near_kink"])),
        "near_boundary_fraction": float(np.mean(masks["near_boundary"])),
        "near_zero_spot_fraction": float(np.mean(masks["near_zero_spot"])),
    }
    write_json(out_dir / "diagnostics_summary.json", metrics)
    write_csv(
        out_dir / "boundary_curve.csv",
        [
            {
                "option_type": cfg.option_type,
                "tau": float(t),
                "boundary": float(detail.value),
                "status": detail.status,
                "exercise_nodes": detail.exercise_nodes,
                "continuation_nodes": detail.continuation_nodes,
            }
            for t, detail in zip(result.taus, boundary_details)
        ],
    )
    write_csv(out_dir / "greek_summary.csv", _greek_summary_rows(delta, gamma, masks))
    write_csv(out_dir / "mask_summary.csv", _mask_summary_rows(masks))
    save_npz(
        out_dir / "surface_greeks.npz",
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
    option_label = cfg.option_type.replace("_", " ").title()
    save_price_surface(
        result.spots / cfg.k,
        result.taus,
        result.values / cfg.k,
        fig_root / "price_surface.png",
        f"{option_label} price surface",
        colorbar_label=r"$U/K$",
    )
    save_price_surface(
        result.spots / cfg.k,
        result.taus,
        delta,
        fig_root / "delta_surface.png",
        f"{option_label} Delta surface",
        colorbar_label=r"$\Delta$",
    )
    save_price_surface(
        result.spots / cfg.k,
        result.taus,
        gamma,
        fig_root / "gamma_surface.png",
        f"{option_label} Gamma surface",
        colorbar_label=r"$\Gamma$",
    )
    save_curve(
        result.taus,
        boundary / cfg.k,
        fig_root / "boundary_curve.png",
        f"{option_label} early-exercise boundary",
        r"$B/K$",
    )
    return metrics


def run_many(
    config_paths: list[str],
    *,
    output_root: str | Path = "results/experiment_02_boundary_greeks",
    figure_root: str | Path = "results/figures/stage03_boundary_greeks",
    overrides: dict | None = None,
) -> dict:
    output_root = Path(output_root)
    summaries = [run(path, output_root=output_root, figure_root=figure_root, overrides=overrides) for path in config_paths]
    combined = {
        "cases": len(summaries),
        "option_types": [row["option_type"] for row in summaries],
        "max_obstacle_violation": max(row["obstacle_violation"] for row in summaries),
        "max_complementarity_error": max(row["complementarity_error"] for row in summaries),
        "max_monotonicity_violation": max(row["monotonicity_violation"] for row in summaries),
        "max_convexity_violation": max(row["convexity_violation"] for row in summaries),
    }
    write_json(ensure_dir(output_root) / "combined_summary.json", combined)
    write_csv(ensure_dir(output_root) / "combined_summary.csv", summaries)
    return combined


def _greek_summary_rows(delta: np.ndarray, gamma: np.ndarray, masks: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for name, mask in [("all", np.ones_like(delta, dtype=bool)), *masks.items()]:
        if not np.any(mask):
            rows.append({"region": name, "samples": 0})
            continue
        rows.append(
            {
                "region": name,
                "samples": int(np.count_nonzero(mask)),
                "delta_min": float(np.min(delta[mask])),
                "delta_max": float(np.max(delta[mask])),
                "delta_mean": float(np.mean(delta[mask])),
                "gamma_min": float(np.min(gamma[mask])),
                "gamma_max": float(np.max(gamma[mask])),
                "gamma_mean_abs": float(np.mean(np.abs(gamma[mask]))),
            }
        )
    return rows


def _mask_summary_rows(masks: dict[str, np.ndarray]) -> list[dict]:
    total = next(iter(masks.values())).size
    return [
        {
            "mask": name,
            "samples": int(np.count_nonzero(mask)),
            "fraction": float(np.count_nonzero(mask) / total),
        }
        for name, mask in masks.items()
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_put.yaml")
    parser.add_argument("--both", action="store_true", help="Run base put and dividend-call configs.")
    args = parser.parse_args()
    if args.both:
        print(run_many(["configs/base_put.yaml", "configs/base_call_dividend.yaml"]))
    else:
        print(run(args.config))
