"""Stage 4: boundary, Greek, and shape diagnostics for one option config."""

from __future__ import annotations

import argparse

from common import ensure_dir, load_config, write_csv, write_json
from diagnostics.shape import convexity_violation, monotonicity_violation, obstacle_violation
from diagnostics.solver_checks import complementarity_error
from experiment_utils import save_npz, solver_config_from_dict
from solvers.boundaries import boundary_curve
from solvers.cn_psor import solve_option
from solvers.greeks import delta_gamma


def run(config_path: str) -> dict:
    cfg = solver_config_from_dict(load_config(config_path))
    result = solve_option(cfg)
    boundary = boundary_curve(result.spots, result.values, result.payoff, cfg.option_type)
    delta, gamma = delta_gamma(result.values, result.spots)
    out_dir = ensure_dir(f"results/experiment_02_boundary_greeks_{cfg.option_type}")
    metrics = {
        "option_type": cfg.option_type,
        "obstacle_violation": obstacle_violation(result.values, result.payoff),
        "complementarity_error": complementarity_error(result),
        "monotonicity_violation": monotonicity_violation(result.values, cfg.option_type),
        "convexity_violation": convexity_violation(result.values),
        "boundary_at_maturity": float(boundary[-1]),
    }
    write_json(out_dir / "diagnostics_summary.json", metrics)
    write_csv(
        out_dir / "boundary_curve.csv",
        [{"tau": float(t), "boundary": float(b)} for t, b in zip(result.taus, boundary)],
    )
    save_npz(out_dir / "surface_greeks.npz", spots=result.spots, taus=result.taus, values=result.values, delta=delta, gamma=gamma)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_put.yaml")
    args = parser.parse_args()
    print(run(args.config))
