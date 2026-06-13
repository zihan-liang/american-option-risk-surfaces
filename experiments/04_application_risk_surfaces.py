"""Stage 6: application risk-surface sweep."""

from __future__ import annotations

import argparse
import itertools
import time

from common import ensure_dir, load_config, write_csv, write_json
from diagnostics.shape import convexity_violation, monotonicity_violation, obstacle_violation
from experiment_utils import solver_config_from_dict
from solvers.boundaries import boundary_curve
from solvers.cn_psor import solve_option


def run(config_path: str) -> dict:
    raw = load_config(config_path)
    out_dir = ensure_dir("results/experiment_04_application_risk_surfaces")
    rows = []
    option_types = ["put", "call"]
    for option_type, t, sigma, r, q in itertools.product(
        option_types,
        raw["t_values"],
        raw["sigma_values"],
        raw["r_values"],
        raw["q_values"],
    ):
        cfg = solver_config_from_dict(
            {
                "option_type": option_type,
                "k": raw["k"],
                "t": t,
                "r": r,
                "q": q,
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
        start = time.perf_counter()
        result = solve_option(cfg)
        elapsed = time.perf_counter() - start
        boundary = boundary_curve(result.spots, result.values, result.payoff, option_type)
        rows.append(
            {
                "option_type": option_type,
                "T": t,
                "sigma": sigma,
                "r": r,
                "q": q,
                "runtime_seconds": elapsed,
                "initial_boundary": float(boundary[-1]),
                "obstacle_violation": obstacle_violation(result.values, result.payoff),
                "monotonicity_violation": monotonicity_violation(result.values, option_type),
                "convexity_violation": convexity_violation(result.values),
                "max_psor_iterations": max(result.psor_iterations),
            }
        )
    write_csv(out_dir / "risk_metrics.csv", rows)
    summary = {"cases": len(rows), "note": "Review failed or delicate regimes before paper claims."}
    write_json(out_dir / "application_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/application_sweep.yaml")
    args = parser.parse_args()
    print(run(args.config))
