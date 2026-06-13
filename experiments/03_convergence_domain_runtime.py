"""Stage 5: grid/domain sensitivity and runtime study."""

from __future__ import annotations

import argparse
import time

import numpy as np

from common import ensure_dir, load_config, write_csv, write_json
from diagnostics.shape import obstacle_violation
from experiment_utils import solver_config_from_dict
from solvers.boundaries import boundary_curve
from solvers.cn_psor import solve_option


def run(config_path: str) -> dict:
    raw = load_config(config_path)
    base = raw["base"]
    out_dir = ensure_dir("results/experiment_03_convergence_domain_runtime")
    rows = []
    reference_price = None
    reference_boundary = None

    for grid in raw["grids"]:
        cfg = solver_config_from_dict({**base, "m": int(grid), "n": int(grid)})
        start = time.perf_counter()
        result = solve_option(cfg)
        elapsed = time.perf_counter() - start
        boundary = boundary_curve(result.spots, result.values, result.payoff, cfg.option_type)
        if reference_price is None or grid == max(raw["grids"]):
            reference_price = result.terminal_values
            reference_boundary = boundary
        rows.append(
            {
                "grid": grid,
                "runtime_seconds": elapsed,
                "max_psor_iterations": max(result.psor_iterations),
                "obstacle_violation": obstacle_violation(result.values, result.payoff),
                "initial_boundary": float(boundary[-1]),
            }
        )

    domain_rows = []
    for domain in raw["domains"]:
        cfg = solver_config_from_dict({**base, "m": 80, "n": 80, "s_max": float(domain)})
        result = solve_option(cfg)
        boundary = boundary_curve(result.spots, result.values, result.payoff, cfg.option_type)
        domain_rows.append({"s_max": domain, "initial_boundary": float(boundary[-1])})

    write_csv(out_dir / "runtime_table.csv", rows)
    write_csv(out_dir / "domain_sensitivity.csv", domain_rows)
    summary = {"selected_grid": max(raw["grids"]), "selected_domain": base["s_max"]}
    write_json(out_dir / "convergence_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/convergence.yaml")
    args = parser.parse_args()
    print(run(args.config))
