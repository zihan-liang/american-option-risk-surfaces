"""Stage 3: validate the CN/PSOR benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import ensure_dir, load_config, write_csv, write_json
from diagnostics.solver_checks import closed_form_error, solver_summary
from experiment_utils import solver_config_from_dict
from solvers.black_scholes import black_scholes_price
from solvers.boundaries import boundary_curve
from solvers.cn_psor import solve_option


def run(config_path: str) -> dict:
    cfg_raw = load_config(config_path)
    common = cfg_raw.get("common", {})
    out_dir = ensure_dir("results/experiment_01_solver_validation")
    rows = []

    euro_cfg = solver_config_from_dict({**common, **cfg_raw["european_put"]})
    euro = solve_option(euro_cfg)
    euro_err = closed_form_error(euro)
    rows.append({"check": "european_put_closed_form", **euro_err})

    call_cfg = solver_config_from_dict({**common, **cfg_raw["no_dividend_call"]})
    call = solve_option(call_cfg)
    exact_call = black_scholes_price(
        call.spots, call_cfg.k, call_cfg.t, call_cfg.r, call_cfg.q, call_cfg.sigma, "call"
    )
    rows.append(
        {
            "check": "no_dividend_american_call",
            "max_error": float(abs(call.terminal_values - exact_call).max()),
            "rmse": float(((call.terminal_values - exact_call) ** 2).mean() ** 0.5),
        }
    )

    put_cfg = solver_config_from_dict({**common, **cfg_raw["american_put"]})
    put = solve_option(put_cfg)
    put_summary = solver_summary(put)
    boundary = boundary_curve(put.spots, put.values, put.payoff, "put")
    rows.append(
        {
            "check": "american_put_lcp",
            "max_error": put_summary["obstacle_violation"],
            "rmse": put_summary["complementarity_error"],
        }
    )

    write_csv(out_dir / "validation_metrics.csv", rows)
    summary = {
        "european_put": euro_err,
        "no_dividend_call_max_error": rows[1]["max_error"],
        "american_put": put_summary,
        "american_put_initial_boundary": float(boundary[-1]),
    }
    write_json(out_dir / "validation_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/solver_validation.yaml")
    args = parser.parse_args()
    print(run(args.config))
