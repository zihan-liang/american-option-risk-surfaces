"""Stage 3: validate the CN/PSOR benchmark."""

from __future__ import annotations

import argparse
import time

from common import ensure_dir, load_config, write_csv, write_json
from diagnostics.shape import convexity_violation, monotonicity_violation, obstacle_violation
from diagnostics.solver_checks import closed_form_error, closed_form_error_in_region, solver_summary
from experiment_utils import solver_config_from_dict
from solvers.black_scholes import black_scholes_price
from solvers.boundaries import boundary_curve
from solvers.boundary_conditions import boundary_values
from solvers.cn_psor import solve_option


def run(config_path: str) -> dict:
    cfg_raw = load_config(config_path)
    common = cfg_raw.get("common", {})
    audit = cfg_raw.get("audit", {})
    grids = audit.get("grids", [common.get("m", 120)])
    domains = audit.get("domains", [common.get("s_max", 4.0)])
    region_min = float(audit.get("region_min", 0.4)) * float(common.get("k", 1.0))
    region_max = float(audit.get("region_max", 1.8)) * float(common.get("k", 1.0))
    out_dir = ensure_dir("results/experiment_01_solver_validation")
    rows = []

    euro_cfg = solver_config_from_dict({**common, **cfg_raw["european_put"]})
    start = time.perf_counter()
    euro = solve_option(euro_cfg)
    euro_runtime = time.perf_counter() - start
    euro_err = closed_form_error(euro)
    euro_region = closed_form_error_in_region(euro, region_min, region_max)
    rows.append({"check": "european_put_closed_form", "runtime_seconds": euro_runtime, **euro_err, **euro_region})

    call_cfg = solver_config_from_dict({**common, **cfg_raw["no_dividend_call"]})
    start = time.perf_counter()
    call = solve_option(call_cfg)
    call_runtime = time.perf_counter() - start
    exact_call = black_scholes_price(
        call.spots, call_cfg.k, call_cfg.t, call_cfg.r, call_cfg.q, call_cfg.sigma, "call"
    )
    call_err = abs(call.terminal_values - exact_call)
    call_region = (call.spots >= region_min) & (call.spots <= region_max)
    rows.append(
        {
            "check": "no_dividend_american_call",
            "runtime_seconds": call_runtime,
            "max_error": float(call_err.max()),
            "rmse": float((call_err**2).mean() ** 0.5),
            "region_max_error": float(call_err[call_region].max()),
            "region_rmse": float((call_err[call_region] ** 2).mean() ** 0.5),
        }
    )

    put_cfg = solver_config_from_dict({**common, **cfg_raw["american_put"]})
    start = time.perf_counter()
    put = solve_option(put_cfg)
    put_runtime = time.perf_counter() - start
    put_summary = solver_summary(put)
    boundary = boundary_curve(put.spots, put.values, put.payoff, "put")
    rows.append(
        {
            "check": "american_put_lcp",
            "runtime_seconds": put_runtime,
            "max_error": put_summary["obstacle_violation"],
            "rmse": put_summary["complementarity_error"],
            "region_max_error": 0.0,
            "region_rmse": 0.0,
        }
    )

    dividend_call_cfg = solver_config_from_dict({**common, **cfg_raw["dividend_call"]})
    dividend_call = solve_option(dividend_call_cfg)
    dividend_call_summary = solver_summary(dividend_call)
    dividend_call_boundary = boundary_curve(
        dividend_call.spots, dividend_call.values, dividend_call.payoff, "call"
    )

    diagnostics_rows = [
        _diagnostic_row("american_put", put),
        _diagnostic_row("no_dividend_american_call", call),
        _diagnostic_row("dividend_american_call", dividend_call),
    ]
    boundary_rows = [
        {
            "case": "american_put",
            "tau": float(tau),
            "boundary": float(value),
        }
        for tau, value in zip(put.taus, boundary)
    ] + [
        {
            "case": "dividend_american_call",
            "tau": float(tau),
            "boundary": float(value),
        }
        for tau, value in zip(dividend_call.taus, dividend_call_boundary)
    ]

    grid_rows = []
    for grid in grids:
        grid_common = {**common, "m": int(grid), "n": int(grid)}
        grid_euro_cfg = solver_config_from_dict({**grid_common, **cfg_raw["european_put"]})
        start = time.perf_counter()
        grid_euro = solve_option(grid_euro_cfg)
        elapsed = time.perf_counter() - start
        grid_euro_err = closed_form_error(grid_euro)
        grid_region = closed_form_error_in_region(grid_euro, region_min, region_max)
        grid_rows.append(
            {
                "case": "european_put_closed_form",
                "grid": int(grid),
                "runtime_seconds": elapsed,
                **grid_euro_err,
                **grid_region,
            }
        )
        grid_call_cfg = solver_config_from_dict({**grid_common, **cfg_raw["no_dividend_call"]})
        start = time.perf_counter()
        grid_call = solve_option(grid_call_cfg)
        elapsed = time.perf_counter() - start
        exact = black_scholes_price(
            grid_call.spots,
            grid_call_cfg.k,
            grid_call_cfg.t,
            grid_call_cfg.r,
            grid_call_cfg.q,
            grid_call_cfg.sigma,
            "call",
        )
        err = abs(grid_call.terminal_values - exact)
        region = (grid_call.spots >= region_min) & (grid_call.spots <= region_max)
        grid_rows.append(
            {
                "case": "no_dividend_american_call",
                "grid": int(grid),
                "runtime_seconds": elapsed,
                "max_error": float(err.max()),
                "rmse": float((err**2).mean() ** 0.5),
                "region_max_error": float(err[region].max()),
                "region_rmse": float((err[region] ** 2).mean() ** 0.5),
            }
        )
        grid_put_cfg = solver_config_from_dict({**grid_common, **cfg_raw["american_put"]})
        start = time.perf_counter()
        grid_put = solve_option(grid_put_cfg)
        elapsed = time.perf_counter() - start
        grid_boundary = boundary_curve(grid_put.spots, grid_put.values, grid_put.payoff, "put")
        grid_put_summary = solver_summary(grid_put)
        grid_rows.append(
            {
                "case": "american_put_lcp",
                "grid": int(grid),
                "runtime_seconds": elapsed,
                "max_error": grid_put_summary["obstacle_violation"],
                "rmse": grid_put_summary["complementarity_error"],
                "region_max_error": 0.0,
                "region_rmse": 0.0,
                "initial_boundary": float(grid_boundary[-1]),
            }
        )

    domain_rows = []
    for domain in domains:
        for case_name, case_cfg in [
            ("american_put", cfg_raw["american_put"]),
            ("dividend_american_call", cfg_raw["dividend_call"]),
        ]:
            domain_cfg = solver_config_from_dict({**common, **case_cfg, "s_max": float(domain)})
            start = time.perf_counter()
            domain_result = solve_option(domain_cfg)
            elapsed = time.perf_counter() - start
            domain_boundary = boundary_curve(
                domain_result.spots,
                domain_result.values,
                domain_result.payoff,
                domain_cfg.option_type,
            )
            domain_rows.append(
                {
                    "case": case_name,
                    "s_max": float(domain),
                    "runtime_seconds": elapsed,
                    "initial_boundary": float(domain_boundary[-1]),
                    "obstacle_violation": obstacle_violation(domain_result.values, domain_result.payoff),
                }
            )

    boundary_condition_rows = _boundary_condition_rows(common, cfg_raw)

    write_csv(out_dir / "validation_metrics.csv", rows)
    write_csv(out_dir / "diagnostics_table.csv", diagnostics_rows)
    write_csv(out_dir / "grid_refinement.csv", grid_rows)
    write_csv(out_dir / "domain_sensitivity.csv", domain_rows)
    write_csv(out_dir / "boundary_curve.csv", boundary_rows)
    write_csv(out_dir / "boundary_conditions.csv", boundary_condition_rows)
    summary = {
        "european_put": euro_err,
        "european_put_region": euro_region,
        "no_dividend_call_max_error": rows[1]["max_error"],
        "no_dividend_call_region_max_error": rows[1]["region_max_error"],
        "american_put": put_summary,
        "american_put_initial_boundary": float(boundary[-1]),
        "dividend_call": dividend_call_summary,
        "dividend_call_initial_boundary": float(dividend_call_boundary[-1]),
        "audit_grids": grids,
        "audit_domains": domains,
        "interpretation": (
            "Smoke-grid errors are dominated by uniform-grid resolution near the payoff kink, "
            "finite-domain truncation, and second-order CN time stepping from nonsmooth terminal data. "
            "Rannacher smoothing is configured but not implemented in the current solver."
        ),
    }
    write_json(out_dir / "validation_summary.json", summary)
    return summary


def _diagnostic_row(case: str, result) -> dict:
    return {
        "case": case,
        "obstacle_violation": obstacle_violation(result.values, result.payoff),
        "complementarity_error": solver_summary(result)["complementarity_error"],
        "monotonicity_violation": monotonicity_violation(result.values, result.config.option_type),
        "convexity_violation": convexity_violation(result.values),
        "max_psor_iterations": max(result.psor_iterations) if result.psor_iterations else 0,
        "all_psor_converged": all(result.psor_converged),
    }


def _boundary_condition_rows(common: dict, cfg_raw: dict) -> list[dict]:
    tau = 0.75 * float(common["t"])
    k = float(common["k"])
    s_max = float(common["s_max"])
    rows = []
    for name, case in [
        ("american_put", cfg_raw["american_put"]),
        ("no_dividend_american_call", cfg_raw["no_dividend_call"]),
        ("dividend_american_call", cfg_raw["dividend_call"]),
    ]:
        q = float(case.get("q", common["q"]))
        option_type = case["option_type"]
        left, right = boundary_values(option_type, tau, k, s_max, float(common["r"]), q)
        rows.append(
            {
                "case": name,
                "tau": tau,
                "left_boundary": left,
                "right_boundary": right,
                "notes": "American put uses K at S=0; calls use max(immediate, large-spot European asymptotic).",
            }
        )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/solver_validation.yaml")
    args = parser.parse_args()
    print(run(args.config))
