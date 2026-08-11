"""Numerical validation diagnostics for CN/PSOR outputs."""

from __future__ import annotations

import numpy as np

from diagnostics.shape import obstacle_violation
from solvers.black_scholes import black_scholes_price
from solvers.cn_psor import SolverResult


def closed_form_error(result: SolverResult) -> dict[str, float]:
    cfg = result.config
    exact = black_scholes_price(result.spots, cfg.k, cfg.t, cfg.r, cfg.q, cfg.sigma, cfg.option_type)
    err = np.abs(result.terminal_values - exact)
    return {"max_error": float(np.max(err)), "rmse": float(np.sqrt(np.mean(err**2)))}


def closed_form_error_in_region(result: SolverResult, s_min: float, s_max: float) -> dict[str, float]:
    cfg = result.config
    exact = black_scholes_price(result.spots, cfg.k, cfg.t, cfg.r, cfg.q, cfg.sigma, cfg.option_type)
    region = (result.spots >= s_min) & (result.spots <= s_max)
    err = np.abs(result.terminal_values[region] - exact[region])
    return {
        "region_max_error": float(np.max(err)),
        "region_rmse": float(np.sqrt(np.mean(err**2))),
    }


def complementarity_error(result: SolverResult) -> float:
    return max(result.lcp_complementarity_errors, default=0.0)


def solver_summary(result: SolverResult) -> dict[str, float | int | bool]:
    summary = {
        "obstacle_violation": obstacle_violation(result.values, result.payoff),
        "complementarity_error": complementarity_error(result),
        "max_lcp_natural_residual": max(result.lcp_residuals, default=0.0),
        "max_lcp_obstacle_violation": max(result.lcp_obstacle_violations, default=0.0),
        "max_lcp_continuation_violation": max(
            result.lcp_continuation_violations, default=0.0
        ),
        "max_lcp_iterations": max(result.lcp_iterations) if result.lcp_iterations else 0,
        "all_lcp_converged": all(result.lcp_converged),
        "max_psor_iterations": max(result.psor_iterations) if result.psor_iterations else 0,
        "all_psor_converged": all(result.psor_converged),
    }
    return summary
