"""Numerical validation diagnostics for CN/PSOR outputs."""

from __future__ import annotations

import numpy as np

from diagnostics.shape import obstacle_violation
from solvers.black_scholes import black_scholes_price
from solvers.boundary_conditions import boundary_values
from solvers.cn_psor import SolverConfig, SolverResult, cn_matrices


def closed_form_error(result: SolverResult) -> dict[str, float]:
    cfg = result.config
    exact = black_scholes_price(result.spots, cfg.k, cfg.t, cfg.r, cfg.q, cfg.sigma, cfg.option_type)
    err = np.abs(result.terminal_values - exact)
    return {"max_error": float(np.max(err)), "rmse": float(np.sqrt(np.mean(err**2)))}


def complementarity_error(result: SolverResult) -> float:
    cfg = result.config
    spots, lower, upper, a, b_mat = cn_matrices(cfg)
    max_error = 0.0
    for step in range(1, len(result.taus)):
        tau_old = result.taus[step - 1]
        tau_new = result.taus[step]
        left_old, right_old = boundary_values(cfg.option_type, tau_old, cfg.k, cfg.s_max, cfg.r, cfg.q)
        left_new, right_new = boundary_values(cfg.option_type, tau_new, cfg.k, cfg.s_max, cfg.r, cfg.q)
        rhs = b_mat @ result.values[step - 1, 1:-1]
        dt = cfg.t / cfg.n
        rhs[0] += 0.5 * dt * lower[0] * (left_old + left_new)
        rhs[-1] += 0.5 * dt * upper[-1] * (right_old + right_new)
        residual = a @ result.values[step, 1:-1] - rhs
        gap = result.values[step, 1:-1] - result.payoff[1:-1]
        max_error = max(max_error, float(np.max(np.abs(gap * residual))))
    return max_error


def solver_summary(result: SolverResult) -> dict[str, float | int | bool]:
    return {
        "obstacle_violation": obstacle_violation(result.values, result.payoff),
        "complementarity_error": complementarity_error(result),
        "max_psor_iterations": max(result.psor_iterations) if result.psor_iterations else 0,
        "all_psor_converged": all(result.psor_converged),
    }
