"""Crank-Nicolson / PSOR benchmark solver for American options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from solvers.black_scholes import OptionType, payoff
from solvers.boundary_conditions import boundary_values


@dataclass(frozen=True)
class SolverConfig:
    option_type: OptionType = "put"
    k: float = 1.0
    t: float = 1.0
    r: float = 0.05
    q: float = 0.0
    sigma: float = 0.2
    s_max: float = 4.0
    m: int = 160
    n: int = 160
    omega: float = 1.2
    tol: float = 1e-10
    max_iter: int = 20_000
    american: bool = True
    rannacher_steps: int = 0


@dataclass
class SolverResult:
    config: SolverConfig
    spots: np.ndarray
    taus: np.ndarray
    values: np.ndarray
    payoff: np.ndarray
    psor_iterations: list[int]
    psor_converged: list[bool]

    @property
    def terminal_values(self) -> np.ndarray:
        return self.values[-1]


def _operator_coefficients(cfg: SolverConfig, spots: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = spots[1] - spots[0]
    interior = spots[1:-1]
    diffusion = 0.5 * cfg.sigma**2 * interior**2
    drift = (cfg.r - cfg.q) * interior
    lower = diffusion / ds**2 - drift / (2.0 * ds)
    diag = -2.0 * diffusion / ds**2 - cfg.r
    upper = diffusion / ds**2 + drift / (2.0 * ds)
    return lower, diag, upper


def _dense_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray) -> np.ndarray:
    size = len(diag)
    mat = np.diag(diag)
    if size > 1:
        mat += np.diag(upper[:-1], 1)
        mat += np.diag(lower[1:], -1)
    return mat


def fd_operator_matrix(cfg: SolverConfig, spots: np.ndarray | None = None) -> np.ndarray:
    """Return the interior finite-difference matrix for the Black-Scholes operator L."""
    if spots is None:
        spots = np.linspace(0.0, cfg.s_max, cfg.m + 1)
    lower, diag, upper = _operator_coefficients(cfg, spots)
    return _dense_tridiagonal(lower, diag, upper)


def time_boundary_values(cfg: SolverConfig, tau: float) -> tuple[float, float]:
    if cfg.american:
        return boundary_values(cfg.option_type, tau, cfg.k, cfg.s_max, cfg.r, cfg.q)
    if cfg.option_type == "put":
        return cfg.k * np.exp(-cfg.r * tau), 0.0
    if cfg.option_type == "call":
        return 0.0, cfg.s_max * np.exp(-cfg.q * tau) - cfg.k * np.exp(-cfg.r * tau)
    raise ValueError(f"unknown option_type: {cfg.option_type}")


def psor_lcp(
    a: np.ndarray,
    b: np.ndarray,
    obstacle: np.ndarray,
    *,
    omega: float,
    tol: float,
    max_iter: int,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Solve A x >= b, x >= obstacle, (x-obstacle)'(Ax-b)=0 by PSOR."""
    x = np.maximum(obstacle.copy(), b.copy()) if x0 is None else np.maximum(obstacle, x0.copy())
    n = len(x)
    for iteration in range(1, max_iter + 1):
        old = x.copy()
        for i in range(n):
            left = np.dot(a[i, :i], x[:i])
            right = np.dot(a[i, i + 1 :], old[i + 1 :])
            candidate = (b[i] - left - right) / a[i, i]
            relaxed = old[i] + omega * (candidate - old[i])
            x[i] = max(obstacle[i], relaxed)
        diff = float(np.max(np.abs(x - old)))
        if diff < tol:
            return x, {"iterations": iteration, "converged": True, "last_update": diff}
    return x, {"iterations": max_iter, "converged": False, "last_update": diff}


def solve_option(cfg: SolverConfig) -> SolverResult:
    if cfg.m < 3 or cfg.n < 1:
        raise ValueError("m must be at least 3 and n at least 1")
    if cfg.option_type not in ("put", "call"):
        raise ValueError("option_type must be 'put' or 'call'")
    if not 0.0 < cfg.omega < 2.0:
        raise ValueError("PSOR omega must be in (0, 2)")

    spots = np.linspace(0.0, cfg.s_max, cfg.m + 1)
    taus = np.linspace(0.0, cfg.t, cfg.n + 1)
    dt = taus[1] - taus[0]
    lower, diag, upper = _operator_coefficients(cfg, spots)
    l_mat = fd_operator_matrix(cfg, spots)
    eye = np.eye(cfg.m - 1)
    a = eye - 0.5 * dt * l_mat
    b_mat = eye + 0.5 * dt * l_mat

    phi = payoff(spots, cfg.k, cfg.option_type)
    values = np.zeros((cfg.n + 1, cfg.m + 1), dtype=float)
    values[0] = phi
    psor_iterations: list[int] = []
    psor_converged: list[bool] = []

    for step in range(cfg.n):
        tau_old = taus[step]
        tau_new = taus[step + 1]
        left_old, right_old = time_boundary_values(cfg, tau_old)
        left_new, right_new = time_boundary_values(cfg, tau_new)
        values[step, 0] = left_old
        values[step, -1] = right_old

        rhs = b_mat @ values[step, 1:-1]
        rhs[0] += 0.5 * dt * lower[0] * (left_old + left_new)
        rhs[-1] += 0.5 * dt * upper[-1] * (right_old + right_new)

        obstacle = phi[1:-1]
        if cfg.american:
            solution, info = psor_lcp(
                a,
                rhs,
                obstacle,
                omega=cfg.omega,
                tol=cfg.tol,
                max_iter=cfg.max_iter,
                x0=values[step, 1:-1],
            )
            psor_iterations.append(int(info["iterations"]))
            psor_converged.append(bool(info["converged"]))
        else:
            solution = np.linalg.solve(a, rhs)
            psor_iterations.append(0)
            psor_converged.append(True)

        values[step + 1, 0] = left_new
        values[step + 1, -1] = right_new
        values[step + 1, 1:-1] = solution

    return SolverResult(cfg, spots, taus, values, phi, psor_iterations, psor_converged)


def cn_matrices(cfg: SolverConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spots = np.linspace(0.0, cfg.s_max, cfg.m + 1)
    dt = cfg.t / cfg.n
    lower, diag, upper = _operator_coefficients(cfg, spots)
    l_mat = fd_operator_matrix(cfg, spots)
    eye = np.eye(cfg.m - 1)
    return spots, lower, upper, eye - 0.5 * dt * l_mat, eye + 0.5 * dt * l_mat
