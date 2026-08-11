"""Finite-difference and LCP solvers for one-dimensional American options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from solvers.black_scholes import OptionType, payoff
from solvers.boundary_conditions import boundary_values


TimeScheme = Literal["cn", "rannacher_cn", "sdirk2"]
LCPSolver = Literal["psor", "policy_iteration", "penalty_newton"]


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
    time_scheme: TimeScheme = "cn"
    lcp_solver: LCPSolver = "psor"
    penalty: float = 1e8


@dataclass
class SolverResult:
    config: SolverConfig
    spots: np.ndarray
    taus: np.ndarray
    values: np.ndarray
    payoff: np.ndarray
    lcp_iterations: list[int]
    lcp_converged: list[bool]
    lcp_residuals: list[float]
    lcp_complementarity_errors: list[float]
    lcp_obstacle_violations: list[float]
    lcp_continuation_violations: list[float]
    time_substeps: list[int]

    @property
    def terminal_values(self) -> np.ndarray:
        return self.values[-1]

    @property
    def psor_iterations(self) -> list[int]:
        """Compatibility alias retained for existing experiment scripts."""
        return self.lcp_iterations

    @property
    def psor_converged(self) -> list[bool]:
        """Compatibility alias retained for existing experiment scripts."""
        return self.lcp_converged


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
    _validate_lcp_inputs(a, b, obstacle)
    x = np.maximum(obstacle.copy(), b.copy()) if x0 is None else np.maximum(obstacle, x0.copy())
    n = len(x)
    diff = float("inf")
    for iteration in range(1, max_iter + 1):
        old = x.copy()
        for i in range(n):
            left = np.dot(a[i, :i], x[:i])
            right = np.dot(a[i, i + 1 :], old[i + 1 :])
            candidate = (b[i] - left - right) / a[i, i]
            relaxed = old[i] + omega * (candidate - old[i])
            x[i] = max(obstacle[i], relaxed)
        diff = float(np.max(np.abs(x - old)))
        residual = lcp_natural_residual(a, x, b, obstacle)
        if diff < tol and residual < tol:
            return x, {
                "iterations": iteration,
                "converged": True,
                "last_update": diff,
                "natural_residual": residual,
            }
    return x, {
        "iterations": max_iter,
        "converged": False,
        "last_update": diff,
        "natural_residual": lcp_natural_residual(a, x, b, obstacle),
    }


def _validate_lcp_inputs(a: np.ndarray, b: np.ndarray, obstacle: np.ndarray) -> None:
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("a must be a square matrix")
    if b.shape != (a.shape[0],) or obstacle.shape != b.shape:
        raise ValueError("b and obstacle must match the matrix dimension")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(obstacle).all():
        raise ValueError("LCP inputs must be finite")
    if np.any(np.diag(a) <= 0.0):
        raise ValueError("LCP matrix must have a positive diagonal")


def _matvec(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Deterministic dense matrix-vector product without Accelerate FP-status leakage."""
    return np.einsum("ij,j->i", a, x, optimize=False)


def lcp_natural_residual(a: np.ndarray, x: np.ndarray, b: np.ndarray, obstacle: np.ndarray) -> float:
    """Infinity norm of min(Ax-b, x-obstacle), the natural LCP residual."""
    _validate_lcp_inputs(a, b, obstacle)
    if x.shape != b.shape or not np.isfinite(x).all():
        raise ValueError("x must be finite and match b")
    return float(np.max(np.abs(np.minimum(_matvec(a, x) - b, x - obstacle))))


def lcp_complementarity_error(
    a: np.ndarray,
    x: np.ndarray,
    b: np.ndarray,
    obstacle: np.ndarray,
) -> float:
    """Maximum absolute product of obstacle gap and continuation residual."""
    _validate_lcp_inputs(a, b, obstacle)
    if x.shape != b.shape or not np.isfinite(x).all():
        raise ValueError("x must be finite and match b")
    return float(np.max(np.abs((x - obstacle) * (_matvec(a, x) - b))))


def policy_iteration_lcp(
    a: np.ndarray,
    b: np.ndarray,
    obstacle: np.ndarray,
    *,
    tol: float,
    max_iter: int,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Solve the discrete obstacle problem by active-set policy iteration."""
    _validate_lcp_inputs(a, b, obstacle)
    x = np.maximum(obstacle, b) if x0 is None else np.maximum(obstacle, x0.copy())
    previous_policy: np.ndarray | None = None
    update = float("inf")
    for iteration in range(1, max_iter + 1):
        continuation_residual = _matvec(a, x) - b
        obstacle_gap = x - obstacle
        exercise = obstacle_gap <= continuation_residual
        system = a.copy()
        rhs = b.copy()
        system[exercise, :] = 0.0
        system[exercise, exercise] = 1.0
        rhs[exercise] = obstacle[exercise]
        candidate = np.linalg.solve(system, rhs)
        update = float(np.max(np.abs(candidate - x)))
        x = candidate
        residual = lcp_natural_residual(a, x, b, obstacle)
        stable = previous_policy is not None and np.array_equal(exercise, previous_policy)
        if residual <= tol and (update <= tol or stable):
            return x, {
                "iterations": iteration,
                "converged": True,
                "last_update": update,
                "natural_residual": residual,
            }
        previous_policy = exercise
    return x, {
        "iterations": max_iter,
        "converged": False,
        "last_update": update,
        "natural_residual": lcp_natural_residual(a, x, b, obstacle),
    }


def penalty_newton_lcp(
    a: np.ndarray,
    b: np.ndarray,
    obstacle: np.ndarray,
    *,
    penalty: float,
    tol: float,
    max_iter: int,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Solve a penalized obstacle equation with a semismooth Newton iteration."""
    _validate_lcp_inputs(a, b, obstacle)
    if not np.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("penalty must be positive and finite")
    x = np.maximum(obstacle, b) if x0 is None else x0.copy()
    previous_active: np.ndarray | None = None
    update = float("inf")
    equation_residual = float("inf")
    for iteration in range(1, max_iter + 1):
        active = x < obstacle
        system = a.copy()
        diagonal = np.diag_indices_from(system)
        system[diagonal] += penalty * active.astype(float)
        rhs = b + penalty * active.astype(float) * obstacle
        candidate = np.linalg.solve(system, rhs)
        update = float(np.max(np.abs(candidate - x)))
        x = candidate
        equation = _matvec(a, x) - b - penalty * np.maximum(obstacle - x, 0.0)
        equation_residual = float(np.max(np.abs(equation)))
        natural_residual = lcp_natural_residual(a, x, b, obstacle)
        stable = previous_active is not None and np.array_equal(active, previous_active)
        if stable and (equation_residual <= tol or update <= tol):
            return x, {
                "iterations": iteration,
                "converged": True,
                "last_update": update,
                "equation_residual": equation_residual,
                "natural_residual": natural_residual,
            }
        previous_active = active
    return x, {
        "iterations": max_iter,
        "converged": False,
        "last_update": update,
        "equation_residual": equation_residual,
        "natural_residual": lcp_natural_residual(a, x, b, obstacle),
    }


def brennan_schwartz_lcp(
    a: np.ndarray,
    b: np.ndarray,
    obstacle: np.ndarray,
    *,
    tol: float = 1e-10,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Projected tridiagonal Brennan--Schwartz solve for ordered put-style LCPs."""
    _validate_lcp_inputs(a, b, obstacle)
    if not np.allclose(a, np.triu(np.tril(a, 1), -1)):
        raise ValueError("Brennan-Schwartz requires a tridiagonal matrix")
    diagonal = np.diag(a).copy()
    lower = np.diag(a, -1).copy()
    upper = np.diag(a, 1).copy()
    rhs = b.copy()
    for i in range(1, len(diagonal)):
        multiplier = lower[i - 1] / diagonal[i - 1]
        diagonal[i] -= multiplier * upper[i - 1]
        rhs[i] -= multiplier * rhs[i - 1]
    x = np.empty_like(rhs)
    x[-1] = max(obstacle[-1], rhs[-1] / diagonal[-1])
    for i in range(len(x) - 2, -1, -1):
        x[i] = max(obstacle[i], (rhs[i] - upper[i] * x[i + 1]) / diagonal[i])
    residual = lcp_natural_residual(a, x, b, obstacle)
    return x, {
        "iterations": 1,
        "converged": residual <= tol,
        "last_update": 0.0,
        "natural_residual": residual,
    }


def _solve_lcp(
    cfg: SolverConfig,
    a: np.ndarray,
    rhs: np.ndarray,
    obstacle: np.ndarray,
    x0: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    if cfg.lcp_solver == "psor":
        solution, info = psor_lcp(
            a,
            rhs,
            obstacle,
            omega=cfg.omega,
            tol=cfg.tol,
            max_iter=cfg.max_iter,
            x0=x0,
        )
    elif cfg.lcp_solver == "policy_iteration":
        solution, info = policy_iteration_lcp(
            a,
            rhs,
            obstacle,
            tol=cfg.tol,
            max_iter=cfg.max_iter,
            x0=x0,
        )
    elif cfg.lcp_solver == "penalty_newton":
        solution, info = penalty_newton_lcp(
            a,
            rhs,
            obstacle,
            penalty=cfg.penalty,
            tol=cfg.tol,
            max_iter=cfg.max_iter,
            x0=x0,
        )
    else:
        raise ValueError(f"unknown lcp_solver: {cfg.lcp_solver}")
    continuation_residual = _matvec(a, solution) - rhs
    info = dict(info)
    info.update(
        {
            "natural_residual": lcp_natural_residual(a, solution, rhs, obstacle),
            "complementarity_error": lcp_complementarity_error(
                a, solution, rhs, obstacle
            ),
            "obstacle_violation": float(np.max(np.maximum(obstacle - solution, 0.0))),
            "continuation_violation": float(
                np.max(np.maximum(-continuation_residual, 0.0))
            ),
        }
    )
    return solution, info


def _boundary_source(
    cfg: SolverConfig,
    tau: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    source = np.zeros(cfg.m - 1, dtype=float)
    left, right = time_boundary_values(cfg, tau)
    source[0] = lower[0] * left
    source[-1] = upper[-1] * right
    return source


def _linear_or_lcp_solve(
    cfg: SolverConfig,
    a: np.ndarray,
    rhs: np.ndarray,
    obstacle: np.ndarray,
    x0: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    if cfg.american:
        return _solve_lcp(cfg, a, rhs, obstacle, x0)
    solution = np.linalg.solve(a, rhs)
    return solution, {
        "iterations": 0,
        "converged": True,
        "last_update": 0.0,
        "natural_residual": 0.0,
        "complementarity_error": 0.0,
        "obstacle_violation": 0.0,
        "continuation_violation": 0.0,
    }


def _theta_step(
    cfg: SolverConfig,
    u_old: np.ndarray,
    obstacle: np.ndarray,
    l_mat: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    tau_old: float,
    tau_new: float,
    theta: float,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    dt = tau_new - tau_old
    eye = np.eye(len(u_old))
    a = eye - theta * dt * l_mat
    b_mat = eye + (1.0 - theta) * dt * l_mat
    rhs = _matvec(b_mat, u_old)
    rhs += dt * (
        (1.0 - theta) * _boundary_source(cfg, tau_old, lower, upper)
        + theta * _boundary_source(cfg, tau_new, lower, upper)
    )
    return _linear_or_lcp_solve(cfg, a, rhs, obstacle, u_old)


def _sdirk2_step(
    cfg: SolverConfig,
    u_old: np.ndarray,
    obstacle: np.ndarray,
    l_mat: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    tau_old: float,
    tau_new: float,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    dt = tau_new - tau_old
    gamma = 1.0 - 1.0 / np.sqrt(2.0)
    tau_stage = tau_old + gamma * dt
    a = np.eye(len(u_old)) - gamma * dt * l_mat
    source_stage = _boundary_source(cfg, tau_stage, lower, upper)
    rhs_stage = u_old + gamma * dt * source_stage
    stage, info_stage = _linear_or_lcp_solve(cfg, a, rhs_stage, obstacle, u_old)

    stage_reaction = np.zeros_like(stage)
    if cfg.american:
        stage_reaction = (_matvec(a, stage) - rhs_stage) / (gamma * dt)
    rhs_final = u_old + (1.0 - gamma) * dt * (
        _matvec(l_mat, stage) + source_stage + stage_reaction
    )
    rhs_final += gamma * dt * _boundary_source(cfg, tau_new, lower, upper)
    solution, info_final = _linear_or_lcp_solve(cfg, a, rhs_final, obstacle, stage)
    info = {
        "iterations": int(info_stage["iterations"]) + int(info_final["iterations"]),
        "converged": bool(info_stage["converged"]) and bool(info_final["converged"]),
        "last_update": max(float(info_stage["last_update"]), float(info_final["last_update"])),
        "natural_residual": max(
            float(info_stage.get("natural_residual", 0.0)),
            float(info_final.get("natural_residual", 0.0)),
        ),
        "complementarity_error": max(
            float(info_stage.get("complementarity_error", 0.0)),
            float(info_final.get("complementarity_error", 0.0)),
        ),
        "obstacle_violation": max(
            float(info_stage.get("obstacle_violation", 0.0)),
            float(info_final.get("obstacle_violation", 0.0)),
        ),
        "continuation_violation": max(
            float(info_stage.get("continuation_violation", 0.0)),
            float(info_final.get("continuation_violation", 0.0)),
        ),
    }
    return solution, info


def solve_option(cfg: SolverConfig) -> SolverResult:
    if cfg.m < 3 or cfg.n < 1:
        raise ValueError("m must be at least 3 and n at least 1")
    if cfg.option_type not in ("put", "call"):
        raise ValueError("option_type must be 'put' or 'call'")
    if not 0.0 < cfg.omega < 2.0:
        raise ValueError("PSOR omega must be in (0, 2)")
    if cfg.time_scheme not in ("cn", "rannacher_cn", "sdirk2"):
        raise ValueError(f"unknown time_scheme: {cfg.time_scheme}")
    if cfg.lcp_solver not in ("psor", "policy_iteration", "penalty_newton"):
        raise ValueError(f"unknown lcp_solver: {cfg.lcp_solver}")
    if cfg.rannacher_steps < 0 or cfg.rannacher_steps > cfg.n:
        raise ValueError("rannacher_steps must lie between zero and n")
    if cfg.time_scheme == "rannacher_cn" and cfg.rannacher_steps < 1:
        raise ValueError("rannacher_cn requires at least one rannacher step")

    spots = np.linspace(0.0, cfg.s_max, cfg.m + 1)
    taus = np.linspace(0.0, cfg.t, cfg.n + 1)
    lower, diag, upper = _operator_coefficients(cfg, spots)
    l_mat = fd_operator_matrix(cfg, spots)

    phi = payoff(spots, cfg.k, cfg.option_type)
    values = np.zeros((cfg.n + 1, cfg.m + 1), dtype=float)
    values[0] = phi
    lcp_iterations: list[int] = []
    lcp_converged: list[bool] = []
    lcp_residuals: list[float] = []
    lcp_complementarity_errors: list[float] = []
    lcp_obstacle_violations: list[float] = []
    lcp_continuation_violations: list[float] = []
    time_substeps: list[int] = []

    for step in range(cfg.n):
        tau_old = float(taus[step])
        tau_new = taus[step + 1]
        left_old, right_old = time_boundary_values(cfg, tau_old)
        left_new, right_new = time_boundary_values(cfg, tau_new)
        values[step, 0] = left_old
        values[step, -1] = right_old

        obstacle = phi[1:-1]
        u_old = values[step, 1:-1]
        if cfg.time_scheme == "rannacher_cn" and step < cfg.rannacher_steps:
            tau_mid = tau_old + 0.5 * (tau_new - tau_old)
            midpoint, info_first = _theta_step(
                cfg, u_old, obstacle, l_mat, lower, upper, tau_old, tau_mid, theta=1.0
            )
            solution, info_second = _theta_step(
                cfg, midpoint, obstacle, l_mat, lower, upper, tau_mid, tau_new, theta=1.0
            )
            info = {
                "iterations": int(info_first["iterations"]) + int(info_second["iterations"]),
                "converged": bool(info_first["converged"]) and bool(info_second["converged"]),
                "natural_residual": max(
                    float(info_first.get("natural_residual", 0.0)),
                    float(info_second.get("natural_residual", 0.0)),
                ),
                "complementarity_error": max(
                    float(info_first.get("complementarity_error", 0.0)),
                    float(info_second.get("complementarity_error", 0.0)),
                ),
                "obstacle_violation": max(
                    float(info_first.get("obstacle_violation", 0.0)),
                    float(info_second.get("obstacle_violation", 0.0)),
                ),
                "continuation_violation": max(
                    float(info_first.get("continuation_violation", 0.0)),
                    float(info_second.get("continuation_violation", 0.0)),
                ),
            }
            time_substeps.append(2)
        elif cfg.time_scheme == "sdirk2":
            solution, info = _sdirk2_step(
                cfg, u_old, obstacle, l_mat, lower, upper, tau_old, tau_new
            )
            time_substeps.append(2)
        else:
            solution, info = _theta_step(
                cfg, u_old, obstacle, l_mat, lower, upper, tau_old, tau_new, theta=0.5
            )
            time_substeps.append(1)

        lcp_iterations.append(int(info["iterations"]))
        lcp_converged.append(bool(info["converged"]))
        lcp_residuals.append(float(info.get("natural_residual", 0.0)))
        lcp_complementarity_errors.append(float(info.get("complementarity_error", 0.0)))
        lcp_obstacle_violations.append(float(info.get("obstacle_violation", 0.0)))
        lcp_continuation_violations.append(float(info.get("continuation_violation", 0.0)))

        values[step + 1, 0] = left_new
        values[step + 1, -1] = right_new
        values[step + 1, 1:-1] = solution

    return SolverResult(
        cfg,
        spots,
        taus,
        values,
        phi,
        lcp_iterations,
        lcp_converged,
        lcp_residuals,
        lcp_complementarity_errors,
        lcp_obstacle_violations,
        lcp_continuation_violations,
        time_substeps,
    )


def cn_matrices(cfg: SolverConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spots = np.linspace(0.0, cfg.s_max, cfg.m + 1)
    dt = cfg.t / cfg.n
    lower, diag, upper = _operator_coefficients(cfg, spots)
    l_mat = fd_operator_matrix(cfg, spots)
    eye = np.eye(cfg.m - 1)
    return spots, lower, upper, eye - 0.5 * dt * l_mat, eye + 0.5 * dt * l_mat
