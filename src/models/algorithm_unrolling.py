"""Learned feasible warm starts with mandatory natural-residual PSOR cleanup."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Mapping, Sequence

import numpy as np

from solvers.black_scholes import payoff
from solvers.cn_psor import SolverConfig, cn_matrices, time_boundary_values


ArrayPredictor = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class CleanupResult:
    solution: np.ndarray
    iterations: int
    converged: bool
    natural_residual: float
    residual_tolerance: float
    cleanup_seconds: float


@dataclass(frozen=True)
class WarmStartResult:
    initial_guess: np.ndarray
    fallback_used: bool
    fallback_reason: str | None
    inference_seconds: float
    projection_seconds: float
    fallback_seconds: float


@dataclass(frozen=True)
class SurfaceSolveResult:
    config: SolverConfig
    spots: np.ndarray
    taus: np.ndarray
    values: np.ndarray
    payoff: np.ndarray
    cleanup_iterations: tuple[int, ...]
    converged: tuple[bool, ...]
    natural_residuals: tuple[float, ...]
    residual_tolerance: float
    fallback_count: int
    fallback_reasons: tuple[str, ...]
    timings_seconds: Mapping[str, float]

    @property
    def cleanup_iterations_total(self) -> int:
        return int(sum(self.cleanup_iterations))

    @property
    def terminal_values(self) -> np.ndarray:
        return self.values[-1]


@dataclass(frozen=True)
class FeatureNormalizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "FeatureNormalizer":
        values = _finite_matrix(features, "features")
        mean = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
        scale = np.where(scale > 1.0e-12, scale, 1.0)
        return cls(mean=mean, scale=scale)

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = _finite_matrix(features, "features")
        if values.shape[1] != len(self.mean) or self.mean.shape != self.scale.shape:
            raise ValueError("feature normalizer dimension mismatch")
        transformed = (values - self.mean) / self.scale
        if not np.all(np.isfinite(transformed)):
            raise ValueError("normalized features must be finite")
        return transformed

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "mean": [float(value) for value in self.mean],
            "scale": [float(value) for value in self.scale],
        }


def build_warm_start_mlp(input_dim: int, *, hidden: int, depth: int):
    """Build the trainable correction model without importing torch at module import."""
    import torch.nn as nn

    if input_dim < 1 or hidden < 1 or depth < 1:
        raise ValueError("input_dim, hidden, and depth must be positive")
    layers: list[nn.Module] = []
    width = input_dim
    for _ in range(depth):
        layers.extend((nn.Linear(width, hidden), nn.Tanh()))
        width = hidden
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


def project_feasible(values: np.ndarray, obstacle: np.ndarray) -> np.ndarray:
    candidate = np.asarray(values, dtype=float)
    lower_bound = np.asarray(obstacle, dtype=float)
    if candidate.shape != lower_bound.shape:
        raise ValueError("values and obstacle must have identical shapes")
    if not np.all(np.isfinite(lower_bound)):
        raise ValueError("obstacle must be finite")
    return np.maximum(candidate, lower_bound)


def natural_residual_vector(
    matrix: np.ndarray,
    rhs: np.ndarray,
    obstacle: np.ndarray,
    solution: np.ndarray,
) -> np.ndarray:
    a, b, phi, x = _validated_lcp(matrix, rhs, obstacle, solution)
    linear_residual = a @ x - b
    return x - np.maximum(phi, x - linear_residual)


def natural_residual(
    matrix: np.ndarray,
    rhs: np.ndarray,
    obstacle: np.ndarray,
    solution: np.ndarray,
) -> float:
    return float(np.max(np.abs(natural_residual_vector(matrix, rhs, obstacle, solution))))


def projected_sor_cleanup(
    matrix: np.ndarray,
    rhs: np.ndarray,
    obstacle: np.ndarray,
    *,
    omega: float,
    residual_tolerance: float,
    max_iter: int,
    initial_guess: np.ndarray,
) -> CleanupResult:
    """Run ordinary PSOR until the campaign natural-residual criterion passes."""
    a, b, phi, initial = _validated_lcp(matrix, rhs, obstacle, initial_guess)
    if not 0.0 < omega < 2.0:
        raise ValueError("omega must lie in (0, 2)")
    if not np.isfinite(residual_tolerance) or residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be finite and positive")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    diagonal = np.diag(a)
    if np.any(~np.isfinite(diagonal)) or np.any(np.abs(diagonal) <= np.finfo(float).eps):
        raise ValueError("matrix diagonal must be finite and nonzero")

    started_at = perf_counter()
    x = project_feasible(initial, phi)
    residual = natural_residual(a, b, phi, x)
    if residual <= residual_tolerance:
        return CleanupResult(
            solution=x,
            iterations=0,
            converged=True,
            natural_residual=residual,
            residual_tolerance=float(residual_tolerance),
            cleanup_seconds=float(perf_counter() - started_at),
        )

    for iteration in range(1, max_iter + 1):
        old = x.copy()
        for index in range(len(x)):
            left = float(np.dot(a[index, :index], x[:index]))
            right = float(np.dot(a[index, index + 1 :], old[index + 1 :]))
            candidate = (b[index] - left - right) / a[index, index]
            relaxed = old[index] + omega * (candidate - old[index])
            x[index] = max(phi[index], relaxed)
        residual = natural_residual(a, b, phi, x)
        if residual <= residual_tolerance:
            return CleanupResult(
                solution=x,
                iterations=iteration,
                converged=True,
                natural_residual=residual,
                residual_tolerance=float(residual_tolerance),
                cleanup_seconds=float(perf_counter() - started_at),
            )

    return CleanupResult(
        solution=x,
        iterations=max_iter,
        converged=False,
        natural_residual=residual,
        residual_tolerance=float(residual_tolerance),
        cleanup_seconds=float(perf_counter() - started_at),
    )


def build_feasible_warm_start(
    predictor: ArrayPredictor,
    features: np.ndarray,
    *,
    previous: np.ndarray,
    obstacle: np.ndarray,
    strike: float,
    blend: float,
) -> WarmStartResult:
    """Predict a correction, then enforce feasibility or fall back safely."""
    previous_values = np.asarray(previous, dtype=float).reshape(-1)
    obstacle_values = np.asarray(obstacle, dtype=float).reshape(-1)
    if previous_values.shape != obstacle_values.shape:
        raise ValueError("previous and obstacle must have identical shapes")
    if not np.isfinite(strike) or strike <= 0.0:
        raise ValueError("strike must be finite and positive")
    if not np.isfinite(blend) or not 0.0 <= blend <= 1.0:
        raise ValueError("blend must lie in [0, 1]")

    inference_started = perf_counter()
    try:
        prediction = np.asarray(predictor(np.asarray(features, dtype=float)), dtype=float).reshape(-1)
    except Exception as error:
        inference_seconds = float(perf_counter() - inference_started)
        return _fallback_warm_start(
            previous_values,
            obstacle_values,
            reason=f"prediction_error:{type(error).__name__}",
            inference_seconds=inference_seconds,
        )
    inference_seconds = float(perf_counter() - inference_started)

    if prediction.shape != previous_values.shape:
        return _fallback_warm_start(
            previous_values,
            obstacle_values,
            reason="prediction_shape_mismatch",
            inference_seconds=inference_seconds,
        )
    if not np.all(np.isfinite(prediction)):
        return _fallback_warm_start(
            previous_values,
            obstacle_values,
            reason="nonfinite_prediction",
            inference_seconds=inference_seconds,
        )

    projection_started = perf_counter()
    initial_guess = project_feasible(previous_values + blend * strike * prediction, obstacle_values)
    projection_seconds = float(perf_counter() - projection_started)
    return WarmStartResult(
        initial_guess=initial_guess,
        fallback_used=False,
        fallback_reason=None,
        inference_seconds=inference_seconds,
        projection_seconds=projection_seconds,
        fallback_seconds=0.0,
    )


def warm_start_features(
    config: SolverConfig,
    spots: np.ndarray,
    tau_new: float,
    previous: np.ndarray,
    rhs: np.ndarray,
    obstacle: np.ndarray,
) -> np.ndarray:
    interior_spots = np.asarray(spots, dtype=float)[1:-1]
    previous_values = np.asarray(previous, dtype=float).reshape(-1)
    rhs_values = np.asarray(rhs, dtype=float).reshape(-1)
    obstacle_values = np.asarray(obstacle, dtype=float).reshape(-1)
    size = len(interior_spots)
    if any(len(values) != size for values in (previous_values, rhs_values, obstacle_values)):
        raise ValueError("warm-start feature arrays must match the interior grid")
    return np.column_stack(
        [
            interior_spots / config.k,
            np.full(size, tau_new / config.t),
            np.full(size, config.sigma),
            np.full(size, config.r),
            np.full(size, config.q),
            previous_values / config.k,
            rhs_values / config.k,
            obstacle_values / config.k,
        ]
    )


def solve_cn_lcp_surface(
    config: SolverConfig,
    *,
    residual_tolerance: float,
    omega: float,
    max_iter: int,
    warm_start_predictor: ArrayPredictor | None = None,
    blend: float = 1.0,
    unrolled_omegas: np.ndarray | None = None,
) -> SurfaceSolveResult:
    """Solve a CN surface with either the ordinary or learned feasible start."""
    if config.m < 3 or config.n < 1:
        raise ValueError("m must be at least 3 and n at least 1")
    if not config.american:
        raise ValueError("M03 requires an American-option LCP")

    total_started = perf_counter()
    spots, lower, upper, matrix, rhs_matrix = cn_matrices(config)
    taus = np.linspace(0.0, config.t, config.n + 1)
    dt = config.t / config.n
    obstacle_full = payoff(spots, config.k, config.option_type)
    obstacle = obstacle_full[1:-1]
    values = np.zeros((config.n + 1, config.m + 1), dtype=float)
    values[0] = obstacle_full
    iterations: list[int] = []
    converged: list[bool] = []
    residuals: list[float] = []
    fallback_count = 0
    fallback_reasons: list[str] = []
    timings = {
        name: 0.0
        for name in (
            "assembly",
            "feature_construction",
            "inference",
            "projection",
            "fallback",
            "cleanup",
        )
    }

    for step in range(config.n):
        assembly_started = perf_counter()
        tau_old = float(taus[step])
        tau_new = float(taus[step + 1])
        left_old, right_old = time_boundary_values(config, tau_old)
        left_new, right_new = time_boundary_values(config, tau_new)
        values[step, 0] = left_old
        values[step, -1] = right_old
        previous = values[step, 1:-1]
        rhs = rhs_matrix @ previous
        rhs[0] += 0.5 * dt * lower[0] * (left_old + left_new)
        rhs[-1] += 0.5 * dt * upper[-1] * (right_old + right_new)
        timings["assembly"] += float(perf_counter() - assembly_started)

        if warm_start_predictor is not None:
            feature_started = perf_counter()
            features = warm_start_features(config, spots, tau_new, previous, rhs, obstacle)
            timings["feature_construction"] += float(perf_counter() - feature_started)
            warm = build_feasible_warm_start(
                warm_start_predictor,
                features,
                previous=previous,
                obstacle=obstacle,
                strike=config.k,
                blend=blend,
            )
            initial_guess = warm.initial_guess
            timings["inference"] += warm.inference_seconds
            timings["projection"] += warm.projection_seconds
            timings["fallback"] += warm.fallback_seconds
            if warm.fallback_used:
                fallback_count += 1
                fallback_reasons.append(str(warm.fallback_reason))
        elif unrolled_omegas is not None:
            inference_started = perf_counter()
            initial_guess = project_feasible(previous, obstacle)
            initial_guess = unrolled_projected_jacobi(
                matrix, rhs, obstacle, unrolled_omegas, initial_guess
            )
            timings["inference"] += float(perf_counter() - inference_started)
            timings["projection"] += 0.0
        else:
            projection_started = perf_counter()
            initial_guess = project_feasible(previous, obstacle)
            timings["projection"] += float(perf_counter() - projection_started)

        cleanup = projected_sor_cleanup(
            matrix,
            rhs,
            obstacle,
            omega=omega,
            residual_tolerance=residual_tolerance,
            max_iter=max_iter,
            initial_guess=initial_guess,
        )
        cleanup_iterations = cleanup.iterations
        timings["cleanup"] += cleanup.cleanup_seconds

        if (warm_start_predictor is not None or unrolled_omegas is not None) and not cleanup.converged:
            fallback_count += 1
            fallback_reasons.append("cleanup_nonconvergence")
            fallback_started = perf_counter()
            fallback_initial = project_feasible(previous, obstacle)
            timings["fallback"] += float(perf_counter() - fallback_started)
            fallback_cleanup = projected_sor_cleanup(
                matrix,
                rhs,
                obstacle,
                omega=omega,
                residual_tolerance=residual_tolerance,
                max_iter=max_iter,
                initial_guess=fallback_initial,
            )
            cleanup_iterations += fallback_cleanup.iterations
            timings["cleanup"] += fallback_cleanup.cleanup_seconds
            cleanup = fallback_cleanup

        values[step + 1, 0] = left_new
        values[step + 1, -1] = right_new
        values[step + 1, 1:-1] = cleanup.solution
        iterations.append(int(cleanup_iterations))
        converged.append(bool(cleanup.converged))
        residuals.append(float(cleanup.natural_residual))

    timings["total"] = float(perf_counter() - total_started)
    return SurfaceSolveResult(
        config=config,
        spots=spots,
        taus=taus,
        values=values,
        payoff=obstacle_full,
        cleanup_iterations=tuple(iterations),
        converged=tuple(converged),
        natural_residuals=tuple(residuals),
        residual_tolerance=float(residual_tolerance),
        fallback_count=fallback_count,
        fallback_reasons=tuple(fallback_reasons),
        timings_seconds={name: float(value) for name, value in timings.items()},
    )


def surface_training_examples(result: SurfaceSolveResult) -> tuple[np.ndarray, np.ndarray]:
    """Expand one complete solved regime into local features and corrections."""
    config = result.config
    spots, lower, upper, _, rhs_matrix = cn_matrices(config)
    dt = config.t / config.n
    obstacle = result.payoff[1:-1]
    feature_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    for step in range(config.n):
        tau_old = float(result.taus[step])
        tau_new = float(result.taus[step + 1])
        left_old, right_old = time_boundary_values(config, tau_old)
        left_new, right_new = time_boundary_values(config, tau_new)
        previous = result.values[step, 1:-1]
        rhs = rhs_matrix @ previous
        rhs[0] += 0.5 * dt * lower[0] * (left_old + left_new)
        rhs[-1] += 0.5 * dt * upper[-1] * (right_old + right_new)
        feature_batches.append(warm_start_features(config, spots, tau_new, previous, rhs, obstacle))
        target_batches.append((result.values[step + 1, 1:-1] - previous) / config.k)
    return np.vstack(feature_batches), np.concatenate(target_batches)


def _fallback_warm_start(
    previous: np.ndarray,
    obstacle: np.ndarray,
    *,
    reason: str,
    inference_seconds: float,
) -> WarmStartResult:
    fallback_started = perf_counter()
    initial_guess = project_feasible(previous, obstacle)
    fallback_seconds = float(perf_counter() - fallback_started)
    return WarmStartResult(
        initial_guess=initial_guess,
        fallback_used=True,
        fallback_reason=reason,
        inference_seconds=inference_seconds,
        projection_seconds=0.0,
        fallback_seconds=fallback_seconds,
    )


def _validated_lcp(
    matrix: np.ndarray,
    rhs: np.ndarray,
    obstacle: np.ndarray,
    solution: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(matrix, dtype=float)
    b = np.asarray(rhs, dtype=float).reshape(-1)
    phi = np.asarray(obstacle, dtype=float).reshape(-1)
    x = np.asarray(solution, dtype=float).reshape(-1).copy()
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("matrix must be square")
    if any(len(values) != a.shape[0] for values in (b, phi, x)):
        raise ValueError("LCP arrays must match the matrix dimension")
    if not all(np.all(np.isfinite(values)) for values in (a, b, phi, x)):
        raise ValueError("LCP inputs must be finite")
    return a, b, phi, x


@dataclass(frozen=True)
class UnrolledOmegaResult:
    """Trained per-iteration relaxation parameters for unrolled projected Jacobi."""

    omegas: np.ndarray
    training_seconds: float


def train_unrolled_omegas(
    train_lcp_data: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    K: int,
    *,
    learning_rate: float,
    epochs: int,
    seed: int,
) -> UnrolledOmegaResult:
    """Train K relaxation parameters for unrolled projected Jacobi.

    Each training sample is (A, b, phi, V_prev, V_true) where:
    - A: LCP matrix (d x d)
    - b: RHS vector (d,)
    - phi: obstacle vector (d,)
    - V_prev: previous-time-step solution (d,)
    - V_true: tight-tolerance PSOR solution (d,)

    Training minimizes MSE(V_K, V_true) over the K-iteration unrolled Jacobi
    trajectory started from max(phi, V_prev).
    """
    import torch

    if K < 1:
        raise ValueError("K must be positive")
    if not train_lcp_data:
        raise ValueError("training data must not be empty")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("epochs must be a positive integer")

    torch.manual_seed(seed)
    omegas_param = torch.nn.Parameter(torch.ones(K, dtype=torch.float64))
    optimizer = torch.optim.Adam([omegas_param], lr=float(learning_rate))

    training_started = perf_counter()
    for _epoch in range(epochs):
        epoch_loss = 0.0
        for A_np, b_np, phi_np, vprev_np, vtrue_np in train_lcp_data:
            A_t = torch.as_tensor(A_np, dtype=torch.float64)
            b_t = torch.as_tensor(b_np, dtype=torch.float64)
            phi_t = torch.as_tensor(phi_np, dtype=torch.float64)
            V = torch.max(phi_t, torch.as_tensor(vprev_np, dtype=torch.float64))
            V_true = torch.as_tensor(vtrue_np, dtype=torch.float64)

            diag = torch.diag(A_t)
            for omega in omegas_param:
                residual = b_t - A_t @ V
                direction = residual / diag
                V = torch.max(phi_t, V + omega * direction)

            loss = torch.nn.functional.mse_loss(V, V_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())

        with torch.no_grad():
            omegas_param.clamp_(0.01, 1.99)

    training_seconds = float(perf_counter() - training_started)
    return UnrolledOmegaResult(
        omegas=omegas_param.detach().cpu().numpy().copy(),
        training_seconds=training_seconds,
    )


def unrolled_projected_jacobi(
    matrix: np.ndarray,
    rhs: np.ndarray,
    obstacle: np.ndarray,
    omegas: np.ndarray,
    initial_guess: np.ndarray,
) -> np.ndarray:
    """Run K projected Jacobi iterations with learned omega parameters.

    Each iteration:
        V_{k+1} = max(obstacle, V_k + omega_k * (rhs - matrix @ V_k) / diag(matrix))

    This is the projected Jacobi over-relaxation (PJOR) from Schwab & Stein (2022).
    The obstacle projection is exact (no learned approximation).
    """
    A, b, phi, x = _validated_lcp(matrix, rhs, obstacle, initial_guess)
    omega_values = np.asarray(omegas, dtype=float).reshape(-1)
    if omega_values.size == 0:
        raise ValueError("omegas must not be empty")
    if not np.all(np.isfinite(omega_values)):
        raise ValueError("omegas must be finite")
    if np.any(omega_values <= 0.0) or np.any(omega_values >= 2.0):
        raise ValueError("omegas must lie in (0, 2)")

    diag = np.diag(A)
    if np.any(~np.isfinite(diag)) or np.any(np.abs(diag) <= np.finfo(float).eps):
        raise ValueError("matrix diagonal must be finite and nonzero")

    V = x.copy()
    for omega in omega_values:
        residual = b - A @ V
        direction = residual / diag
        V = np.maximum(phi, V + omega * direction)

    return V


def extract_lcp_training_data(
    result: "SurfaceSolveResult",
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Extract per-time-step LCP training tuples from a completed surface solve.

    Returns list of (A, b, phi, V_prev, V_true) for each time step.
    V_prev is the solution at tau_n; V_true is the solution at tau_{n+1}.
    """
    from solvers.cn_psor import cn_matrices, time_boundary_values

    config = result.config
    spots, lower, upper, matrix, rhs_matrix = cn_matrices(config)
    dt = config.t / config.n
    obstacle = result.payoff[1:-1]
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for step in range(config.n):
        tau_old = float(result.taus[step])
        tau_new = float(result.taus[step + 1])
        left_old, right_old = time_boundary_values(config, tau_old)
        left_new, right_new = time_boundary_values(config, tau_new)
        previous = result.values[step, 1:-1].copy()
        rhs = rhs_matrix @ previous
        rhs[0] += 0.5 * dt * lower[0] * (left_old + left_new)
        rhs[-1] += 0.5 * dt * upper[-1] * (right_old + right_new)
        v_true = result.values[step + 1, 1:-1].copy()
        samples.append((matrix.copy(), rhs.copy(), obstacle.copy(), previous.copy(), v_true))

    return samples




def _finite_matrix(values: np.ndarray, label: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{label} must be a non-empty matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be finite")
    return matrix
