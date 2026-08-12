"""Tests for unrolled projected Jacobi (PJOR) solver."""

import numpy as np

from models.algorithm_unrolling import (
    UnrolledOmegaResult,
    extract_lcp_training_data,
    natural_residual,
    natural_residual_vector,
    project_feasible,
    projected_sor_cleanup,
    solve_cn_lcp_surface,
    train_unrolled_omegas,
    unrolled_projected_jacobi,
)
from solvers.cn_psor import SolverConfig


def _small_lcp():
    """3x3 tridiagonal LCP with known properties."""
    matrix = np.array(
        [
            [2.0, -0.5, 0.0],
            [-0.5, 2.0, -0.5],
            [0.0, -0.5, 2.0],
        ]
    )
    rhs = np.array([0.1, 0.8, 0.1])
    obstacle = np.array([0.2, 0.4, 0.2])
    return matrix, rhs, obstacle


def test_unrolled_jacobi_monotonically_reduces_residual():
    """K unrolled iterations should not increase the natural residual."""
    matrix, rhs, obstacle = _small_lcp()
    x0 = project_feasible(obstacle, obstacle)  # start at obstacle

    # Use omega=1.0 (standard Jacobi) for 3 iterations
    omegas = np.array([1.0, 1.0, 1.0])
    result = unrolled_projected_jacobi(matrix, rhs, obstacle, omegas, x0)

    # Natural residual should be reduced or unchanged
    initial_residual = natural_residual(matrix, rhs, obstacle, x0)
    final_residual = natural_residual(matrix, rhs, obstacle, result)
    assert final_residual <= initial_residual + 1e-14


def test_unrolled_jacobi_preserves_obstacle_feasibility():
    """Every unrolled iterate must satisfy V >= obstacle."""
    matrix, rhs, obstacle = _small_lcp()
    x0 = project_feasible(np.array([0.0, 0.3, 0.0]), obstacle)

    omegas = np.array([0.5, 1.0, 1.5])
    result = unrolled_projected_jacobi(matrix, rhs, obstacle, omegas, x0)

    assert np.all(result >= obstacle - 1e-14)


def test_unrolled_jacobi_with_zero_omegas_preserves_initial():
    """omega=0 should leave the initial guess unchanged."""
    matrix, rhs, obstacle = _small_lcp()
    x0 = obstacle.copy()

    eps = 1e-6
    omegas = np.array([eps, eps, eps])
    result = unrolled_projected_jacobi(matrix, rhs, obstacle, omegas, x0)

    # With epsilon omega, change should be bounded by O(eps)
    change = np.max(np.abs(result - x0))
    assert change < 1e-3, f"change {change} too large for omega={eps}"


def test_trained_omegas_stay_in_valid_range():
    """Trained omegas must remain in (0, 2)."""
    matrix, rhs, obstacle = _small_lcp()
    x0 = obstacle.copy()
    # True solution: run PSOR to tight tolerance
    true_solution, _ = _psor_solve(matrix, rhs, obstacle)

    train_data = [(matrix, rhs, obstacle, x0, true_solution)]

    for K in [2, 3, 5]:
        result = train_unrolled_omegas(
            train_data,
            K=K,
            learning_rate=0.01,
            epochs=10,
            seed=42,
        )
        assert isinstance(result, UnrolledOmegaResult)
        assert len(result.omegas) == K
        assert np.all(result.omegas > 0.0)
        assert np.all(result.omegas < 2.0)
        assert result.training_seconds >= 0.0


def test_trained_omegas_improve_solution():
    """Training should produce omegas that reduce MSE vs true solution."""
    matrix, rhs, obstacle = _small_lcp()
    x0 = obstacle.copy()
    true_solution, _ = _psor_solve(matrix, rhs, obstacle)

    train_data = [(matrix, rhs, obstacle, x0, true_solution)]

    result = train_unrolled_omegas(
        train_data,
        K=5,
        learning_rate=0.01,
        epochs=50,
        seed=42,
    )

    improved = unrolled_projected_jacobi(matrix, rhs, obstacle, result.omegas, x0)

    # Trained omegas should produce better solution than untrained (omega=1.0)
    untrained = unrolled_projected_jacobi(
        matrix, rhs, obstacle, np.ones(5), x0
    )

    trained_error = np.max(np.abs(improved - true_solution))
    untrained_error = np.max(np.abs(untrained - true_solution))

    # Trained should be at least as good
    assert trained_error <= untrained_error + 1e-12


def test_unrolled_surface_solve_produces_valid_output():
    """Unrolled solver integrated into surface solve must produce valid output."""
    config = SolverConfig(
        option_type="put",
        m=10,
        n=4,
        omega=1.2,
        tol=1e-9,
        max_iter=5000,
    )

    # Use untrained omegas (omega=1.0 for all iterations)
    omegas = np.ones(3)

    result = solve_cn_lcp_surface(
        config,
        residual_tolerance=config.tol,
        omega=config.omega,
        max_iter=config.max_iter,
        unrolled_omegas=omegas,
    )

    assert result.residual_tolerance == config.tol
    assert result.cleanup_iterations_total == sum(result.cleanup_iterations)
    assert result.cleanup_iterations_total > 0
    assert max(result.natural_residuals) <= config.tol
    assert np.all(result.values >= result.payoff[None, :] - 1e-12)

    # Unrolled inference time should be recorded
    assert result.timings_seconds["inference"] > 0.0
    # No feature construction for unrolled
    assert result.timings_seconds["feature_construction"] == 0.0


def test_unrolled_vs_baseline_same_tolerance():
    """Unrolled + cleanup must reach the same tolerance as baseline PSOR."""
    matrix, rhs, obstacle = _small_lcp()
    tolerance = 1e-11

    # Baseline: projected previous + PSOR
    baseline = projected_sor_cleanup(
        matrix, rhs, obstacle,
        omega=1.2,
        residual_tolerance=tolerance,
        max_iter=10000,
        initial_guess=obstacle,
    )
    assert baseline.converged

    # Unrolled + PSOR cleanup
    x0 = obstacle.copy()
    omegas = np.array([1.0, 1.0, 1.0])
    unrolled = unrolled_projected_jacobi(matrix, rhs, obstacle, omegas, x0)
    unrolled_clean = projected_sor_cleanup(
        matrix, rhs, obstacle,
        omega=1.2,
        residual_tolerance=tolerance,
        max_iter=10000,
        initial_guess=unrolled,
    )
    assert unrolled_clean.converged

    assert natural_residual(matrix, rhs, obstacle, unrolled_clean.solution) <= tolerance
    assert np.allclose(baseline.solution, unrolled_clean.solution, atol=5e-11, rtol=0.0)


def test_extract_lcp_training_data():
    """Extracted training data should have correct shapes and finite values."""
    config = SolverConfig(
        option_type="put",
        m=10,
        n=4,
        omega=1.2,
        tol=1e-9,
        max_iter=5000,
    )
    result = solve_cn_lcp_surface(
        config,
        residual_tolerance=config.tol,
        omega=config.omega,
        max_iter=config.max_iter,
    )

    train_data = extract_lcp_training_data(result)

    assert len(train_data) == config.n  # one per time step
    for A, b, phi, V_prev, V_true in train_data:
        d = config.m - 1
        assert A.shape == (d, d)
        assert b.shape == (d,)
        assert phi.shape == (d,)
        assert V_prev.shape == (d,)
        assert V_true.shape == (d,)
        assert np.all(np.isfinite(A))
        assert np.all(np.isfinite(b))
        assert np.all(np.isfinite(phi))
        assert np.all(np.isfinite(V_prev))
        assert np.all(np.isfinite(V_true))


def test_unrolled_with_surface_solve_matches_mlp_path_api():
    """Both unrolled_omegas and warm_start_predictor paths produce valid results."""
    config = SolverConfig(
        option_type="put",
        m=10,
        n=4,
        omega=1.2,
        tol=1e-9,
        max_iter=5000,
    )

    # Unrolled path
    omegas = np.ones(3)
    unrolled_result = solve_cn_lcp_surface(
        config,
        residual_tolerance=config.tol,
        omega=config.omega,
        max_iter=config.max_iter,
        unrolled_omegas=omegas,
    )

    # MLP warm-start path (with dummy predictor)
    mlp_result = solve_cn_lcp_surface(
        config,
        residual_tolerance=config.tol,
        omega=config.omega,
        max_iter=config.max_iter,
        warm_start_predictor=lambda features: np.zeros(len(features)),
        blend=1.0,
    )

    # Baseline path (no warm-start)
    baseline_result = solve_cn_lcp_surface(
        config,
        residual_tolerance=config.tol,
        omega=config.omega,
        max_iter=config.max_iter,
    )

    # All should produce valid surfaces
    for result in (unrolled_result, mlp_result, baseline_result):
        assert max(result.natural_residuals) <= config.tol
        assert np.all(result.values >= result.payoff[None, :] - 1e-12)


def _psor_solve(matrix, rhs, obstacle):
    """Small helper to get a tight-tolerance PSOR solution for testing."""
    from solvers.cn_psor import psor_lcp
    return psor_lcp(
        matrix, rhs, obstacle,
        omega=1.2, tol=1e-12, max_iter=10000,
        x0=obstacle,
    )
