import numpy as np

from models.algorithm_unrolling import (
    build_feasible_warm_start,
    natural_residual,
    project_feasible,
    projected_sor_cleanup,
    solve_cn_lcp_surface,
)
from solvers.cn_psor import SolverConfig


def _small_lcp():
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


def test_projection_and_learned_initial_guess_are_hard_feasible():
    _, _, obstacle = _small_lcp()
    projected = project_feasible(np.array([-1.0, 0.5, 0.1]), obstacle)
    assert np.array_equal(projected, np.array([0.2, 0.5, 0.2]))

    warm = build_feasible_warm_start(
        lambda features: np.array([-10.0, 0.25, -10.0]),
        np.zeros((3, 2)),
        previous=np.array([0.3, 0.4, 0.3]),
        obstacle=obstacle,
        strike=1.0,
        blend=1.0,
    )

    assert np.all(warm.initial_guess >= obstacle)
    assert not warm.fallback_used
    assert warm.inference_seconds >= 0.0
    assert warm.projection_seconds >= 0.0


def test_nonfinite_prediction_falls_back_to_projected_previous_solution():
    _, _, obstacle = _small_lcp()
    previous = np.array([0.3, 0.45, 0.3])
    warm = build_feasible_warm_start(
        lambda features: np.array([0.1, np.nan, 0.1]),
        np.zeros((3, 2)),
        previous=previous,
        obstacle=obstacle,
        strike=1.0,
        blend=1.0,
    )

    assert warm.fallback_used
    assert warm.fallback_reason == "nonfinite_prediction"
    assert np.array_equal(warm.initial_guess, project_feasible(previous, obstacle))
    assert warm.fallback_seconds >= 0.0


def test_baseline_and_learned_cleanup_use_identical_natural_residual_tolerance():
    matrix, rhs, obstacle = _small_lcp()
    tolerance = 1.0e-11
    baseline = projected_sor_cleanup(
        matrix,
        rhs,
        obstacle,
        omega=1.2,
        residual_tolerance=tolerance,
        max_iter=10_000,
        initial_guess=obstacle,
    )
    warm = projected_sor_cleanup(
        matrix,
        rhs,
        obstacle,
        omega=1.2,
        residual_tolerance=tolerance,
        max_iter=10_000,
        initial_guess=np.array([0.25, 0.5, 0.25]),
    )

    assert baseline.converged and warm.converged
    assert baseline.residual_tolerance == warm.residual_tolerance == tolerance
    assert natural_residual(matrix, rhs, obstacle, baseline.solution) <= tolerance
    assert natural_residual(matrix, rhs, obstacle, warm.solution) <= tolerance
    assert np.allclose(baseline.solution, warm.solution, atol=5.0e-11, rtol=0.0)


def test_surface_result_counts_every_cleanup_iteration_and_time_component():
    config = SolverConfig(
        option_type="put",
        m=10,
        n=4,
        omega=1.2,
        tol=1.0e-9,
        max_iter=5_000,
    )
    result = solve_cn_lcp_surface(
        config,
        residual_tolerance=config.tol,
        omega=config.omega,
        max_iter=config.max_iter,
        warm_start_predictor=lambda features: np.zeros(len(features)),
        blend=1.0,
    )
    baseline = solve_cn_lcp_surface(
        config,
        residual_tolerance=config.tol,
        omega=config.omega,
        max_iter=config.max_iter,
    )

    assert result.residual_tolerance == config.tol
    assert result.cleanup_iterations_total == sum(result.cleanup_iterations)
    assert result.cleanup_iterations_total > 0
    assert max(result.natural_residuals) <= config.tol
    assert np.all(result.values >= result.payoff[None, :] - 1.0e-12)
    assert result.timings_seconds["cleanup"] > 0.0
    assert result.timings_seconds["feature_construction"] > 0.0
    assert baseline.timings_seconds["feature_construction"] == 0.0
    assert result.timings_seconds["total"] >= sum(
        result.timings_seconds[name]
        for name in (
            "assembly",
            "feature_construction",
            "inference",
            "projection",
            "fallback",
            "cleanup",
        )
    )
