from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from experiments.reference_audits.crr50000_core import crr_american_put
from experiments.reference_audits.dirk_binomial_50k_convergence import (
    BENCHMARK_STEPS,
    CONTRACT,
    REGIME_ID,
    REPRODUCTION_MAX_ABS_TOL,
    RUIXI_EXPORT_COMMIT,
    RUIXI_SOURCE_COMMIT,
    SLICE_TAUS,
    SPATIAL_FIXED_N,
    SPATIAL_LEVELS,
    SPOT_POINTS,
    TEMPORAL_FIXED_M,
    TEMPORAL_LEVELS,
    common_spot_grid,
    compute_error_metrics,
    interpolate_surface_like_ruixi,
    observed_order,
    solve_dirk_price_slices,
)


def test_frozen_protocol_matches_existing_high_accuracy_reference() -> None:
    assert RUIXI_SOURCE_COMMIT == "5f63fa7b12f704c3577c72f9d7b7e5146e78c1ca"
    assert RUIXI_EXPORT_COMMIT == "63ce7cf6ffe04a778dc2f3b0ce691a26a1aca5c6"
    assert REGIME_ID == "put_T100_s060_r005_q010"
    assert CONTRACT == {
        "option_type": "put",
        "K": 1.0,
        "T": 1.0,
        "r": 0.05,
        "q": 0.10,
        "sigma": 0.60,
        "Smax": 4.0,
    }
    assert BENCHMARK_STEPS == 50_000
    assert REPRODUCTION_MAX_ABS_TOL == 1e-9


def test_refinement_sequences_change_only_one_axis() -> None:
    assert SPATIAL_FIXED_N == 960
    assert SPATIAL_LEVELS == (120, 240, 480, 960)
    assert TEMPORAL_FIXED_M == 480
    assert TEMPORAL_LEVELS == (60, 120, 240, 480, 960, 1920)
    assert (480, 960) in {(m, SPATIAL_FIXED_N) for m in SPATIAL_LEVELS}
    assert (480, 960) in {(TEMPORAL_FIXED_M, n) for n in TEMPORAL_LEVELS}


def test_common_price_grid_is_exactly_the_frozen_matched_grid() -> None:
    spots = common_spot_grid()
    assert SPOT_POINTS == 401
    assert len(spots) == 401
    assert spots[0] == pytest.approx(0.5)
    assert spots[-1] == pytest.approx(1.5)
    assert np.allclose(np.diff(spots), 0.0025, rtol=0.0, atol=2e-16)
    assert SLICE_TAUS == (1.0, 0.5, 0.1)


def test_error_metrics_include_all_predeclared_price_norms() -> None:
    benchmark = np.array([1.0, 2.0, 4.0])
    approximate = np.array([1.1, 1.8, 4.2])
    metrics = compute_error_metrics(approximate, benchmark)

    assert set(metrics) == {
        "bias",
        "mae",
        "rmse",
        "relative_l2",
        "linf",
        "p95_abs",
    }
    assert metrics["bias"] == pytest.approx(1.0 / 30.0)
    assert metrics["mae"] == pytest.approx(1.0 / 6.0)
    assert metrics["rmse"] == pytest.approx(math.sqrt(0.03))
    assert metrics["relative_l2"] == pytest.approx(0.3 / math.sqrt(21.0))
    assert metrics["linf"] == pytest.approx(0.2)
    assert metrics["p95_abs"] == pytest.approx(0.2)


def test_observed_order_uses_refinement_ratio() -> None:
    assert observed_order(4e-4, 1e-4) == pytest.approx(2.0)
    assert observed_order(8e-4, 2e-4, refinement_ratio=2.0) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        observed_order(0.0, 1e-4)
    with pytest.raises(ValueError):
        observed_order(1e-4, -1e-5)


def test_interpolation_matches_ruixi_space_then_time_linear_convention() -> None:
    source_m = np.array([0.5, 1.0, 1.5])
    source_tau = np.array([0.0, 0.5, 1.0])
    surface = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
        ]
    )
    query_m = np.array([0.75, 1.25])
    query_tau = np.array([0.25, 0.75])
    got = interpolate_surface_like_ruixi(
        source_m, source_tau, surface, query_m, query_tau
    )
    assert got == pytest.approx(np.array([1.0, 3.0]))


def test_crr_core_reproduces_uploaded_50000_step_anchor() -> None:
    # Frozen value from the uploaded binomial_benchmark_50000 archive and the
    # audited consolidated anchor table. This is a numerical benchmark, not an
    # analytic exact solution.
    value = crr_american_put(1.0, 0.1, steps=50_000)
    assert value == pytest.approx(0.07752300721771982, abs=5e-13, rel=0.0)


def test_ruixi_solver_smoke_uses_imported_original_solver() -> None:
    spots = np.linspace(0.8, 1.2, 5)
    values, metadata = solve_dirk_price_slices(
        M=24,
        N=24,
        query_spots=spots,
        query_taus=(0.1,),
    )
    assert values.shape == (1, 5)
    assert np.all(np.isfinite(values))
    assert metadata["method"] == "dirk_policy"
    assert metadata["time_grid"] == "quadratic"
    assert metadata["converged"] is True
    assert metadata["spot_grid"] == "sinh_strike_concentrated"
    assert metadata["lcp_tolerance"] == pytest.approx(1e-12)
    assert metadata["obstacle_tolerance"] == pytest.approx(1e-12)
