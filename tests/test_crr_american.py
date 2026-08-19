from __future__ import annotations

import numpy as np
import pytest

from solvers.crr_american import (
    crr_american_put_curve,
    crr_american_put_price,
    crr_convergence_rows,
)


def test_crr_american_put_respects_intrinsic_and_basic_bounds():
    value = crr_american_put_price(
        spot=0.8,
        strike=1.0,
        maturity=1.0,
        rate=0.05,
        dividend_yield=0.02,
        volatility=0.25,
        steps=500,
    )
    assert np.isfinite(value)
    assert value >= 0.2 - 1e-12
    assert value <= 1.0


def test_crr_maturity_zero_is_intrinsic_value():
    assert crr_american_put_price(0.8, 1.0, 0.0, 0.05, 0.02, 0.25, 10) == pytest.approx(0.2)
    assert crr_american_put_price(1.2, 1.0, 0.0, 0.05, 0.02, 0.25, 10) == pytest.approx(0.0)


def test_crr_curve_matches_scalar_prices():
    spots = np.array([0.7, 1.0, 1.3])
    curve = crr_american_put_curve(
        spots,
        strike=1.0,
        maturity=1.0,
        rate=0.05,
        dividend_yield=0.02,
        volatility=0.25,
        steps=300,
    )
    expected = np.array([
        crr_american_put_price(
            spot=float(s),
            strike=1.0,
            maturity=1.0,
            rate=0.05,
            dividend_yield=0.02,
            volatility=0.25,
            steps=300,
        )
        for s in spots
    ])
    np.testing.assert_allclose(curve, expected, rtol=0.0, atol=1e-12)


def test_crr_convergence_rows_report_reference_and_previous_depth_errors():
    spots = np.array([0.8, 1.0, 1.2])
    reference = crr_american_put_curve(
        spots,
        strike=1.0,
        maturity=1.0,
        rate=0.05,
        dividend_yield=0.02,
        volatility=0.25,
        steps=1200,
    )
    rows = crr_convergence_rows(
        spots,
        reference,
        strike=1.0,
        maturity=1.0,
        rate=0.05,
        dividend_yield=0.02,
        volatility=0.25,
        depths=(100, 200, 400),
    )
    assert [row["depth"] for row in rows] == [100, 200, 400]
    assert rows[0]["previous_depth_rmse"] is None
    assert rows[1]["previous_depth_rmse"] is not None
    assert rows[-1]["cnpsor_rmse"] <= rows[0]["cnpsor_rmse"] * 2.0
    assert all("cnpsor_max_abs" in row for row in rows)


def test_crr_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="positive"):
        crr_american_put_price(0.0, 1.0, 1.0, 0.05, 0.02, 0.25, 100)
    with pytest.raises(ValueError, match="steps"):
        crr_american_put_price(1.0, 1.0, 1.0, 0.05, 0.02, 0.25, 0)
