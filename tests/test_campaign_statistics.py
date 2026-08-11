from __future__ import annotations

import pytest

from campaign.statistics import (
    MetricGate,
    ScenarioObservation,
    decide_promotion,
    holm_adjust,
    measure_timing,
    summarize_paired_scenarios,
)


def test_paired_summary_aggregates_grid_points_within_each_scenario() -> None:
    """This fails if pairing or resampling happens at grid-point level."""
    baseline = [
        ScenarioObservation("put|r=0.01|seed=101", [1.0]),
        ScenarioObservation("put|r=0.03|seed=101", [5.0, 7.0, 9.0]),
    ]
    candidate = [
        ScenarioObservation("put|r=0.01|seed=101", [3.0]),
        ScenarioObservation("put|r=0.03|seed=101", [6.0, 8.0, 10.0]),
    ]

    summary = summarize_paired_scenarios(
        baseline,
        candidate,
        lower_is_better=True,
        bootstrap_iterations=200,
        random_seed=17,
    )

    assert summary.scenario_ids == (
        "put|r=0.01|seed=101",
        "put|r=0.03|seed=101",
    )
    assert summary.scenario_count == 2
    assert summary.mean_difference == pytest.approx(1.5)
    assert summary.sample_standard_deviation == pytest.approx(2**-0.5)
    assert summary.relative_percent_change == pytest.approx(-37.5)


def test_paired_summary_is_deterministic_and_uses_explicit_direction() -> None:
    baseline = {"put|seed=101": [4.0], "put|seed=203": [8.0]}
    candidate = {"put|seed=203": [6.0], "put|seed=101": [2.0]}

    lower = summarize_paired_scenarios(
        baseline, candidate, lower_is_better=True, bootstrap_iterations=100, random_seed=3
    )
    repeat = summarize_paired_scenarios(
        baseline, candidate, lower_is_better=True, bootstrap_iterations=100, random_seed=3
    )
    higher = summarize_paired_scenarios(
        baseline, candidate, lower_is_better=False, bootstrap_iterations=100, random_seed=3
    )

    assert lower.confidence_interval == repeat.confidence_interval
    assert lower.confidence_interval == pytest.approx((-2.0, -2.0))
    assert lower.relative_percent_change == pytest.approx(33.3333333333)
    assert higher.relative_percent_change == pytest.approx(-33.3333333333)


@pytest.mark.parametrize(
    ("baseline", "candidate", "message"),
    [
        (
            [
                ScenarioObservation("put|seed=101", [1.0]),
                ScenarioObservation("put|seed=101", [2.0]),
            ],
            [ScenarioObservation("put|seed=101", [1.0])],
            "duplicate baseline scenario ID",
        ),
        (
            [ScenarioObservation("put|seed=101", [1.0])],
            [ScenarioObservation("put|seed=203", [1.0])],
            "identical scenario IDs",
        ),
        (
            [ScenarioObservation("put|seed=101", [float("nan")])],
            [ScenarioObservation("put|seed=101", [1.0])],
            "finite values",
        ),
        (
            [ScenarioObservation("put|seed=101", [1.0])],
            [ScenarioObservation("put|seed=101", [float("inf")])],
            "finite values",
        ),
    ],
)
def test_paired_summary_rejects_invalid_scenario_inputs(
    baseline: list[ScenarioObservation], candidate: list[ScenarioObservation], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        summarize_paired_scenarios(baseline, candidate, lower_is_better=True)


def test_paired_summary_rejects_zero_baseline_for_relative_change() -> None:
    with pytest.raises(ValueError, match="zero baseline"):
        summarize_paired_scenarios(
            {"put|seed=101": [0.0], "put|seed=203": [0.0]},
            {"put|seed=101": [1.0], "put|seed=203": [2.0]},
            lower_is_better=True,
        )


def test_paired_summary_rejects_a_single_scenario_with_undefined_sample_spread() -> None:
    with pytest.raises(ValueError, match="at least two scenarios"):
        summarize_paired_scenarios(
            {"put|seed=101": [1.0]},
            {"put|seed=101": [2.0]},
            lower_is_better=True,
        )


def test_paired_summary_requires_an_explicit_boolean_direction() -> None:
    with pytest.raises(ValueError, match="lower_is_better must be a boolean"):
        summarize_paired_scenarios(
            {"put|seed=101": [1.0], "put|seed=203": [2.0]},
            {"put|seed=101": [2.0], "put|seed=203": [3.0]},
            lower_is_better=float("nan"),
        )


def test_holm_adjustment_preserves_order_and_uses_step_down_rejections() -> None:
    result = holm_adjust([0.04, 0.001, 0.03])

    assert result.adjusted_p_values == pytest.approx((0.06, 0.003, 0.06))
    assert result.reject == (False, True, False)


def test_holm_adjustment_handles_ties_and_one_hypothesis() -> None:
    ties = holm_adjust([0.03, 0.03])
    one = holm_adjust([0.03])

    assert ties.adjusted_p_values == pytest.approx((0.06, 0.06))
    assert ties.reject == (False, False)
    assert one.adjusted_p_values == pytest.approx((0.03,))
    assert one.reject == (True,)


def test_holm_adjustment_rejects_non_finite_p_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        holm_adjust([0.01, float("nan")])


def test_promotion_requires_practical_primary_improvement_and_all_metric_gates() -> None:
    decision = decide_promotion(
        [
            MetricGate(
                "price_rmse",
                relative_percent_change=3.0,
                confidence_interval=(-0.4, -0.1),
                lower_is_better=True,
                is_primary=True,
            ),
            MetricGate(
                "delta_rmse",
                relative_percent_change=-1.5,
                confidence_interval=(-0.1, 0.2),
                lower_is_better=True,
            ),
            MetricGate(
                "natural_residual",
                relative_percent_change=-4.5,
                confidence_interval=(-0.3, 0.4),
                lower_is_better=True,
                is_residual=True,
            ),
        ]
    )

    assert decision.status == "promote"
    assert decision.failed_metrics == ()
    assert decision.improved_primary_metrics == ("price_rmse",)


def test_promotion_expands_seed_count_when_primary_interval_is_inconclusive() -> None:
    decision = decide_promotion(
        [
            MetricGate(
                "price_rmse",
                relative_percent_change=3.0,
                confidence_interval=(-0.4, 0.1),
                lower_is_better=True,
                is_primary=True,
            )
        ]
    )

    assert decision.status == "inconclusive_expand_to_ten_seeds"
    assert decision.failed_metrics == ()


@pytest.mark.parametrize(
    "metric",
    [
        MetricGate(
            "price_rmse",
            relative_percent_change=-2.01,
            confidence_interval=(-0.1, 0.1),
            lower_is_better=True,
            is_primary=True,
        ),
        MetricGate(
            "natural_residual",
            relative_percent_change=-5.01,
            confidence_interval=(-0.1, 0.1),
            lower_is_better=True,
            is_residual=True,
            is_primary=True,
        ),
    ],
)
def test_promotion_fails_metrics_beyond_their_regression_limit(metric: MetricGate) -> None:
    decision = decide_promotion([metric])

    assert decision.status == "failed_metric_gate"
    assert decision.failed_metrics == (metric.name,)


def test_promotion_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        decide_promotion(
            [
                MetricGate(
                    "price_rmse",
                    relative_percent_change=float("inf"),
                    confidence_interval=(-0.1, -0.01),
                    lower_is_better=True,
                    is_primary=True,
                )
            ]
        )


def test_timing_uses_five_warmups_and_thirty_measurements_by_default() -> None:
    calls = 0
    clock_values = iter(value for index in range(30) for value in (float(index), index + 0.125))

    def operation() -> None:
        nonlocal calls
        calls += 1

    result = measure_timing(operation, clock=lambda: next(clock_values))

    assert calls == 35
    assert result.samples == pytest.approx((0.125,) * 30)
    assert result.median_seconds == pytest.approx(0.125)
    assert result.median_absolute_deviation_seconds == pytest.approx(0.0)
    assert result.p95_seconds == pytest.approx(0.125)


def test_timing_rejects_non_finite_clock_samples() -> None:
    clock_values = iter((0.0, float("nan")))

    with pytest.raises(ValueError, match="finite"):
        measure_timing(lambda: None, warmups=0, repetitions=1, clock=lambda: next(clock_values))
