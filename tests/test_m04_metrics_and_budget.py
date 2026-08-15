from __future__ import annotations

import pytest


def test_complete_regime_metrics_preserve_missing_counts_and_use_regime_means() -> None:
    from campaign.m04 import aggregate_boundary_metrics

    result = aggregate_boundary_metrics(
        predictions=[0.6, None, 0.8, 0.7],
        targets=[0.5, 0.6, 0.7, None],
        statuses=["defined", "no_continuation_region", "defined", "unavailable"],
    )

    assert result["defined_count"] == 2
    assert result["target_defined_count"] == 3
    assert result["missing_boundary_rate"] == pytest.approx(1 / 3)
    assert result["false_finite_boundary_rate"] == pytest.approx(1.0)
    assert result["boundary_rmse"] == pytest.approx(0.1)


def test_budget_blocks_before_model_or_timing_when_limit_is_exceeded() -> None:
    from campaign.m04 import M04Budget

    clock = [0.0]
    budget = M04Budget(limit_seconds=1.0, clock=lambda: clock[0])
    budget.record("oracle_generation", 0.75)
    assert budget.checkpoint("before_model") is None
    clock[0] = 1.05
    budget.record("preprocessing", 0.30)

    assert budget.checkpoint("before_timing") == "resource_blocked"
    assert budget.state == "resource_blocked"
    assert budget.category_seconds["oracle_generation"] == pytest.approx(0.75)


def test_budget_uses_elapsed_wall_time_including_inference_and_extraction() -> None:
    from campaign.m04 import M04Budget

    clock = [100.0]
    budget = M04Budget(limit_seconds=1.0, clock=lambda: clock[0])
    budget.record("preprocessing", 0.01)
    clock[0] = 101.01
    budget.record("inference", 0.40)
    budget.record("extraction", 0.40)

    assert budget.checkpoint("after_extraction") == "resource_blocked"
    assert budget.elapsed_seconds == pytest.approx(1.01)
    assert budget.category_seconds["inference"] == pytest.approx(0.40)
    assert budget.category_seconds["extraction"] == pytest.approx(0.40)


def test_put_boundary_monotonicity_allows_decrease_as_time_to_maturity_increases() -> None:
    from campaign.m04 import put_boundary_monotonicity

    assert put_boundary_monotonicity([1.0, 0.9, 0.8]) == 0
    assert put_boundary_monotonicity([0.8, 0.9, 0.7]) == 1


def test_monotonicity_diagnostics_do_not_join_across_a_missing_middle_level() -> None:
    from campaign.m04 import put_boundary_monotonicity_diagnostics

    result = put_boundary_monotonicity_diagnostics([0.8, None, 1.2])

    assert result == {
        "adjacent_transition_denominator": 0,
        "violation_count": 0,
        "violation_rate": None,
        "missing_transition_count": 2,
    }


def test_screen_decision_uses_complete_regime_pairs_and_holm_for_two_heads() -> None:
    from campaign.m04 import decide_m04_screen

    rows = []
    for scenario, baseline, scalar, implicit in (("a", 1.0, 0.8, 0.95), ("b", 1.1, 0.85, 1.0), ("c", 0.9, 0.7, 0.86)):
        for seed in (101, 203):
            for method, value in (("price_surrogate_boundary_extraction", baseline), ("scalar_boundary_head", scalar), ("implicit_level_set", implicit)):
                rows.append({"method": method, "seed": seed, "scenario_id": scenario, "boundary_rmse": value, "missing_boundary_rate": 0.0, "monotonicity_violation_rate": 0.0, "oracle_status": "defined"})

    decision = decide_m04_screen(rows, bootstrap_iterations=200, bootstrap_seed=7)

    assert set(decision["comparisons"]) == {"scalar_boundary_head", "implicit_level_set"}
    assert decision["comparisons"]["scalar_boundary_head"]["scenario_count"] == 3
    assert decision["comparisons"]["scalar_boundary_head"]["holm_adjusted_p_value"] is not None


def test_screen_decision_preserves_positive_improvement_for_a_better_candidate() -> None:
    from campaign.m04 import decide_m04_screen

    rows = []
    for scenario, baseline, candidate in (("a", 1.0, 0.5), ("b", 1.2, 0.6), ("c", 0.8, 0.4)):
        for method, value in (("price_surrogate_boundary_extraction", baseline), ("scalar_boundary_head", candidate)):
            rows.append({"method": method, "seed": 101, "scenario_id": scenario, "boundary_rmse": value, "missing_boundary_rate": 0.0, "monotonicity_violation_rate": 0.0, "oracle_status": "defined"})

    result = decide_m04_screen(rows, bootstrap_iterations=200, bootstrap_seed=3)

    assert result["comparisons"]["scalar_boundary_head"]["relative_percent_improvement"] > 2.0
