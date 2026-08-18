from __future__ import annotations

from pathlib import Path

import pytest

from hd_pipeline_windows_final import (
    PHASE_ORDER,
    project_canonical_seconds_final,
    relobralo_budget_safe,
    stress_budget_decision,
)

ROOT = Path(__file__).resolve().parents[1]


def test_final_phase_order_places_parallel_benchmark_before_canonical_runs():
    assert PHASE_ORDER == [
        "preflight",
        "hd2_phase_a",
        "hd2_screen",
        "hd2_conditional",
        "hd2_select",
        "hd2_finalize",
        "hd3_preflight",
        "hd3_parallelism_benchmark",
        "hd3_10d",
        "hd3_20d",
        "hd3_50d",
        "hd3_100d_stress",
        "finalize",
    ]


def test_projection_uses_batch_matched_parallel_comparison():
    benchmark = {
        "smoke_steps": 200,
        "sequential": {
            "d10_FC_HVI": {"wall_seconds": 40.0},
            "d10_DEEPSETS_HVI": {"wall_seconds": 50.0},
            "d50_FC_HVI": {"wall_seconds": 100.0},
            "d50_DEEPSETS_HVI": {"wall_seconds": 120.0},
        },
        "comparison_sequential_d10": {
            "FC_HVI": {"wall_seconds": 30.0},
            "DEEPSETS_HVI": {"wall_seconds": 30.0},
        },
        "concurrent_d10": {"wall_seconds": 45.0},
    }
    seq = project_canonical_seconds_final(benchmark, canonical_steps=6000, parallel_enabled=False)
    par = project_canonical_seconds_final(benchmark, canonical_steps=6000, parallel_enabled=True)
    assert seq == pytest.approx((90.0 + 122.5 + 220.0) * 30.0)
    # Parallel speed ratio is 45/(30+30)=0.75, while the d10 parallel pair
    # itself uses its directly measured 45-second wall time.
    assert par == pytest.approx((45.0 + 122.5 * 0.75 + 220.0) * 30.0)
    assert par < seq


def test_relobralo_budget_reserves_hd3_canonical_and_finalization():
    assert relobralo_budget_safe(
        hours_remaining=14.0,
        projected_relobralo_seconds=3600.0,
        canonical_reserve_hours=10.0,
        finalization_reserve_hours=1.0,
    )
    assert not relobralo_budget_safe(
        hours_remaining=11.0,
        projected_relobralo_seconds=3600.0,
        canonical_reserve_hours=10.0,
        finalization_reserve_hours=1.0,
    )


def test_stress_budget_accounts_for_dimension_and_step_scaling():
    decision = stress_budget_decision(
        d50_pair_seconds=3600.0,
        canonical_steps=6000,
        stress_steps=4000,
        hours_remaining=3.0,
        finalization_reserve_hours=1.0,
    )
    assert decision["projected_stress_seconds"] == pytest.approx(4800.0)
    assert decision["run_stress"] is True
    blocked = stress_budget_decision(
        d50_pair_seconds=3600.0,
        canonical_steps=6000,
        stress_steps=4000,
        hours_remaining=2.0,
        finalization_reserve_hours=1.0,
    )
    assert blocked["run_stress"] is False


def test_final_windows_runner_is_one_command_resume_interface():
    runner = (ROOT / "scripts" / "run_hd2_hd3_windows.ps1").read_text(encoding="utf-8")
    assert "experiments\\hd_pipeline_windows_final.py" in runner
    assert ".\\.venv-pinn\\Scripts\\python.exe" in runner
    assert "$LASTEXITCODE" in runner
    for phase in ("all", "preflight", "hd2", "hd3", "finalize", "status"):
        assert phase in runner
    lowered = runner.lower()
    assert "pip install" not in lowered
    assert "conda install" not in lowered


def test_runtime_directories_are_explicitly_ignored():
    for result_dir in (
        "hd2_hvi_objective_stabilization",
        "hd3_classical_vs_structured_high_dimensional_pinn",
        "hd_pipeline",
    ):
        text = (ROOT / "results" / result_dir / ".gitignore").read_text(encoding="utf-8")
        assert "runtime/" in text
