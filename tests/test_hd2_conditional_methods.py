from __future__ import annotations

import json
from pathlib import Path

from hd2_conditional_methods import (
    build_conditional_plan,
    resolved_objective_for_candidate,
    select_with_conditionals,
)

ROOT = Path(__file__).resolve().parents[1]


def _cfg() -> dict:
    return json.loads((ROOT / "configs" / "hd2_hvi_objective_stabilization.json").read_text())


def _metric(d: int, candidate: str, price: float, gamma: float, boundary: float, pde: float, fb: float):
    return {
        "dimension": d,
        "candidate": candidate,
        "price_rmse": price,
        "gamma_rmse": gamma,
        "boundary_rmse": boundary,
        "pde_p95": pde,
        "fb_p95": fb,
        "integrity": True,
    }


def test_conditional_plan_runs_only_mechanistically_triggered_methods():
    cfg = _cfg()
    mandatory_no_candidate = {"selection_status": "no_stabilized_candidate"}
    plan = build_conditional_plan(
        cfg,
        mandatory_decision=mandatory_no_candidate,
        conflict_trigger=True,
        eight_probe_trigger=False,
        relobralo_budget_safe=True,
    )
    assert plan == ["HVI-CAGrad", "HVI-ReLoBRaLo"]
    plan = build_conditional_plan(
        cfg,
        mandatory_decision={"selection_status": "stabilized_candidate_selected"},
        conflict_trigger=False,
        eight_probe_trigger=True,
        relobralo_budget_safe=True,
    )
    assert plan == ["HVI-HD1-anchor-8probe"]
    assert build_conditional_plan(
        cfg,
        mandatory_decision={"selection_status": "stabilized_candidate_selected"},
        conflict_trigger=False,
        eight_probe_trigger=False,
        relobralo_budget_safe=True,
    ) == []


def test_conditional_candidates_resolve_exact_hd3_objective_metadata():
    cfg = _cfg()
    cagrad = resolved_objective_for_candidate("HVI-CAGrad", cfg)
    assert cagrad == {
        "candidate": "HVI-CAGrad",
        "warm_fraction": 0.2,
        "final_physics_weight": 0.1,
        "fallback_objective": False,
        "optimizer_mode": "cagrad",
        "cagrad_alpha": 0.5,
        "cagrad_rescale": 1,
    }
    probe = resolved_objective_for_candidate("HVI-HD1-anchor-8probe", cfg)
    assert probe["optimizer_mode"] == "fixed"
    assert probe["hutchinson_probe_count"] == 8
    relobralo = resolved_objective_for_candidate("HVI-ReLoBRaLo", cfg)
    assert relobralo["optimizer_mode"] == "relobralo"
    assert relobralo["relobralo_temperature"] == 1.0
    assert relobralo["relobralo_alpha"] == 0.999
    assert relobralo["relobralo_rho"] == 1.0


def test_eight_probe_candidate_can_use_exact_d2_anchor_alias_and_d5_new_run():
    cfg = _cfg()
    rows = [
        _metric(2, "SUP-HD2", 1.0, 1.0, 1.0, 1.0, 1.0),
        _metric(5, "SUP-HD2", 1.0, 1.0, 1.0, 1.0, 1.0),
        _metric(2, "HVI-HD1-anchor", 1.01, 1.02, 1.01, 0.70, 0.70),
        _metric(5, "HVI-HD1-anchor-8probe", 1.02, 1.03, 1.02, 0.60, 0.60),
    ]
    decision = select_with_conditionals(rows, cfg)
    assert decision["selection_status"] == "stabilized_candidate_selected"
    assert decision["resolved_objective"]["candidate"] == "HVI-HD1-anchor-8probe"
    assert decision["candidate_audit"]["HVI-HD1-anchor-8probe"]["dimensions"]["2"]["reused_anchor_d2"] is True


def test_final_selection_prefers_fixed_schedule_on_equal_physics_score():
    cfg = _cfg()
    rows = [
        _metric(2, "SUP-HD2", 1.0, 1.0, 1.0, 1.0, 1.0),
        _metric(5, "SUP-HD2", 1.0, 1.0, 1.0, 1.0, 1.0),
        _metric(2, "HVI-warm-low", 1.01, 1.01, 1.01, 0.60, 0.60),
        _metric(5, "HVI-warm-low", 1.01, 1.01, 1.01, 0.60, 0.60),
        _metric(2, "HVI-CAGrad", 1.01, 1.01, 1.01, 0.60, 0.60),
        _metric(5, "HVI-CAGrad", 1.01, 1.01, 1.01, 0.60, 0.60),
    ]
    decision = select_with_conditionals(rows, cfg)
    assert decision["resolved_objective"]["candidate"] == "HVI-warm-low"


def test_final_selection_falls_back_exactly_when_nothing_passes():
    cfg = _cfg()
    rows = [
        _metric(2, "SUP-HD2", 1.0, 1.0, 1.0, 1.0, 1.0),
        _metric(5, "SUP-HD2", 1.0, 1.0, 1.0, 1.0, 1.0),
        _metric(2, "HVI-CAGrad", 1.2, 0.2, 0.2, 0.1, 0.1),
        _metric(5, "HVI-CAGrad", 1.2, 0.2, 0.2, 0.1, 0.1),
    ]
    decision = select_with_conditionals(rows, cfg)
    assert decision["selection_status"] == "no_stabilized_candidate"
    assert decision["resolved_objective"] == {
        "candidate": "HVI-warm-low",
        "warm_fraction": 0.5,
        "final_physics_weight": 0.01,
        "fallback_objective": True,
        "optimizer_mode": "fixed",
        "hutchinson_probe_count": 2,
    }
