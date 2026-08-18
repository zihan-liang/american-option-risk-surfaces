from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from hd_pipeline.gradient_diagnostics import (
    layerwise_gradient_diagnostics,
    objective_gradient_diagnostics,
)
from hd2_hvi_objective_stabilization import (
    curriculum,
    persistent_conflict,
    select_hd2_candidate,
)

ROOT = Path(__file__).resolve().parents[1]


def test_hd2_frozen_config_contract():
    cfg = json.loads((ROOT / "configs" / "hd2_hvi_objective_stabilization.json").read_text())
    assert cfg["seed"] == 101
    assert cfg["dimensions"] == [2, 5]
    assert cfg["candidates"] == [
        "SUP-HD2",
        "HVI-HD1-anchor",
        "HVI-low",
        "HVI-warm",
        "HVI-warm-low",
    ]
    assert cfg["total_steps"] == 10000
    assert cfg["checkpoint_interval"] == 250
    assert cfg["gradient_diagnostic_interval"] == 500
    assert cfg["hutchinson_probe_count"] == 2
    assert cfg["max_cuda_allocation_gib"] == 7.2


def test_hd2_candidate_schedule_is_frozen():
    cfg = json.loads((ROOT / "configs" / "hd2_hvi_objective_stabilization.json").read_text())
    assert cfg["candidate_settings"] == {
        "SUP-HD2": {"warm_start_steps": 10000, "final_physics_weight": 0.0},
        "HVI-HD1-anchor": {"warm_start_steps": 2000, "final_physics_weight": 0.1},
        "HVI-low": {"warm_start_steps": 2000, "final_physics_weight": 0.01},
        "HVI-warm": {"warm_start_steps": 5000, "final_physics_weight": 0.1},
        "HVI-warm-low": {"warm_start_steps": 5000, "final_physics_weight": 0.01},
    }


class _TwoParameterModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))


def test_gradient_diagnostics_detect_known_alignment_and_conflict():
    model = _TwoParameterModel()
    base = model.w[0] + 2.0 * model.w[1]
    sup = base.square()
    phys_aligned = 3.0 * base.square()
    aligned = objective_gradient_diagnostics(sup, phys_aligned, model, lambda_weight=0.1)
    assert aligned["cosine"] == pytest.approx(1.0, abs=1e-12)
    assert aligned["raw_norm_ratio"] == pytest.approx(3.0, rel=1e-12)
    assert aligned["weighted_norm_ratio"] == pytest.approx(0.3, rel=1e-12)
    phys_conflict = -0.5 * base.square()
    conflict = objective_gradient_diagnostics(sup, phys_conflict, model, lambda_weight=0.1)
    assert conflict["cosine"] == pytest.approx(-1.0, abs=1e-12)
    assert conflict["sup_dot_total"] > 0.0


def test_gradient_diagnostics_leave_graph_available_for_training_backward():
    torch.manual_seed(4)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 1),
    ).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(12, 2, dtype=torch.float64)
    for _ in range(3):
        pred = model(x).reshape(-1)
        sup = pred.square().mean()
        phys = (pred - 1.0).square().mean()
        row = objective_gradient_diagnostics(sup, phys, model, lambda_weight=0.1)
        assert all(torch.isfinite(torch.tensor(v)) for v in row.values())
        optimizer.zero_grad(set_to_none=True)
        (sup + 0.1 * phys).backward()
        optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_layerwise_gradient_diagnostics_are_named_and_detached():
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 4),
        torch.nn.Tanh(),
        torch.nn.Linear(4, 1),
    ).double()
    x = torch.ones(5, 2, dtype=torch.float64)
    pred = model(x).reshape(-1)
    sup = pred.square().mean()
    phys = (pred - 2.0).square().mean()
    groups = {
        "encoder": list(model[0].parameters()),
        "head": list(model[2].parameters()),
    }
    rows = layerwise_gradient_diagnostics(sup, phys, groups, lambda_weight=0.1)
    assert [row["group"] for row in rows] == ["encoder", "head"]
    assert all(isinstance(row["cosine"], float) for row in rows)


def test_curriculum_matches_frozen_warm_start_and_final_weight():
    weight, eps = curriculum(1999, 10000, 2000, 0.1)
    assert weight == 0.0
    assert eps == pytest.approx(1e-2)
    weight, eps = curriculum(9999, 10000, 2000, 0.1)
    assert weight == pytest.approx(0.1)
    assert eps == pytest.approx(1e-4)
    weight, eps = curriculum(4999, 10000, 5000, 0.01)
    assert weight == 0.0
    assert eps == pytest.approx(1e-2)
    weight, eps = curriculum(9999, 10000, 5000, 0.01)
    assert weight == pytest.approx(0.01)
    assert eps == pytest.approx(1e-4)


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


def test_selection_is_price_first_and_chooses_best_eligible_physics_candidate():
    rows = [
        _metric(2, "SUP-HD2", 1.0, 1.0, 1.0, 1.0, 1.0),
        _metric(5, "SUP-HD2", 1.0, 1.0, 1.0, 1.0, 1.0),
        _metric(2, "HVI-low", 1.02, 1.05, 1.03, 0.60, 0.70),
        _metric(5, "HVI-low", 1.03, 1.04, 1.04, 0.65, 0.70),
        _metric(2, "HVI-warm-low", 1.01, 1.02, 1.02, 0.50, 0.60),
        _metric(5, "HVI-warm-low", 1.02, 1.03, 1.01, 0.55, 0.60),
        # Huge physics gain cannot compensate for a failed price gate.
        _metric(2, "HVI-warm", 1.06, 0.2, 0.2, 0.1, 0.1),
        _metric(5, "HVI-warm", 1.06, 0.2, 0.2, 0.1, 0.1),
    ]
    decision = select_hd2_candidate(rows)
    assert decision["selection_status"] == "stabilized_candidate_selected"
    assert decision["resolved_objective"]["candidate"] == "HVI-warm-low"
    assert decision["resolved_objective"]["fallback_objective"] is False
    assert "HVI-warm" not in decision["eligible_candidates"]


def test_selection_uses_exact_fallback_when_no_candidate_passes():
    rows = [
        _metric(2, "SUP-HD2", 1.0, 1.0, 1.0, 1.0, 1.0),
        _metric(5, "SUP-HD2", 1.0, 1.0, 1.0, 1.0, 1.0),
        _metric(2, "HVI-low", 1.06, 0.5, 0.5, 0.1, 0.1),
        _metric(5, "HVI-low", 1.06, 0.5, 0.5, 0.1, 0.1),
    ]
    decision = select_hd2_candidate(rows)
    assert decision["selection_status"] == "no_stabilized_candidate"
    assert decision["resolved_objective"] == {
        "candidate": "HVI-warm-low",
        "warm_fraction": 0.5,
        "final_physics_weight": 0.01,
        "fallback_objective": True,
    }


def test_persistent_conflict_requires_negative_cosine_and_validation_stall():
    records = [
        {"step": 1000, "cosine": 0.2, "validation_price_rmse": 0.010},
        {"step": 1500, "cosine": 0.1, "validation_price_rmse": 0.008},
        {"step": 2000, "cosine": 0.0, "validation_price_rmse": 0.006},
        {"step": 2500, "cosine": -0.20, "validation_price_rmse": 0.0065},
        {"step": 3000, "cosine": -0.30, "validation_price_rmse": 0.0068},
    ]
    result = persistent_conflict(records, warm_start_steps=2000)
    assert result["triggered"] is True
    assert result["trigger_steps"] == [2500, 3000]
    aligned = [dict(row, cosine=0.1) for row in records]
    assert persistent_conflict(aligned, warm_start_steps=2000)["triggered"] is False
