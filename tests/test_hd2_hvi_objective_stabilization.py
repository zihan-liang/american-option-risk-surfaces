from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from hd_pipeline.gradient_diagnostics import (
    layerwise_gradient_diagnostics,
    objective_gradient_diagnostics,
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
