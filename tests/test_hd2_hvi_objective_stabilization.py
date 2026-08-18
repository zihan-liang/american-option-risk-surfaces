from __future__ import annotations

import json
from pathlib import Path

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
