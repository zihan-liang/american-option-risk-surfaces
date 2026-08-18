from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_hd3_frozen_config_contract():
    cfg = json.loads((ROOT / "configs" / "hd3_classical_vs_structured_high_dimensional_pinn.json").read_text())
    assert cfg["seed"] == 101
    assert cfg["canonical_dimensions"] == [10, 20, 50]
    assert cfg["stress_dimensions"] == [100]
    assert cfg["canonical_steps"] == 6000
    assert cfg["stress_steps"] == 4000
    assert cfg["models"] == ["FC_HVI", "DEEPSETS_HVI"]
    assert cfg["fallback_objective"] == {
        "candidate": "HVI-warm-low",
        "warm_fraction": 0.5,
        "final_physics_weight": 0.01,
    }


def test_hd3_architecture_and_budget_are_frozen():
    cfg = json.loads((ROOT / "configs" / "hd3_classical_vs_structured_high_dimensional_pinn.json").read_text())
    assert cfg["fc_hidden_layers"] == 6
    assert cfg["fc_width"] == 128
    assert cfg["transverse_scale"] == 0.35
    assert cfg["hutchinson_probe_count"] == 2
    assert cfg["pipeline_wall_clock_hours"] == 18.0
