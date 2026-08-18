from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from hd3_classical_vs_structured_high_dimensional_pinn import (
    freeze_objective,
    resolve_objective,
)
from models.hd3_high_dimensional_pinn import FCPINN
from models.high_dimensional_pinn import initialize_model as initialize_deepsets
from physics.hd3_high_dimensional_sampling import (
    permutation_consistency_rmse,
    sample_transverse_states,
    transverse_invariance_rmse,
)

ROOT = Path(__file__).resolve().parents[1]


def _hd3_config() -> dict:
    return json.loads(
        (ROOT / "configs" / "hd3_classical_vs_structured_high_dimensional_pinn.json").read_text()
    )


def test_hd3_frozen_config_contract():
    cfg = _hd3_config()
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
    cfg = _hd3_config()
    assert cfg["fc_hidden_layers"] == 6
    assert cfg["fc_width"] == 128
    assert cfg["transverse_scale"] == 0.35
    assert cfg["hutchinson_probe_count"] == 2
    assert cfg["pipeline_wall_clock_hours"] == 18.0


def test_hd3_objective_fallback_is_exact_when_hd2_has_no_candidate(tmp_path: Path):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"selection_status": "no_stabilized_candidate"}), encoding="utf-8")
    got = resolve_objective(selection, _hd3_config())
    assert got == {
        "candidate": "HVI-warm-low",
        "warm_fraction": 0.5,
        "final_physics_weight": 0.01,
        "fallback_objective": True,
    }


def test_hd3_objective_uses_selected_hd2_candidate(tmp_path: Path):
    selected = {
        "candidate": "HVI-low",
        "warm_fraction": 0.2,
        "final_physics_weight": 0.01,
        "fallback_objective": False,
    }
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "selection_status": "stabilized_candidate_selected",
                "resolved_objective": selected,
            }
        ),
        encoding="utf-8",
    )
    assert resolve_objective(selection, _hd3_config()) == selected


def test_frozen_objective_cannot_change_after_canonical_started(tmp_path: Path):
    path = tmp_path / "resolved_objective.json"
    first = {
        "candidate": "HVI-low",
        "warm_fraction": 0.2,
        "final_physics_weight": 0.01,
        "fallback_objective": False,
    }
    first_hash = freeze_objective(path, first, canonical_started=False)
    assert freeze_objective(path, first, canonical_started=True) == first_hash
    with pytest.raises(RuntimeError, match="objective_immutable"):
        freeze_objective(
            path,
            {
                "candidate": "HVI-warm-low",
                "warm_fraction": 0.5,
                "final_physics_weight": 0.01,
                "fallback_objective": True,
            },
            canonical_started=True,
        )


def test_fc_model_accepts_all_declared_dimensions_and_hard_terminal_obstacle():
    for d in (10, 20, 50, 100):
        model = FCPINN(d).double()
        x = torch.zeros(3, d, dtype=torch.float64)
        tau = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        y = model(x, tau)
        assert y.shape == (3,)
        assert y[0].item() == pytest.approx(0.0, abs=0.0)
        payoff = torch.relu(1.0 - torch.exp(x.mean(dim=1)))
        assert torch.all(y >= payoff)
        assert model.hidden_layers == 6
        assert model.width == 128


def test_transverse_sampler_preserves_aggregate_log_geometric_mean():
    x, tau, m = sample_transverse_states(
        256,
        50,
        seed=901,
        m_bounds=(math.log(0.5), math.log(1.5)),
        tau_bounds=(0.0025, 1.0),
        transverse_scale=0.35,
        dtype=torch.float64,
    )
    assert x.shape == (256, 50)
    assert tau.shape == (256,)
    assert m.shape == (256,)
    assert torch.allclose(x.mean(dim=1), m, atol=1e-12, rtol=0)
    assert torch.allclose(torch.exp(x.mean(dim=1)), torch.exp(m), atol=1e-12, rtol=0)


def test_transverse_sampler_skip_reconstructs_sobol_stream_exactly():
    full_x, full_tau, full_m = sample_transverse_states(
        128,
        20,
        seed=902,
        m_bounds=(math.log(0.5), math.log(1.5)),
        tau_bounds=(0.0025, 1.0),
        transverse_scale=0.35,
        dtype=torch.float64,
    )
    x, tau, m = sample_transverse_states(
        64,
        20,
        seed=902,
        m_bounds=(math.log(0.5), math.log(1.5)),
        tau_bounds=(0.0025, 1.0),
        transverse_scale=0.35,
        skip=64,
        dtype=torch.float64,
    )
    assert torch.equal(x, full_x[64:])
    assert torch.equal(tau, full_tau[64:])
    assert torch.equal(m, full_m[64:])


def test_deepsets_permutation_consistency_is_machine_precision():
    model = initialize_deepsets(101, dtype=torch.float64)
    x, tau, _ = sample_transverse_states(
        64,
        10,
        seed=903,
        m_bounds=(math.log(0.5), math.log(1.5)),
        tau_bounds=(0.0025, 1.0),
        transverse_scale=0.35,
        dtype=torch.float64,
    )
    permutation = torch.tensor([3, 1, 8, 0, 9, 2, 7, 4, 6, 5])
    assert permutation_consistency_rmse(model, x, tau, permutation) < 1e-12


class _AggregateOracle(torch.nn.Module):
    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return torch.exp(x.mean(dim=-1)) + tau


def test_transverse_invariance_metric_is_zero_for_aggregate_oracle():
    groups = []
    taus = []
    for seed in (910, 911, 912, 913):
        x, tau, m = sample_transverse_states(
            6,
            20,
            seed=seed,
            m_bounds=(math.log(0.8), math.log(0.8) + 1e-14),
            tau_bounds=(0.4, 0.4 + 1e-14),
            transverse_scale=0.35,
            dtype=torch.float64,
        )
        # Remove the tiny Sobol variation so each repeat has exactly one aggregate state.
        x = x - x.mean(dim=1, keepdim=True) + torch.tensor(math.log(0.8), dtype=torch.float64)
        tau = torch.full_like(tau, 0.4)
        groups.append(x)
        taus.append(tau)
    x_grouped = torch.stack(groups, dim=0)
    tau_grouped = torch.stack(taus, dim=0)
    assert transverse_invariance_rmse(_AggregateOracle(), x_grouped, tau_grouped) < 1e-12
