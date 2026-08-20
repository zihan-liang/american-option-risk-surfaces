from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "hd3_matched_american_put_boundary_complete.py"
spec = importlib.util.spec_from_file_location("hd3_boundary_complete_invariants", MODULE_PATH)
assert spec and spec.loader
hd3f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hd3f)


class AnalyticGeometricModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        g = torch.exp(x.mean(dim=1))
        return g**3 + 2.0 * g + 0.0 * tau


def test_reduced_delta_equals_sum_of_component_spot_deltas_on_diagonal():
    d = 10
    model = AnalyticGeometricModel()
    spots = np.asarray([0.3, 0.7, 1.2], dtype=float)
    taus = np.asarray([0.1, 0.5, 1.0], dtype=float)

    x = torch.log(torch.tensor(spots, dtype=torch.float64)).reshape(-1, 1).repeat(1, d)
    x.requires_grad_(True)
    tau = torch.tensor(taus, dtype=torch.float64)
    value = model(x, tau)
    v_x = torch.autograd.grad(value.sum(), x)[0]
    summed_component_delta = (v_x / torch.exp(x)).sum(dim=1).detach().numpy()

    _, reduced_delta, _ = hd3f.diagonal_value_and_greeks(
        model,
        d=d,
        spots=spots,
        taus=taus,
        device=torch.device("cpu"),
    )
    assert np.allclose(reduced_delta, summed_component_delta, rtol=1e-5, atol=1e-6)


def test_actual_frozen_trainer_physics_transform_uses_final_config_bounds():
    cfg = hd3f.load_config()
    hd3f._install_training_namespace()
    unit = torch.zeros((2, 12), dtype=torch.float64)
    unit[1, :] = 1.0
    x, tau = hd3f._old._transform_physics_unit(unit, 10, cfg)
    lo, hi = hd3f.aggregate_bounds(cfg)
    assert torch.allclose(x.detach().mean(dim=1), torch.tensor([lo, hi], dtype=x.dtype), atol=1e-12, rtol=0.0)
    assert torch.allclose(tau.detach(), torch.tensor(cfg["tau"], dtype=tau.dtype), atol=1e-12, rtol=0.0)


def test_checkpoint_save_hook_adds_final_namespace_and_training_domain(monkeypatch):
    captured = {}

    def capture_save(path, payload):
        captured.update(payload)

    monkeypatch.setattr(hd3f._runtime, "save_checkpoint_atomic", capture_save)
    hd3f._install_training_namespace()
    hd3f._old.save_checkpoint_atomic(Path("dummy.pt"), {"dimension": 10})

    cfg = hd3f.load_config()
    assert captured["experiment_id"] == "hd3_matched_american_put_boundary_complete"
    assert captured["training_domain"] == hd3f.training_domain_metadata(cfg)
    assert captured["target_reduced_contract"] == cfg["target_reduced_contract"]
    assert captured["objective"] == cfg["objective"]
    assert captured["parameter_mapping"]["dimension"] == 10


def test_put_transition_ignores_low_premium_otm_tail_after_true_boundary():
    # Near expiry, a far-OTM American put can have value below the contact
    # tolerance even though that region is continuation, not exercise. The
    # free-boundary detector must use the economically relevant ITM side and
    # must not interpret the near-zero OTM tail as a second exercise region.
    spots = np.asarray([0.20, 0.30, 0.40, 0.50, 0.80, 0.99, 1.00, 1.10, 1.20])
    premium = np.asarray([0.0, 0.0, 0.5e-6, 2.0e-6, 3.0e-6, 4.0e-6, 2.0e-6, 0.2e-6, 0.0])
    result = hd3f.transition_boundary(spots, premium, tolerance=1e-6, min_continuation_nodes=2)
    assert result["found"] is True
    assert result["reason"] == "clean_transition"
    assert np.isclose(result["S_star_over_K"], 0.43333333333333335, atol=1e-12, rtol=0.0)
