from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from models.high_dimensional_pinn import initialize_model
from physics.high_dimensional_black_scholes import (
    black_scholes_vi_residual,
    covariance_matrix,
    effective_geometric_basket_parameters,
    full_hessian_contraction,
    geometric_basket_from_log_moneyness,
    hutchinson_hessian_contraction,
)
from hd1_windows_high_dimensional_pinn_48h import (
    REQUIRED_OUTPUTS,
    classify_decision,
    curriculum,
    validate_config,
)


def test_geometric_basket_reduction_formula():
    sg, qg = effective_geometric_basket_parameters(5, 0.25, 0.02, 0.30)
    assert sg**2 == pytest.approx(0.25**2 * (0.30 + 0.70 / 5))
    assert qg == pytest.approx(0.02 + 0.5 * (0.25**2 - sg**2))
    x = torch.log(torch.tensor([[0.5, 1.0, 1.5]], dtype=torch.float64))
    assert geometric_basket_from_log_moneyness(x).item() == pytest.approx((0.5 * 1.0 * 1.5) ** (1 / 3))


def test_deepsets_is_permutation_invariant_and_terminal_condition_is_hard():
    model = initialize_model(101, dtype=torch.float64)
    x = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float64)
    tau = torch.tensor([0.7], dtype=torch.float64)
    assert torch.allclose(model(x, tau), model(x[:, [2, 0, 1]], tau), atol=1e-12, rtol=0)
    terminal = model(x, torch.zeros_like(tau))
    payoff = torch.relu(1.0 - torch.exp(x.mean(dim=1)))
    assert torch.equal(terminal, payoff)


def test_operator_matches_quadratic_analytic_contraction():
    x = torch.tensor([[0.2, -0.1]], dtype=torch.float64, requires_grad=True)
    tau = torch.tensor([0.4], dtype=torch.float64, requires_grad=True)
    value = tau + (x * x).sum(dim=1) + 0.3 * x[:, 0] * x[:, 1]
    cov = covariance_matrix(2, 0.25, 0.30, dtype=torch.float64)
    got = full_hessian_contraction(value, x, cov)
    hessian = torch.tensor([[2.0, 0.3], [0.3, 2.0]], dtype=torch.float64)
    expected = torch.sum(cov * hessian)
    assert got.item() == pytest.approx(expected.item(), rel=1e-12, abs=1e-12)
    residual, _ = black_scholes_vi_residual(
        value, x, tau, r=0.05, q=0.02, sigma=0.25, rho=0.30, hessian_mode="full"
    )
    expected_residual = (
        1.0 - 0.5 * expected
        - (0.05 - 0.02 - 0.5 * 0.25**2) * (2.3 * x.detach().sum())
        + 0.05 * value.detach()
    )
    assert residual.item() == pytest.approx(expected_residual.item(), rel=1e-12)


def test_hutchinson_fixed_seed_reproducible_and_within_five_percent_at_d2():
    torch.manual_seed(7)
    x = torch.randn(256, 2, dtype=torch.float64, requires_grad=True)
    value = (x[:, 0] + 0.2 * x[:, 1]) ** 2 + 0.1 * x[:, 0] ** 2
    cov = covariance_matrix(2, 0.25, 0.30, dtype=torch.float64)
    exact = full_hessian_contraction(value, x, cov)
    est1 = hutchinson_hessian_contraction(value, x, cov, probe_count=2, probe_seed=101)
    est2 = hutchinson_hessian_contraction(value, x, cov, probe_count=2, probe_seed=101)
    assert torch.equal(est1, est2)
    rel = torch.linalg.vector_norm(est1 - exact) / torch.linalg.vector_norm(exact)
    assert rel.item() <= 0.05


def test_curriculum_is_frozen():
    assert curriculum(0, 10000) == pytest.approx((0.0, 1e-2))
    assert curriculum(1999, 10000) == pytest.approx((0.0, 1e-2))
    weight, epsilon = curriculum(5999, 10000)
    assert 0.049 < weight < 0.051
    assert 9e-4 < epsilon < 1.2e-3
    assert curriculum(9999, 10000) == pytest.approx((0.1, 1e-4))


def test_config_and_output_schema():
    cfg = json.loads((ROOT / "configs" / "hd1_windows_high_dimensional_pinn_48h.json").read_text())
    validate_config(cfg)
    assert cfg["dimensions"] == [1, 2, 5, 10]
    assert cfg["models"] == ["SUP", "HVI"]
    assert cfg["seed"] == 101 and cfg["adam_steps"] == 10000
    assert REQUIRED_OUTPUTS == {
        "environment_manifest.json", "config_manifest.json", "run_metrics.csv",
        "dimension_summary.csv", "training_history.csv", "hessian_validation.csv",
        "decision.json", "artifact_hashes.json",
    }


def test_no_promotion_beyond_pilot_promising():
    good = dict(
        finite_metrics=True, hashes_valid=True, obstacle_max=0.0,
        hvi_price_rmse=0.90, sup_price_rmse=1.0,
        hvi_pde_p95=0.90, sup_pde_p95=1.0,
        hvi_gamma_rmse=0.90, sup_gamma_rmse=1.0,
        hvi_boundary_rmse=0.96, sup_boundary_rmse=1.0,
        cuda_alloc_gib=6.0, hidden_failures=False,
    )
    decision = classify_decision(good)
    assert decision["classification"] == "pilot_promising"
    assert decision["candidate_supported"] is False
    assert classify_decision(dict(good, hvi_price_rmse=1.20))["classification"] == "diagnostic_only"
