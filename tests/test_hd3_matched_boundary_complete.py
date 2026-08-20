from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "hd3_matched_american_put_boundary_complete.py"
spec = importlib.util.spec_from_file_location("hd3_boundary_complete", MODULE_PATH)
assert spec and spec.loader
hd3f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hd3f)


def test_target_contract_and_scope_are_frozen():
    cfg = hd3f.load_config()
    assert cfg["experiment_id"] == "hd3_matched_american_put_boundary_complete"
    assert cfg["target_reduced_contract"] == {
        "option_type": "put",
        "K": 1.0,
        "T": 1.0,
        "r": 0.05,
        "q_g": 0.10,
        "sigma_g": 0.60,
    }
    assert cfg["rho"] == 0.80
    assert cfg["seed"] == 101
    assert cfg["canonical_dimensions"] == [10, 20, 50]
    assert 100 not in cfg["canonical_dimensions"]
    assert cfg["models"] == ["FC_HVI", "DEEPSETS_HVI"]
    assert cfg["single_seed"] is True
    assert cfg["diagnostic_only"] is True
    assert cfg["posthoc_matched_contract"] is True
    assert cfg["random_seed_robustness"] is False
    assert cfg["candidate_supported"] is False
    assert cfg["blind_holdout_access"] is False


def test_parameter_mapping_matches_same_reduced_contract():
    cfg = hd3f.load_config()
    expected = {
        10: (0.6625891564490792, 0.060487804878048806),
        20: (0.6666666666666666, 0.05777777777777779),
        50: (0.6691496051182058, 0.056119402985074646),
    }
    for d, (sigma, q) in expected.items():
        row = hd3f.parameter_mapping(d, cfg)
        assert math.isclose(row["asset_sigma"], sigma, rel_tol=0.0, abs_tol=1e-15)
        assert math.isclose(row["asset_q"], q, rel_tol=0.0, abs_tol=1e-15)
        assert math.isclose(row["effective_sigma_g"], 0.60, rel_tol=0.0, abs_tol=1e-12)
        assert math.isclose(row["effective_q_g"], 0.10, rel_tol=0.0, abs_tol=1e-12)


def test_final_aggregate_domain_is_boundary_complete():
    cfg = hd3f.load_config()
    lo, hi = hd3f.aggregate_bounds(cfg)
    assert math.isclose(math.exp(lo), 0.2, rel_tol=0.0, abs_tol=1e-14)
    assert math.isclose(math.exp(hi), 1.5, rel_tol=0.0, abs_tol=1e-14)
    assert tuple(cfg["tau"]) == (0.0025, 1.0)


def test_all_labeled_splits_and_physics_use_same_config_bounds():
    cfg = hd3f.load_config()
    lo, hi = hd3f.aggregate_bounds(cfg)
    for split in ("train", "validation", "test", "structural"):
        x, tau, g = hd3f.sample_states(split, n=256, d=10, cfg=cfg)
        m = x.mean(dim=1)
        assert float(m.min()) >= lo - 1e-12
        assert float(m.max()) <= hi + 1e-12
        assert float(g.min()) >= 0.2 - 1e-12
        assert float(g.max()) <= 1.5 + 1e-12
        assert float(tau.min()) >= cfg["tau"][0] - 1e-12
        assert float(tau.max()) <= cfg["tau"][1] + 1e-12

    unit = torch.zeros((2, 12), dtype=torch.float64)
    unit[1, :] = 1.0
    x, tau = hd3f.physics_unit_to_states(unit, d=10, cfg=cfg)
    means = x.detach().mean(dim=1)
    assert torch.allclose(means, torch.tensor([lo, hi], dtype=means.dtype), atol=1e-12, rtol=0.0)
    assert torch.allclose(tau.detach(), torch.tensor(cfg["tau"], dtype=tau.dtype), atol=1e-12, rtol=0.0)


def test_oracle_refinement_grid_covers_full_final_domain():
    cfg = hd3f.load_config()
    g, tau = hd3f.oracle_refinement_axes(cfg)
    assert math.isclose(float(g[0]), 0.2, rel_tol=0.0, abs_tol=1e-14)
    assert math.isclose(float(g[-1]), 1.5, rel_tol=0.0, abs_tol=1e-14)
    assert math.isclose(float(tau[0]), 0.0025, rel_tol=0.0, abs_tol=1e-14)
    assert math.isclose(float(tau[-1]), 1.0, rel_tol=0.0, abs_tol=1e-14)


def _synthetic_oracle():
    spots = np.linspace(0.0, 4.0, 9)
    taus = np.linspace(0.0, 1.0, 6)
    values = np.zeros((len(taus), len(spots)))
    return hd3f.Oracle(10, 0.60, 0.10, spots, taus, values, values, values, np.full(len(taus), 0.4), 0.0, 0.0)


def test_oracle_coverage_assertion_blocks_extrapolation():
    oracle = _synthetic_oracle()
    hd3f.assert_oracle_coverage(oracle, np.array([0.2, 1.5]), np.array([0.0025, 1.0]))
    with pytest.raises(RuntimeError, match="oracle_extrapolation_blocked"):
        hd3f.assert_oracle_coverage(oracle, np.array([4.1]), np.array([0.5]))
    with pytest.raises(RuntimeError, match="oracle_extrapolation_blocked"):
        hd3f.assert_oracle_coverage(oracle, np.array([0.5]), np.array([1.01]))


def test_boundary_grid_and_tolerances_are_frozen():
    cfg = hd3f.load_config()
    spots, taus = hd3f.boundary_axes(cfg)
    assert math.isclose(float(spots[0]), 0.2, rel_tol=0.0, abs_tol=1e-14)
    assert math.isclose(float(spots[-1]), 1.2, rel_tol=0.0, abs_tol=1e-14)
    assert len(spots) == 801
    assert math.isclose(float(taus[0]), 0.01, rel_tol=0.0, abs_tol=1e-14)
    assert math.isclose(float(taus[-1]), 1.0, rel_tol=0.0, abs_tol=1e-14)
    assert len(taus) == 201
    assert cfg["boundary"]["primary_tolerance"] == 1e-6
    assert cfg["boundary"]["sensitivity_tolerances"] == [1e-5, 1e-4]


def test_transition_boundary_detects_and_interpolates_clean_put_transition():
    spots = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
    premium = np.array([0.0, 0.0, 0.5e-6, 2.0e-6, 4.0e-6])
    result = hd3f.transition_boundary(spots, premium, tolerance=1e-6, min_continuation_nodes=2)
    assert result["found"] is True
    assert result["reason"] == "clean_transition"
    assert math.isclose(result["S_star_over_K"], 0.43333333333333335, rel_tol=0.0, abs_tol=1e-12)


def test_transition_boundary_ignores_deep_otm_near_expiry_recontact_after_confirmed_transition():
    spots = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.1, 1.2])
    premium = np.array([0.0, 0.0, 0.5e-6, 2.0e-6, 4.0e-6, 5.0e-5, 1.2e-2, 2.0e-4, 2.0e-8])
    result = hd3f.transition_boundary(spots, premium, tolerance=1e-6, min_continuation_nodes=2)
    assert result["found"] is True
    assert result["reason"] == "clean_transition"
    assert math.isclose(result["S_star_over_K"], 0.43333333333333335, rel_tol=0.0, abs_tol=1e-12)


def test_transition_boundary_rejects_isolated_false_contact():
    spots = np.linspace(0.2, 0.7, 6)
    premium = np.array([2e-6, 3e-6, 0.0, 4e-6, 5e-6, 6e-6])
    result = hd3f.transition_boundary(spots, premium, tolerance=1e-6, min_continuation_nodes=2)
    assert result["found"] is False
    assert result["reason"] == "expected_exercise_side_absent"


@pytest.mark.parametrize(
    "premium,reason",
    [
        (np.array([2e-6, 3e-6, 4e-6]), "all_continuation_like"),
        (np.array([0.0, 0.0, 0.0]), "all_exercise_like"),
        (np.array([0.0]), "insufficient_nodes"),
        (np.array([0.0, 2e-6, 0.0, 3e-6]), "no_clean_transition"),
    ],
)
def test_transition_boundary_missing_reasons(premium, reason):
    spots = np.linspace(0.2, 0.2 + 0.1 * (len(premium) - 1), len(premium))
    result = hd3f.transition_boundary(spots, premium, tolerance=1e-6, min_continuation_nodes=2)
    assert result["found"] is False
    assert result["reason"] == reason
    assert result["S_star_over_K"] is None


def test_missing_boundaries_are_not_zero_filled_into_full_rmse():
    reference = [0.30, 0.35, 0.40]
    predicted = [0.31, None, 0.39]
    metrics = hd3f.compute_boundary_metrics(reference, predicted)
    assert metrics["reference_boundary_count"] == 3
    assert metrics["predicted_boundary_count"] == 2
    assert metrics["matched_boundary_count"] == 2
    assert metrics["boundary_missed_count"] == 1
    assert metrics["boundary_false_count"] == 0
    assert math.isclose(metrics["boundary_coverage"], 2.0 / 3.0)
    assert math.isclose(metrics["boundary_conditional_rmse"], 0.01, abs_tol=1e-12)
    assert metrics["boundary_full_rmse"] is None


class _AnalyticDiagonalModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        g = torch.exp(x.mean(dim=1))
        return g**3 + 2.0 * g + 0.0 * tau


def test_reduced_delta_and_gamma_match_analytic_common_path():
    model = _AnalyticDiagonalModel()
    spots = np.array([0.3, 0.7, 1.2], dtype=float)
    taus = np.array([0.1, 0.5, 1.0], dtype=float)
    values, delta, gamma = hd3f.diagonal_value_and_greeks(model, d=10, spots=spots, taus=taus, device=torch.device("cpu"))
    assert np.allclose(values, spots**3 + 2.0 * spots, rtol=1e-6, atol=1e-7)
    assert np.allclose(delta, 3.0 * spots**2 + 2.0, rtol=1e-5, atol=1e-6)
    assert np.allclose(gamma, 6.0 * spots, rtol=1e-5, atol=1e-6)


def test_surface_arrays_include_common_path_greeks_and_primary_exercise_flag():
    cfg = hd3f.load_config()
    model = _AnalyticDiagonalModel()
    spots = np.array([0.3, 0.7, 1.2], dtype=float)
    taus = np.array([0.1, 0.5, 1.0], dtype=float)
    payload = hd3f.model_surface_arrays(model, d=10, spots=spots, taus=taus, cfg=cfg, device=torch.device("cpu"))
    required = {
        "S_over_K", "tau_over_T", "V_over_K", "payoff_over_K", "premium_over_K",
        "reduced_delta", "reduced_gamma", "scaled_reduced_gamma", "exercise_like_primary",
    }
    assert required.issubset(payload)
    for key in required - {"exercise_like_primary"}:
        assert np.all(np.isfinite(payload[key]))
    assert payload["exercise_like_primary"].dtype == np.bool_


def test_symmetric_diagonal_geometric_mean_equals_common_spot():
    for d in (10, 20, 50):
        spots = np.asarray([0.2, 0.5, 1.0, 1.5])
        x = np.log(spots)[:, None].repeat(d, axis=1)
        assert np.allclose(np.exp(x.mean(axis=1)), spots, rtol=0.0, atol=1e-14)


def test_binomial_final_curve_uses_2048_and_boundary_near_checks():
    cfg = hd3f.load_config()
    assert hd3f.binomial_final_steps(cfg) == 2048
    checks = hd3f.binomial_check_spots(cfg)
    for value in (0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00, 1.20, 1.40):
        assert value in checks
    spots = np.asarray([0.25, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    values = np.asarray([
        hd3f.crr_american_put(float(s), 1.0, k=1.0, r=0.05, q=0.10, sigma=0.60, steps=256)
        for s in spots
    ])
    assert np.all(values >= np.maximum(1.0 - spots, 0.0) - 1e-12)
    assert np.all(np.diff(values) <= 1e-10)


def test_cpu_memory_telemetry_failure_remains_nonfatal(monkeypatch):
    def fail_memory_probe():
        raise OSError("GetProcessMemoryInfo failed")

    monkeypatch.setattr(hd3f, "peak_cpu_memory_bytes", fail_memory_probe)
    assert hd3f.safe_peak_cpu_memory_bytes() is None


def test_new_namespace_cannot_reuse_old_matched_runtime():
    assert hd3f.RESULT_DIR.name == "hd3_matched_american_put_boundary_complete"
    assert hd3f.OLD_RESULT_DIR.name == "hd3_matched_american_put_comparison"
    assert hd3f.RESULT_DIR != hd3f.OLD_RESULT_DIR
    path = hd3f._checkpoint_path(10, "FC_HVI")
    assert hd3f.RUNTIME_DIR in path.parents
    assert hd3f.OLD_RESULT_DIR not in path.parents
