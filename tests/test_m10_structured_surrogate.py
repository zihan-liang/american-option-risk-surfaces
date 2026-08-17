import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STANDALONE = REPO_ROOT / "experiments" / "standalone"
sys.path.insert(0, str(STANDALONE))

import m10_structured_surrogate_core as core  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "standalone" / "m10_structured_surrogate.json"


def test_config_freezes_packet_and_campaign_limits():
    config = json.loads(CONFIG.read_text())
    core.validate_config(config)
    assert config["training_seeds"] == [101, 203, 307, 409, 503]
    assert tuple(config["arms"]) == core.ARMS
    assert config["budget_seconds"] == 3600
    assert config["model"]["device"] == "cpu"
    assert config["model"]["dtype"] == "float64"
    assert config["model"]["optimizer"] == "adam"
    assert config["timing"] == {"warmups": 5, "repetitions": 30}


def test_train_only_representation_scales_are_positive_and_deterministic():
    american = np.array([0.20, 0.25, 0.30, 0.40])
    european = np.array([0.18, 0.20, 0.27, 0.35])
    first = core.fit_representation_scales(american, european)
    second = core.fit_representation_scales(american.copy(), european.copy())
    assert first == second
    assert first["price_scale"] > 0.0
    assert first["premium_scale"] > 0.0
    assert first["premium_scale"] < first["price_scale"]


def test_reconstruction_semantics_are_distinct_and_terminal_scaled_is_exact_at_maturity():
    raw = np.array([-1.0, 0.0, 1.0])
    euro = np.array([0.4, 0.4, 0.4])
    tau = np.array([0.0, 0.5, 1.0])
    scales = {"price_scale": 0.5, "premium_scale": 0.1}
    direct = core.reconstruct_numpy(raw, euro, tau, "direct_price", scales)
    positive = core.reconstruct_numpy(raw, euro, tau, "positive_premium", scales)
    terminal = core.reconstruct_numpy(raw, euro, tau, "terminal_scaled_premium", scales)
    anchor = core.reconstruct_numpy(raw, euro, tau, "european_anchor_correction", scales)
    assert np.allclose(direct, 0.5 * raw)
    assert np.all(positive >= euro)
    assert terminal[0] == pytest.approx(euro[0])
    assert terminal[2] > euro[2]
    assert anchor[0] < euro[0]
    assert not np.allclose(positive, anchor)


def test_torch_and_numpy_reconstruction_match():
    torch = pytest.importorskip("torch")
    raw = np.array([-0.3, 0.2, 0.7]); euro = np.array([0.1, 0.2, 0.3]); tau = np.array([0.0, 0.4, 1.0])
    scales = {"price_scale": 0.8, "premium_scale": 0.05}
    for arm in core.ARMS:
        expected = core.reconstruct_numpy(raw, euro, tau, arm, scales)
        actual = core.reconstruct_torch(torch.tensor(raw, dtype=torch.float64), torch.tensor(euro, dtype=torch.float64),
                                        torch.tensor(tau, dtype=torch.float64), arm, scales).detach().cpu().numpy()
        assert np.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_scenario_bootstrap_reports_supported_five_percent_improvement():
    baseline = np.linspace(0.8, 1.2, 10); candidate = baseline * 0.95
    point, low, high = core.bootstrap_relative_change(baseline, candidate, repetitions=1000, seed=10, improvement=True, floor=1e-12)
    assert point == pytest.approx(5.0); assert low == pytest.approx(5.0); assert high == pytest.approx(5.0)


def test_paired_signflip_and_holm_are_family_aware():
    baseline = np.arange(1.0, 11.0); candidate = baseline - 0.25
    assert core.paired_signflip_pvalue(baseline, candidate) < 0.01
    adjusted = core.holm_adjust({"a": 0.01, "b": 0.02, "c": 0.20})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.04, "c": 0.20})


def test_gate_requires_point_improvement_ci_holm_and_shared_diagnostics():
    degradation = {"delta_rmse": 1.0, "gamma_rmse": 1.0, "boundary_rmse": 1.0, "masked_residual": 4.0}
    gates = {"price_improvement_pct_min": 2.0, "ordinary_degradation_pct_max": 2.0,
             "residual_degradation_pct_max": 5.0, "hard_obstacle_violation_max": 1e-10, "family_alpha": 0.05}
    kwargs = dict(price_improvement_pct=2.5, price_improvement_ci_low_pct=0.1, holm_p_value=0.04,
                  degradation_ci_upper_pct=degradation, obstacle_violation=1e-12, gates=gates)
    assert core.candidate_passes(**kwargs)
    assert not core.candidate_passes(**dict(kwargs, price_improvement_pct=1.99))
    assert not core.candidate_passes(**dict(kwargs, price_improvement_ci_low_pct=0.0))
    assert not core.candidate_passes(**dict(kwargs, holm_p_value=0.051))
    assert not core.candidate_passes(**dict(kwargs, degradation_ci_upper_pct=dict(degradation, gamma_rmse=2.01)))


def test_smoke_never_promotes_and_budget_blocks_canonical():
    assert core.classify_status("smoke", passed=True) == "diagnostic_only"
    assert core.classify_status("canonical", passed=True, budget_exceeded=True) == "resource_blocked"
    assert core.classify_status("canonical", passed=False) == "failed_metric_gate"
    assert core.classify_status("canonical", passed=True) == "candidate_supported"


def test_runner_uses_equal_mlp_parameter_counts_and_identical_seeded_initialization():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import m10_structured_surrogate as runner
    models = [runner.build_seeded_model(seed=101, hidden=8, depth=2) for _ in core.ARMS]
    assert len({runner.model_parameter_count(model) for model in models}) == 1
    reference = list(models[0].state_dict().values())
    for model in models[1:]:
        assert all(torch.equal(a, b) for a, b in zip(reference, model.state_dict().values()))


def test_tiny_oracle_surface_has_exact_european_terminal_anchor_and_finite_diagnostics():
    pytest.importorskip("torch")
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import m10_structured_surrogate as runner
    regime = {"id": "tiny", "sigma": 0.2, "r": 0.03, "q": 0.01}
    base = {"option_type":"put","k":1.0,"t":0.25,"s_max":3.0,"m":16,"n":12,"omega":1.2,"tol":1e-10,"max_iter":20000,"american":True,"time_scheme":"cn","lcp_solver":"psor"}
    surface = runner.solve_surface(base, regime)
    assert surface["european"].shape == surface["target"].shape
    european_surface = surface["european"].reshape(surface["result"].values.shape)
    assert np.allclose(european_surface[0], surface["result"].payoff / surface["cfg"].k, atol=1e-12)
    metrics = runner.surface_metrics(surface["result"].values.copy(), surface, contact_tol=1e-8)
    assert metrics["price_rmse"] == pytest.approx(0.0, abs=1e-14)
    assert metrics["delta_rmse"] == pytest.approx(0.0, abs=1e-14)
    assert metrics["gamma_rmse"] == pytest.approx(0.0, abs=1e-14)
    assert metrics["boundary_rmse"] == pytest.approx(0.0, abs=1e-14)
    assert metrics["obstacle_violation"] <= 1e-10
    assert np.isfinite(metrics["masked_residual"])
