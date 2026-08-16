import json
import sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STANDALONE = REPO_ROOT / "experiments" / "standalone"
sys.path.insert(0, str(STANDALONE))

import m09_multifidelity_core as core  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "standalone" / "m09_multifidelity.json"


def test_config_freezes_packet_design_and_campaign_limits():
    config = json.loads(CONFIG.read_text())
    core.validate_config(config)
    assert config["training_seeds"] == [101, 203, 307, 409, 503]
    assert config["fine_fractions"] == [0.10, 0.25, 0.50, 1.00]
    assert config["budget_seconds"] == 3600
    assert config["coarse_solver"]["m"] == config["coarse_solver"]["n"] == 60
    assert config["fine_solver"]["m"] == config["fine_solver"]["n"] == 120
    assert config["model"]["device"] == "cpu"
    assert config["model"]["dtype"] == "float64"


def test_fine_fraction_selection_is_nested_and_seed_independent():
    keys = [f"r{i:02d}" for i in range(20)]
    a = core.nested_regime_subsets(keys, core.FINE_FRACTIONS, selection_seed=909)
    b = core.nested_regime_subsets(keys, core.FINE_FRACTIONS, selection_seed=909)
    assert a == b
    assert set(a[0.10]) < set(a[0.25]) < set(a[0.50]) < set(a[1.00])
    assert set(a[1.00]) == set(keys)


def test_complete_surface_guard_rejects_partial_regime_rows():
    keys = np.array(["a"] * 9 + ["b"] * 10)
    with pytest.raises(ValueError, match="complete surface"):
        core.assert_complete_regime_rows(keys, 10, ["a"])
    mask = core.assert_complete_regime_rows(keys, 10, ["b"])
    assert int(mask.sum()) == 10


def test_matching_fraction_is_selected_using_validation_error_only():
    target = core.matched_accuracy_target(1.0, 0.02)
    selected = core.choose_smallest_matching_fraction({0.10: 1.4, 0.25: 1.01, 0.50: 0.95, 1.00: 0.9}, target)
    assert target == pytest.approx(1.02)
    assert selected == 0.25
    assert core.choose_smallest_matching_fraction({0.10: 1.4, 0.25: 1.3}, target) is None


def test_bootstrap_ci_and_degradation_are_paired():
    candidate = np.array([1.01, 2.02, 4.04, 8.08])
    baseline = np.array([1.0, 2.0, 4.0, 8.0])
    degradation = core.percent_degradation(candidate, baseline, floor=1e-12)
    mean, low, high = core.paired_bootstrap_mean_ci(degradation, repetitions=200, seed=9)
    assert mean == pytest.approx(1.0)
    assert low == pytest.approx(1.0)
    assert high == pytest.approx(1.0)


def test_gate_requires_cost_ci_and_every_shared_diagnostic():
    upper = {name: 1.0 for name in core.ORDINARY_METRICS}
    upper.update({name: 4.0 for name in core.RESIDUAL_METRICS})
    gates = {
        "fine_label_cost_reduction_pct_min": 2.0,
        "ordinary_degradation_pct_max": 2.0,
        "residual_degradation_pct_max": 5.0,
        "hard_obstacle_violation_max": 1e-10,
    }
    assert core.candidate_passes(cost_reduction_ci_low_pct=2.1, degradation_ci_upper_pct=upper, obstacle_violation=1e-12, gates=gates)
    failed = dict(upper); failed["gamma_rmse"] = 2.01
    assert not core.candidate_passes(cost_reduction_ci_low_pct=2.1, degradation_ci_upper_pct=failed, obstacle_violation=1e-12, gates=gates)
    assert not core.candidate_passes(cost_reduction_ci_low_pct=1.99, degradation_ci_upper_pct=upper, obstacle_violation=1e-12, gates=gates)


def test_smoke_can_never_promote():
    assert core.classify_status("smoke", passed=True) == "diagnostic_only"
    assert core.classify_status("canonical", passed=False) == "failed_metric_gate"
    assert core.classify_status("canonical", passed=True) == "candidate_supported"


def test_runner_uses_same_mlp_parameter_count_and_weight_transfer():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import m09_multifidelity as runner
    from models.mlp import build_mlp

    model_a = build_mlp(7, 1, hidden=8, depth=2).double()
    model_b = build_mlp(7, 1, hidden=8, depth=2).double()
    assert runner.model_parameter_count(model_a) == runner.model_parameter_count(model_b)
    with torch.no_grad():
        for parameter in model_a.parameters():
            parameter.fill_(0.125)
    transferred = __import__("copy").deepcopy(model_a)
    assert all(torch.equal(a, b) for a, b in zip(model_a.parameters(), transferred.parameters()))


def test_tiny_matched_surfaces_are_complete_and_diagnostics_accept_oracle():
    pytest.importorskip("torch")
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import m09_multifidelity as runner

    regime = {"id": "tiny", "sigma": 0.2, "r": 0.03, "q": 0.01}
    coarse_base = {"option_type":"put","k":1.0,"t":0.25,"s_max":3.0,"m":12,"n":8,"omega":1.2,"tol":1e-10,"max_iter":20000,"american":True,"time_scheme":"cn","lcp_solver":"psor"}
    fine_base = dict(coarse_base, m=24, n=16)
    coarse = runner.solve_surface(coarse_base, regime, "coarse")
    fine = runner.solve_surface(fine_base, regime, "fine")
    assert coarse["regime_id"] == fine["regime_id"] == "tiny"
    assert len(coarse["target"]) == (coarse_base["m"] + 1) * (coarse_base["n"] + 1)
    assert len(fine["target"]) == (fine_base["m"] + 1) * (fine_base["n"] + 1)
    metrics = runner.surface_metrics(fine["result"].values.copy(), fine, contact_tol=1e-8)
    assert metrics["price_rmse"] == pytest.approx(0.0, abs=1e-14)
    assert metrics["delta_rmse"] == pytest.approx(0.0, abs=1e-14)
    assert metrics["gamma_rmse"] == pytest.approx(0.0, abs=1e-14)
    assert metrics["boundary_rmse"] == pytest.approx(0.0, abs=1e-14)
    assert metrics["obstacle_violation"] <= 1e-10


def test_constraint_projection_enforces_payoff_terminal_and_spatial_boundaries():
    pytest.importorskip("torch")
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import m09_multifidelity as runner

    regime = {"id": "tiny", "sigma": 0.25, "r": -0.01, "q": 0.02}
    base = {"option_type":"put","k":1.0,"t":0.25,"s_max":3.0,"m":12,"n":8,"omega":1.2,"tol":1e-10,"max_iter":20000,"american":True,"time_scheme":"cn","lcp_solver":"psor"}
    surface = runner.solve_surface(base, regime, "fine")
    bad = np.full_like(surface["result"].values, -5.0)
    repaired = runner.enforce_known_constraints(bad, surface["result"], surface["cfg"])
    assert np.min(repaired - surface["result"].payoff[None, :]) >= -1e-12
    assert np.allclose(repaired[0], surface["result"].payoff)
