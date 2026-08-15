import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
STANDALONE = ROOT / "experiments" / "standalone"
CORE = STANDALONE / "m08_tensor_core.py"
SCRIPT = STANDALONE / "m08_tensor_low_rank.py"
CONFIG = ROOT / "configs" / "standalone" / "m08_tensor_low_rank.json"


def _load(path: Path, name: str):
    sys.path.insert(0, str(STANDALONE))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_config_is_exactly_eight_predeclared_configs_and_packet_gates():
    core = _load(CORE, "m08_core_config")
    config = json.loads(CONFIG.read_text())
    core.validate_config(config)
    assert len(config["configurations"]) == 8
    assert config["tt_completion_seeds"] == [101, 203, 307, 409, 503]
    assert config["gates"] == {
        "compression_ratio_min": 10.0,
        "ordinary_degradation_pct_max": 2.0,
        "residual_degradation_pct_max": 5.0,
    }
    assert config["budget_seconds"] == 3600


def test_tt_svd_reconstructs_rank_one_tensor_and_reports_actual_storage():
    core = _load(CORE, "m08_core_tt")
    a = np.arange(1.0, 5.0)
    b = np.arange(1.0, 4.0)
    c = np.array([1.0, 2.0])
    tensor = np.einsum("i,j,k->ijk", a, b, c)
    cores = core.tt_svd(tensor, max_rank=1)
    reconstructed = core.tt_reconstruct(cores)
    assert np.allclose(reconstructed, tensor, atol=1e-12)
    assert core.tt_rank_vector(cores) == [1, 1]
    assert core.tt_storage(cores) == sum(item.size for item in cores)


def test_pod_and_primal_dual_store_only_reduced_objects():
    core = _load(CORE, "m08_core_rom")
    rng = np.random.default_rng(7)
    tensor = rng.normal(size=(5, 4, 3, 2, 2))
    pod = core.pod_compress(tensor, state_modes=2, rank=2)
    assert core.pod_reconstruct(pod).shape == tensor.shape
    primal = tensor.reshape(20, -1)
    dual = np.abs(rng.normal(size=primal.shape))
    rb = core.primal_dual_greedy_basis(primal, dual, rank=2)
    p_hat, d_hat = core.primal_dual_reconstruct(rb)
    assert p_hat.shape == primal.shape
    assert d_hat.shape == dual.shape
    assert rb.stored_scalars == (
        rb.primal_basis.size + rb.dual_basis.size + rb.primal_coefficients.size + rb.dual_coefficients.size
    )


def test_tt_completion_samples_complete_training_regimes_not_cells():
    core = _load(CORE, "m08_core_completion")
    train = np.arange(20)
    observed = core.select_observed_regimes(train, fraction=0.25, seed=101)
    assert observed.ndim == 1
    assert len(observed) == 5
    assert set(observed).issubset(set(train))
    with pytest.raises(ValueError, match="complete flattened regimes"):
        core.select_observed_regimes(np.zeros((2, 3), dtype=int), fraction=0.5, seed=101)


def test_tt_completion_keeps_best_audit_checkpoint():
    torch = pytest.importorskip("torch")
    core = _load(CORE, "m08_core_completion_fit")
    del torch
    x = np.linspace(-1.0, 1.0, 6)[:, None, None]
    y = np.linspace(0.1, 0.4, 4)[None, :, None]
    z = np.linspace(0.2, 0.8, 3)[None, None, :]
    tensor = x * y + z**2
    rep, history = core.fit_tt_completion(
        tensor,
        state_modes=1,
        rank=2,
        observed_regime_indices=np.array([0, 2, 5, 7, 9, 11]),
        seed=101,
        steps=5,
        batch_size=16,
        learning_rate=0.01,
    )
    assert history[-1]["best_loss"] <= history[0]["loss"] + 1e-15
    assert core.tt_reconstruct(rep.cores).shape == tensor.shape


def test_cn_residual_evaluator_matches_repository_oracle_stencil():
    runner = _load(SCRIPT, "m08_runner_residual")
    from solvers.cn_psor import SolverConfig, solve_option

    cfg = SolverConfig(option_type="put", k=1.0, t=0.5, r=0.05, q=0.01, sigma=0.2, s_max=3.0, m=20, n=10, tol=1e-10)
    result = solve_option(cfg)
    fields = runner.cn_residual_fields(result.values, cfg, result.spots, result.taus, result.payoff)
    natural = np.minimum(fields["residual"], fields["gap"])
    assert np.max(np.abs(natural)) < 5e-8
    assert np.max(np.maximum(-fields["gap"], 0.0)) < 1e-10


def test_dual_tensor_is_built_from_nonnegative_cn_lcp_residual_snapshots():
    runner = _load(SCRIPT, "m08_runner_dual")
    regimes, axes = runner.build_regimes(
        {
            "parameter_grid": {
                "sigma": {"min": 0.2, "max": 0.2, "count": 1},
                "r": {"min": 0.05, "max": 0.05, "count": 1},
                "q": {"min": 0.0, "max": 0.0, "count": 1},
            }
        }
    )
    base = {"option_type":"put","k":1.0,"t":0.5,"s_max":3.0,"m":16,"n":8,"omega":1.2,"tol":1e-10,"max_iter":20000,"american":True,"time_scheme":"cn","lcp_solver":"psor"}
    tensor, surfaces, _ = runner.build_value_tensor(base, regimes, tuple(len(axis) for axis in axes))
    dual = runner.build_dual_tensor(tensor.shape, surfaces, regimes, base)
    assert dual.shape == tensor.shape
    assert np.min(dual) >= -1e-10
    assert np.max(dual) > 0.0


def test_bootstrap_degradation_resamples_paired_regimes():
    runner = _load(SCRIPT, "m08_runner_bootstrap")
    baseline = np.array([1.0, 2.0, 4.0, 8.0])
    candidate = baseline * 1.01
    point, upper = runner._bootstrap_upper(candidate, baseline, floor=1e-12, repetitions=200, seed=8)
    assert point == pytest.approx(1.0)
    assert upper == pytest.approx(1.0)


def test_smoke_is_never_promoted_even_if_gate_passes():
    core = _load(CORE, "m08_core_status")
    assert core.classify_status("smoke", any_configuration_passed=True) == "diagnostic_only"
    assert core.classify_status("canonical", any_configuration_passed=False) == "failed_metric_gate"
    assert core.classify_status("canonical", any_configuration_passed=True) == "candidate_supported"


def test_gate_requires_compression_and_all_shared_diagnostics():
    core = _load(CORE, "m08_core_gate")
    upper = {
        "price_rmse": 1.0,
        "delta_rmse": 1.0,
        "gamma_rmse": 1.0,
        "boundary_rmse": 1.0,
        "obstacle_violation": 1.0,
        "dual_violation": 4.0,
        "complementarity": 4.0,
        "continuation_residual": 4.0,
    }
    assert core.configuration_passes(10.0, upper, ordinary_limit_pct=2.0, residual_limit_pct=5.0)
    failed = dict(upper)
    failed["gamma_rmse"] = 2.01
    assert not core.configuration_passes(100.0, failed, ordinary_limit_pct=2.0, residual_limit_pct=5.0)
    assert not core.configuration_passes(9.99, upper, ordinary_limit_pct=2.0, residual_limit_pct=5.0)
