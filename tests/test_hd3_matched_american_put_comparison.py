from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "hd3_matched_american_put_comparison.py"
spec = importlib.util.spec_from_file_location("hd3_matched", MODULE_PATH)
assert spec and spec.loader
hd3m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hd3m)

import hd3_classical_vs_structured_high_dimensional_pinn as hd3base


def test_parameter_mapping_matches_same_reduced_contract_for_all_canonical_dimensions():
    cfg = hd3m.load_config()
    for d in (10, 20, 50):
        row = hd3m.parameter_mapping(d, cfg)
        assert math.isclose(row["effective_sigma_g"], 0.60, rel_tol=0.0, abs_tol=1e-12)
        assert math.isclose(row["effective_q_g"], 0.10, rel_tol=0.0, abs_tol=1e-12)
        assert row["rho"] == 0.80


def test_dimension_config_uses_dimension_specific_asset_parameters():
    cfg = hd3m.load_config()
    d10 = hd3m.dimension_config(10, cfg)
    d50 = hd3m.dimension_config(50, cfg)
    assert d10["contract"]["sigma"] != d50["contract"]["sigma"]
    assert d10["contract"]["q"] != d50["contract"]["q"]
    assert d10["contract"]["K"] == d50["contract"]["K"] == 1.0
    assert d10["contract"]["T"] == d50["contract"]["T"] == 1.0
    assert d10["contract"]["r"] == d50["contract"]["r"] == 0.05


def test_symmetric_diagonal_geometric_mean_equals_common_spot():
    for d in (10, 20, 50):
        spots = np.asarray([0.5, 0.9, 1.0, 1.4])
        x = np.log(spots)[:, None].repeat(d, axis=1)
        geometric_mean = np.exp(x.mean(axis=1))
        assert np.allclose(geometric_mean, spots, rtol=0.0, atol=1e-14)


def test_crr_american_put_respects_obstacle_and_is_monotone_in_spot():
    spots = np.asarray([0.6, 0.8, 1.0, 1.2, 1.4])
    values = np.asarray([
        hd3m.crr_american_put(float(s), 1.0, k=1.0, r=0.05, q=0.10, sigma=0.60, steps=256)
        for s in spots
    ])
    payoff = np.maximum(1.0 - spots, 0.0)
    assert np.all(values >= payoff - 1e-12)
    assert np.all(np.diff(values) <= 1e-10)
    assert np.all(np.isfinite(values))


def test_crr_converges_between_successive_refinements_at_reference_spot():
    values = [
        hd3m.crr_american_put(1.0, 1.0, k=1.0, r=0.05, q=0.10, sigma=0.60, steps=n)
        for n in (128, 256, 512)
    ]
    assert abs(values[2] - values[1]) < 5e-3
    assert abs(values[2] - values[1]) <= 1.5 * abs(values[1] - values[0]) + 1e-12


def test_oracle_consistency_accepts_identical_surfaces():
    spots = np.linspace(0.0, 4.0, 5)
    taus = np.linspace(0.0, 1.0, 4)
    values = np.arange(20, dtype=float).reshape(4, 5)
    zeros = np.zeros_like(values)
    boundary = np.linspace(0.8, 0.9, 4)
    oracles = {
        d: hd3m.Oracle(d, 0.60, 0.10, spots, taus, values.copy(), zeros, zeros, boundary, 0.0, 0.0)
        for d in (10, 20, 50)
    }
    result = hd3m._oracle_consistency(oracles)
    assert result["passed"] is True
    assert all(row["max_abs_difference"] == 0.0 for row in result["comparisons"])


def test_cpu_memory_telemetry_failure_is_nonfatal(monkeypatch):
    def fail_memory_probe():
        raise OSError("GetProcessMemoryInfo failed")

    monkeypatch.setattr(hd3base, "peak_cpu_memory_bytes", fail_memory_probe)
    assert hd3base._safe_peak_cpu_memory_bytes() is None
