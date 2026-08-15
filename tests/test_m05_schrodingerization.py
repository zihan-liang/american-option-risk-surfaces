import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "standalone" / "m05_schrodingerization.py"


def _module():
    spec = importlib.util.spec_from_file_location("m05_standalone", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dissipative_recovery_converges_without_using_target_solution():
    module = _module()
    matrix = np.array([[2.0, -0.4], [-0.4, 1.5]])
    rhs = np.array([0.3, 0.8])
    recovered, diagnostics = module.dissipative_recovery(
        matrix, rhs, np.zeros(2), decay_tolerance=1.0e-10
    )

    assert np.allclose(recovered, np.linalg.solve(matrix, rhs), atol=2.0e-10, rtol=0.0)
    assert diagnostics["horizon"] > 0.0
    assert diagnostics["normal_condition_number"] >= 1.0


def test_screen_writes_recomputable_diagnostic_only_evidence(tmp_path):
    module = _module()
    output = tmp_path / "M05"
    result = module.run_screen(
        {
            "method_id": "M05",
            "phase": "screen",
            "output_root": str(output),
            "solver": {
                "k": 1.0,
                "t": 1.0,
                "r": 0.05,
                "q": 0.0,
                "sigma": 0.2,
                "s_max": 4.0,
                "medium_m": 16,
                "medium_n": 16,
                "fine_multiplier": 2,
                "tol": 1.0e-11,
                "max_iter": 20000,
            },
            "experiment": {"time_fraction": 0.5, "decay_tolerance": 1.0e-8},
            "gates": {"obstacle_tolerance": 1.0e-10},
        },
        provenance={"branch": "test", "commit": "test"},
    )

    assert result["status"] == "diagnostic_only"
    assert result["claim_status"] == "diagnostic_only"
    assert result["blind_holdout_accessed"] is False
    assert {row["variant"] for row in result["metrics"]} == {
        "frozen_continuation",
        "one_live_obstacle_update",
    }
    assert all(np.isfinite(row["inner_rmse"]) for row in result["metrics"])
    assert all(row["classical_discretization_floor"] > 0.0 for row in result["metrics"])
    assert result["metrics"][1]["obstacle_violation"] <= 1.0e-10

    expected = {
        "artifact_hashes.json",
        "config_manifest.json",
        "decision.json",
        "metrics.csv",
    }
    assert {path.name for path in output.iterdir()} == expected
    hashes = json.loads((output / "artifact_hashes.json").read_text())["files"]
    for name, digest in hashes.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest


def test_invalid_decay_tolerance_is_rejected():
    module = _module()
    with np.testing.assert_raises(ValueError):
        module.dissipative_recovery(
            np.eye(2), np.ones(2), np.zeros(2), decay_tolerance=1.0
        )
