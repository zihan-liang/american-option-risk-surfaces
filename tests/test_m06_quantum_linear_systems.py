import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "standalone" / "m06_quantum_linear_systems.py"


def _module():
    spec = importlib.util.spec_from_file_location("m06_standalone", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hardware_efficient_ansatz_is_normalized_and_dimensionally_exact():
    module = _module()
    state = module.hardware_efficient_state(np.zeros(6), qubits=3, layers=2)
    assert state.shape == (8,)
    assert np.isclose(np.linalg.norm(state), 1.0)


def test_vqls_solves_one_qubit_system_with_low_normalized_residual():
    module = _module()
    matrix = np.array([[1.8, -0.2], [-0.2, 1.4]])
    rhs = np.array([0.3, 0.9])
    result = module.vqls_solve(
        matrix,
        rhs,
        seed=101,
        layers=1,
        iterations=250,
        learning_rate=0.08,
        finite_difference_step=1.0e-4,
        cost_tolerance=1.0e-12,
    )
    assert result["normalized_residual"] < 1.0e-5
    assert result["state_fidelity"] > 0.99999
    assert result["objective_evaluations"] > 0


def test_pauli_audit_reconstructs_actual_matrix_with_declared_threshold():
    module = _module()
    matrix = np.array(
        [[2.0, -0.2, 0.0, 0.0], [-0.3, 2.1, -0.2, 0.0], [0.0, -0.3, 2.2, -0.2], [0.0, 0.0, -0.3, 2.3]]
    )
    audit = module.pauli_encoding_audit(matrix, truncation=1.0e-12)
    assert 0 < audit["retained_terms"] <= 16
    assert audit["relative_spectral_error"] < 1.0e-10


def test_smoke_screen_writes_bounded_hashed_nonpromotion_evidence(tmp_path):
    module = _module()
    output = tmp_path / "M06"
    result = module.run_screen(
        {
            "method_id": "M06",
            "phase": "screen",
            "output_root": str(output),
            "solver": {"k": 1.0, "t": 1.0, "r": 0.05, "q": 0.0, "sigma": 0.2, "s_max": 4.0, "time_fraction": 0.5, "time_steps": 8, "tol": 1e-10, "max_iter": 10000},
            "vqls": {"dimensions": [4, 8], "seeds": [101], "layers": 1, "iterations": 3, "learning_rate": 0.05, "finite_difference_step": 1e-4, "cost_tolerance": 1e-12},
            "resources": {"pauli_truncation": 1e-10, "target_inverse_error": 1e-3, "projected_shots_per_expectation": 1000},
            "gates": {"quantum_error_fraction_of_cn_floor": 0.25},
        },
        provenance={"branch": "test", "commit": "test"},
    )
    assert result["status"] == "diagnostic_only"
    assert result["claim_status"] == "diagnostic_only"
    assert result["blind_holdout_accessed"] is False
    assert len(result["metrics"]) == 2
    assert {row["dimension"] for row in result["resources"]} == {4, 8}
    assert all(row["executed_shots"] == 0 for row in result["metrics"])

    expected = {"artifact_hashes.json", "config_manifest.json", "decision.json", "metrics.csv", "resource_accounting.csv"}
    assert {path.name for path in output.iterdir()} == expected
    hashes = json.loads((output / "artifact_hashes.json").read_text())["files"]
    for name, digest in hashes.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
