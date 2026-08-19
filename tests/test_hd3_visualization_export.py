from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import hd3_visualization_export as export_module
from hd3_visualization_export import (
    CANONICAL_EVIDENCE_SHA,
    TRAINING_IMPLEMENTATION_SHA,
    CheckpointExpectation,
    build_parser,
    diagonal_states,
    effective_contract,
    infer_prices,
    load_completed_model_for_visualization,
    select_crr_depth,
    sha256_path,
    state_dict_digest,
    validate_checkpoint_metadata,
    write_rows_csv,
    write_visualization_manifest,
)


class TinyModel(torch.nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.linear = torch.nn.Linear(d + 1, 1)

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat([x, tau[:, None]], dim=1)).reshape(-1)


def expectation() -> CheckpointExpectation:
    return CheckpointExpectation(
        training_sha="train-sha", config_hash="cfg-hash", objective_hash="obj-hash",
        job_id="canonical_d10_FC_HVI", dimension=10, model_id="FC_HVI",
        completed_steps=6000, run_class="canonical",
    )


def valid_checkpoint_metadata() -> dict:
    return {
        "implementation_sha": "train-sha", "config_hash": "cfg-hash",
        "resolved_objective_hash": "obj-hash", "job_id": "canonical_d10_FC_HVI",
        "dimension": 10, "model_id": "FC_HVI", "run_class": "canonical",
        "completed_step": 6000, "target_steps": 6000,
    }


@pytest.mark.parametrize("field,bad", [
    ("implementation_sha", "wrong"), ("config_hash", "wrong"),
    ("resolved_objective_hash", "wrong"), ("job_id", "wrong"),
    ("dimension", 20), ("model_id", "DEEPSETS_HVI"), ("run_class", "stress"),
    ("completed_step", 5999), ("target_steps", 5999),
])
def test_checkpoint_provenance_rejects_mismatch(field: str, bad):
    payload = copy.deepcopy(valid_checkpoint_metadata())
    payload[field] = bad
    with pytest.raises(RuntimeError, match="checkpoint_provenance_mismatch"):
        validate_checkpoint_metadata(payload, expectation())


def test_visualization_loader_loads_state_without_optimizer_or_parameter_mutation(tmp_path: Path, monkeypatch):
    torch.manual_seed(17)
    source = TinyModel(10)
    state = {k: v.detach().clone() for k, v in source.state_dict().items()}
    path = tmp_path / "checkpoint.pt"
    torch.save(valid_checkpoint_metadata() | {"model_state": state}, path)
    monkeypatch.setattr(export_module.core, "_initialize_model", lambda model_id, d, cfg, device: TinyModel(d).to(device))
    loaded, checkpoint = load_completed_model_for_visualization(path, expectation(), {}, torch.device("cpu"))
    assert not loaded.training
    for key, tensor in state.items():
        torch.testing.assert_close(loaded.state_dict()[key], tensor)
    assert checkpoint["completed_step"] == 6000
    before = state_dict_digest(loaded)
    y = infer_prices(loaded, torch.zeros(7, 10), torch.linspace(0.1, 1.0, 7), batch_size=3)
    assert y.shape == (7,)
    assert np.all(np.isfinite(y))
    assert state_dict_digest(loaded) == before


def test_select_crr_depth_chooses_smallest_passing_depth():
    rows = [
        {"depth": 500, "cnpsor_rmse": 2e-4, "previous_depth_rmse": None},
        {"depth": 1000, "cnpsor_rmse": 8e-5, "previous_depth_rmse": 7e-5},
        {"depth": 2000, "cnpsor_rmse": 6e-5, "previous_depth_rmse": 4e-5},
        {"depth": 4000, "cnpsor_rmse": 5e-5, "previous_depth_rmse": 2e-5},
    ]
    assert select_crr_depth(rows) == (2000, True)


def test_select_crr_depth_falls_back_to_largest_depth_and_marks_failure():
    rows = [
        {"depth": 500, "cnpsor_rmse": 3e-4, "previous_depth_rmse": None},
        {"depth": 4000, "cnpsor_rmse": 2e-4, "previous_depth_rmse": 9e-5},
    ]
    assert select_crr_depth(rows) == (4000, False)


def test_diagonal_states_have_equal_log_prices():
    x, tau, spots = diagonal_states(20, 0.5, 5, torch.device("cpu"))
    assert x.shape == (5, 20)
    torch.testing.assert_close(x, x[:, :1].repeat(1, 20))
    np.testing.assert_allclose(np.exp(x[:, 0].numpy()), spots, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(tau.numpy(), np.full(5, 0.5))


def test_effective_contract_reuses_project_parameter_helper(monkeypatch):
    cfg = {"contract": {"K": 1.0, "T": 1.0, "r": 0.05, "q": 0.02, "sigma": 0.25, "rho": 0.30}}
    calls = []
    def fake(d, sigma, q, rho):
        calls.append((d, sigma, q, rho)); return 0.123, 0.045
    monkeypatch.setattr(export_module.core, "_effective_geometric_basket_parameters", fake)
    assert effective_contract(20, cfg) == {"K": 1.0, "T": 1.0, "r": 0.05, "sigma_g": 0.123, "q_g": 0.045}
    assert calls == [(20, 0.25, 0.02, 0.30)]


def test_write_rows_csv_preserves_stress_labels_and_sha256(tmp_path: Path):
    path = tmp_path / "rows.csv"
    write_rows_csv(path, [{"dimension": 10, "run_class": "canonical"}, {"dimension": 100, "run_class": "stress"}])
    assert "100,stress" in path.read_text()
    assert len(sha256_path(path)) == 64


def test_committed_frozen_evidence_hashes_match_training_manifest():
    cfg, objective, config_hash, objective_hash = export_module.load_frozen_evidence()
    manifest = json.loads(export_module.core.CONFIG_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert config_hash == manifest["config_sha256"]
    assert objective_hash == manifest["resolved_objective_hash"]
    assert manifest["git_commit"] == TRAINING_IMPLEMENTATION_SHA
    assert cfg == manifest["config"]
    assert objective["candidate"] == "HVI-warm-low"


@pytest.mark.parametrize("mode", ["--all", "--reference-only", "--data-only", "--figures-only"])
def test_cli_accepts_declared_modes(mode: str):
    args = build_parser().parse_args([mode])
    assert sum(bool(getattr(args, name)) for name in ("all", "reference_only", "data_only", "figures_only")) == 1


def test_cli_rejects_multiple_modes():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--all", "--figures-only"])


def test_manifest_records_zero_training_and_no_blind_holdout(tmp_path: Path):
    path = write_visualization_manifest(
        output_path=tmp_path / "visualization_manifest.json", mode="all",
        visualization_git_sha="viz-sha", config_hash="cfg-hash", objective_hash="obj-hash",
        selected_crr={10: {"depth": 1000, "passed": True}}, inference_device="cpu",
        output_files=[tmp_path / "dummy.csv"],
    )
    payload = json.loads(path.read_text())
    assert payload["optimizer_steps"] == 0
    assert payload["retraining"] is False
    assert payload["blind_holdout_access"] is False
    assert payload["training_implementation_sha"] == TRAINING_IMPLEMENTATION_SHA
    assert payload["canonical_evidence_commit"] == CANONICAL_EVIDENCE_SHA
    assert payload["visualization_code_git_sha"] == "viz-sha"
