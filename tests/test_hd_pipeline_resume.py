from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from hd_pipeline.runtime import (
    append_jsonl,
    atomic_json,
    capture_rng_state,
    load_checkpoint_verified,
    read_jsonl_recover,
    restore_rng_state,
    save_checkpoint_atomic,
    sha256_file,
    verified_completed_status,
)


def test_jsonl_recovery_ignores_only_incomplete_final_line(tmp_path: Path):
    path = tmp_path / "history.jsonl"
    append_jsonl(path, {"step": 100, "loss": 1.0})
    append_jsonl(path, {"step": 200, "loss": 0.5})
    with path.open("ab") as handle:
        handle.write(b'{"step":300')
    assert [row["step"] for row in read_jsonl_recover(path)] == [100, 200]


def test_jsonl_recovery_rejects_corruption_before_final_line(tmp_path: Path):
    path = tmp_path / "history.jsonl"
    path.write_text('{"step":100}\nnot-json\n{"step":300}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="corrupt_jsonl"):
        read_jsonl_recover(path)


def test_atomic_json_replaces_previous_document(tmp_path: Path):
    path = tmp_path / "status.json"
    atomic_json(path, {"status": "running", "step": 250})
    atomic_json(path, {"status": "completed", "step": 500})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "completed", "step": 500}
    assert not list(tmp_path.glob("*.tmp"))


def test_checkpoint_roundtrip_and_stale_provenance_rejection(tmp_path: Path):
    path = tmp_path / "latest_checkpoint.pt"
    payload = {
        "implementation_sha": "abc",
        "config_hash": "cfg",
        "job_id": "d5_HVI-low",
        "completed_step": 750,
        "model_state": {"w": torch.tensor([1.0])},
    }
    save_checkpoint_atomic(path, payload)
    got = load_checkpoint_verified(
        path,
        implementation_sha="abc",
        config_hash="cfg",
        job_id="d5_HVI-low",
    )
    assert got["completed_step"] == 750
    assert torch.equal(got["model_state"]["w"], torch.tensor([1.0]))
    with pytest.raises(RuntimeError, match="stale_provenance"):
        load_checkpoint_verified(
            path,
            implementation_sha="different",
            config_hash="cfg",
            job_id="d5_HVI-low",
        )


def test_completed_status_requires_current_hashes(tmp_path: Path):
    artifact = tmp_path / "metrics.json"
    artifact.write_text('{"ok": true}\n', encoding="utf-8")
    status = tmp_path / "status.json"
    atomic_json(
        status,
        {
            "status": "completed",
            "implementation_sha": "abc",
            "config_hash": "cfg",
            "artifacts": {str(artifact): sha256_file(artifact)},
        },
    )
    assert verified_completed_status(status, implementation_sha="abc", config_hash="cfg")
    artifact.write_text('{"ok": false}\n', encoding="utf-8")
    assert not verified_completed_status(status, implementation_sha="abc", config_hash="cfg")


def test_rng_roundtrip_restores_torch_stream():
    torch.manual_seed(123)
    state = capture_rng_state()
    expected = torch.rand(4)
    torch.rand(8)
    restore_rng_state(state)
    assert torch.equal(torch.rand(4), expected)
