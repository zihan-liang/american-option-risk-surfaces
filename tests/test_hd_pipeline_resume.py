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
from hd_pipeline_windows import (
    run_state_machine,
    should_enable_parallel,
)

ROOT = Path(__file__).resolve().parents[1]


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


def _fake_jobs(calls: list[str]):
    def job(name: str):
        def run():
            calls.append(name)
            return {"phase": name, "ok": True}
        return run

    return {
        "preflight": job("preflight"),
        "hd2": job("hd2"),
        "hd2_select": job("hd2_select"),
        "hd3": job("hd3"),
        "finalize": job("finalize"),
    }


def test_pipeline_resumes_after_middle_phase_without_rerunning_completed(tmp_path: Path):
    calls: list[str] = []
    jobs = _fake_jobs(calls)
    run_state_machine(
        tmp_path,
        jobs,
        implementation_sha="abc",
        config_hash="cfg",
        stop_after="hd2_select",
    )
    assert calls == ["preflight", "hd2", "hd2_select"]
    calls.clear()
    run_state_machine(
        tmp_path,
        jobs,
        implementation_sha="abc",
        config_hash="cfg",
    )
    assert calls == ["hd3", "finalize"]


def test_pipeline_rejects_stale_phase_provenance(tmp_path: Path):
    calls: list[str] = []
    run_state_machine(
        tmp_path,
        _fake_jobs(calls),
        implementation_sha="abc",
        config_hash="cfg",
        stop_after="hd2",
    )
    with pytest.raises(RuntimeError, match="stale_pipeline_provenance"):
        run_state_machine(
            tmp_path,
            _fake_jobs([]),
            implementation_sha="different",
            config_hash="cfg",
        )


def test_parallelism_requires_speed_gain_memory_headroom_and_success():
    assert should_enable_parallel(
        fc_sequential_seconds=100.0,
        deepsets_sequential_seconds=120.0,
        concurrent_wall_seconds=180.0,
        concurrent_peak_cuda_gib=6.5,
        jobs_succeeded=True,
        speed_ratio_max=0.90,
        memory_limit_gib=7.2,
    )
    assert not should_enable_parallel(
        fc_sequential_seconds=100.0,
        deepsets_sequential_seconds=120.0,
        concurrent_wall_seconds=205.0,
        concurrent_peak_cuda_gib=6.5,
        jobs_succeeded=True,
        speed_ratio_max=0.90,
        memory_limit_gib=7.2,
    )
    assert not should_enable_parallel(
        fc_sequential_seconds=100.0,
        deepsets_sequential_seconds=120.0,
        concurrent_wall_seconds=180.0,
        concurrent_peak_cuda_gib=7.3,
        jobs_succeeded=True,
        speed_ratio_max=0.90,
        memory_limit_gib=7.2,
    )
    assert not should_enable_parallel(
        fc_sequential_seconds=100.0,
        deepsets_sequential_seconds=120.0,
        concurrent_wall_seconds=180.0,
        concurrent_peak_cuda_gib=6.5,
        jobs_succeeded=False,
        speed_ratio_max=0.90,
        memory_limit_gib=7.2,
    )


def test_windows_runner_exposes_only_supported_phases_and_never_installs_packages():
    path = ROOT / "scripts" / "run_hd_pipeline_windows.ps1"
    text = path.read_text(encoding="utf-8")
    for phase in ("all", "preflight", "hd2", "hd3", "finalize", "status"):
        assert phase in text
    assert "experiments\\hd_pipeline_windows.py" in text
    assert "$LASTEXITCODE" in text
    lowered = text.lower()
    assert "pip install" not in lowered
    assert "conda install" not in lowered
    assert "winget" not in lowered
    assert ".venv-pinn" in text


def test_status_wrapper_is_read_only_pipeline_status_call():
    path = ROOT / "scripts" / "check_hd_pipeline_status.ps1"
    text = path.read_text(encoding="utf-8")
    assert "run_hd_pipeline_windows.ps1" in text
    assert "-Phase status" in text
    assert "Remove-Item" not in text
    assert "Set-Content" not in text
