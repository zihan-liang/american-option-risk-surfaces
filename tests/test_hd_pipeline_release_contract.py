from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_release_runner_calls_release_orchestrator_and_never_installs():
    text = (ROOT / "scripts" / "run_hd2_hd3_release_windows.ps1").read_text(encoding="utf-8")
    assert "experiments\\hd_pipeline_windows_release.py" in text
    assert ".\\.venv-pinn\\Scripts\\python.exe" in text
    assert "$LASTEXITCODE" in text
    for phase in ("all", "preflight", "hd2", "hd3", "finalize", "status"):
        assert phase in text
    lowered = text.lower()
    assert "pip install" not in lowered
    assert "conda install" not in lowered
    assert "winget" not in lowered


def test_release_preflight_uses_release_cagrad_test_not_superseded_development_test():
    text = (ROOT / "experiments" / "hd_pipeline_windows_release.py").read_text(encoding="utf-8")
    assert "tests/test_hd_objective_optimizers_release.py" in text
    assert '"tests/test_hd_objective_optimizers.py"' not in text


def test_release_windows_cpu_memory_telemetry_failure_is_nonfatal():
    text = (ROOT / "experiments" / "hd_pipeline_windows_release.py").read_text(encoding="utf-8")
    assert "safe_peak_cpu_memory_bytes" in text
    assert "except OSError" in text
    assert "return 0" in text
    assert "pipeline.hd2.peak_cpu_memory_bytes = safe_peak_cpu_memory_bytes" in text
    assert "pipeline.hd3.peak_cpu_memory_bytes = safe_peak_cpu_memory_bytes" in text


def test_release_status_wrapper_is_read_only():
    text = (ROOT / "scripts" / "check_hd2_hd3_release_status.ps1").read_text(encoding="utf-8")
    assert "run_hd2_hd3_release_windows.ps1" in text
    assert "-Phase status" in text
    assert "Remove-Item" not in text
    assert "Set-Content" not in text
