"""One-command resumable Windows orchestrator for HD2 -> HD3.

Scientific training logic stays in the HD2/HD3 experiment modules.  This file
owns phase orchestration, phase-level provenance, the global active-time budget,
and compact status reporting.  Re-running ``--phase all`` is the normal resume
operation after interruption or reboot.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hd_pipeline.runtime import atomic_json, sha256_file
import hd2_hvi_objective_stabilization as hd2
import hd3_classical_vs_structured_high_dimensional_pinn as hd3
from hd1_windows_high_dimensional_pinn_48h import environment_manifest, git_commit

PIPELINE_DIR = ROOT / "results" / "hd_pipeline"
PIPELINE_RUNTIME_DIR = PIPELINE_DIR / "runtime"
PHASE_STATUS_DIR = PIPELINE_RUNTIME_DIR / "phases"
LOG_DIR = PIPELINE_RUNTIME_DIR / "logs"
HD2_CONFIG = ROOT / "configs" / "hd2_hvi_objective_stabilization.json"
HD3_CONFIG = ROOT / "configs" / "hd3_classical_vs_structured_high_dimensional_pinn.json"
DEFAULT_PHASES = [
    "preflight",
    "hd2_phase_a",
    "hd2_screen",
    "hd2_conditional",
    "hd2_select",
    "hd2_finalize",
    "hd3_preflight",
    "hd3_10d",
    "hd3_20d",
    "hd3_50d",
    "hd3_100d_stress",
    "finalize",
]


def combined_config_hash() -> str:
    import hashlib

    digest = hashlib.sha256()
    for path in (HD2_CONFIG, HD3_CONFIG):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def should_enable_parallel(
    *,
    fc_sequential_seconds: float,
    deepsets_sequential_seconds: float,
    concurrent_wall_seconds: float,
    concurrent_peak_cuda_gib: float,
    jobs_succeeded: bool,
    speed_ratio_max: float,
    memory_limit_gib: float,
) -> bool:
    sequential_sum = float(fc_sequential_seconds) + float(deepsets_sequential_seconds)
    if sequential_sum <= 0.0:
        return False
    return bool(
        jobs_succeeded
        and float(concurrent_wall_seconds) <= float(speed_ratio_max) * sequential_sum
        and float(concurrent_peak_cuda_gib) < float(memory_limit_gib)
    )


def _phase_path(state_root: Path, phase: str) -> Path:
    return Path(state_root) / "phase_status" / f"{phase}.json"


def _read_phase_status(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("corrupt_pipeline_phase_status") from exc
    if not isinstance(value, dict):
        raise RuntimeError("corrupt_pipeline_phase_status")
    return value


def run_state_machine(
    state_root: Path,
    jobs: Mapping[str, Callable[[], Any]],
    *,
    implementation_sha: str,
    config_hash: str,
    stop_after: str | None = None,
) -> list[dict[str, Any]]:
    """Run/resume ordered phase callables with strict provenance checking."""
    state_root = Path(state_root)
    completed: list[dict[str, Any]] = []
    for phase, job in jobs.items():
        status_path = _phase_path(state_root, phase)
        previous = _read_phase_status(status_path)
        if previous is not None:
            if (
                str(previous.get("implementation_sha", "")) != str(implementation_sha)
                or str(previous.get("config_hash", "")) != str(config_hash)
            ):
                raise RuntimeError("stale_pipeline_provenance")
            if previous.get("status") == "completed":
                completed.append(previous)
                if stop_after == phase:
                    break
                continue
        started = time.perf_counter()
        atomic_json(
            status_path,
            {
                "phase": phase,
                "status": "running",
                "implementation_sha": implementation_sha,
                "config_hash": config_hash,
                "started_at_epoch": time.time(),
                "previous_status": previous.get("status") if previous else None,
            },
        )
        try:
            result = job()
        except KeyboardInterrupt:
            atomic_json(
                status_path,
                {
                    "phase": phase,
                    "status": "interrupted",
                    "implementation_sha": implementation_sha,
                    "config_hash": config_hash,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            raise
        except Exception as exc:
            atomic_json(
                status_path,
                {
                    "phase": phase,
                    "status": "failed",
                    "implementation_sha": implementation_sha,
                    "config_hash": config_hash,
                    "elapsed_seconds": time.perf_counter() - started,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise
        payload = {
            "phase": phase,
            "status": "completed",
            "implementation_sha": implementation_sha,
            "config_hash": config_hash,
            "elapsed_seconds": time.perf_counter() - started,
            "completed_at_epoch": time.time(),
            "result": result,
        }
        atomic_json(status_path, payload)
        completed.append(payload)
        if stop_after == phase:
            break
    return completed


def _tracked_source_dirty() -> list[str]:
    """Return tracked modifications outside results/; result evidence may be dirty on resume."""
    raw = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    )
    dirty: list[str] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip().replace("\\", "/")
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if not path.startswith("results/"):
            dirty.append(path)
    return dirty


def _run_focused_tests() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "experiments")])
    env["MPLBACKEND"] = "Agg"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_hd1_high_dimensional_pinn.py",
        "tests/test_hd2_hvi_objective_stabilization.py",
        "tests/test_hd3_high_dimensional_pinn.py",
        "tests/test_hd_pipeline_resume.py",
    ]
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _common_preflight() -> dict[str, Any]:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    if branch != "exp/hd2-hvi-objective-stabilization":
        raise RuntimeError(f"wrong_branch:{branch}")
    dirty = _tracked_source_dirty()
    if dirty:
        raise RuntimeError("tracked_source_dirty:" + ",".join(dirty))
    if shutil.disk_usage(ROOT).free < 5 * 1024**3:
        raise RuntimeError("disk_space_blocked:need_5_gib_free")
    try:
        nvidia_smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        raise RuntimeError("nvidia_smi_required") from exc
    _run_focused_tests()
    manifest = environment_manifest()
    if not manifest.get("cuda_available"):
        raise RuntimeError("cuda_required")
    if int(manifest.get("gpu_total_memory_bytes") or 0) < 7 * 1024**3:
        raise RuntimeError("cuda_memory_blocked")
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "branch": branch,
        "git_commit": git_commit(),
        "combined_config_hash": combined_config_hash(),
        "nvidia_smi": nvidia_smi,
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
        "environment": manifest,
    }
    atomic_json(PIPELINE_DIR / "environment_manifest.json", payload)
    return payload


def _active_elapsed_seconds() -> float:
    total = 0.0
    if not PHASE_STATUS_DIR.exists():
        return 0.0
    for path in PHASE_STATUS_DIR.glob("*.json"):
        status = _read_phase_status(path)
        if status and status.get("status") == "completed":
            total += float(status.get("elapsed_seconds", 0.0))
    return total


def _hours_remaining() -> float:
    cfg = hd3.load_config()
    return max(0.0, float(cfg["pipeline_wall_clock_hours"]) - _active_elapsed_seconds() / 3600.0)


def _hd2_conditional_placeholder() -> dict[str, Any]:
    """Record trigger state; concrete conditional methods are added in the next TDD task."""
    # Do not silently pretend conditionals ran.  The pipeline will block if a
    # mandatory trigger exists until the corresponding implementation is present.
    cfg = hd2.load_config()
    trigger_rows: list[dict[str, Any]] = []
    for d in cfg["dimensions"]:
        path = hd2._job_dir(int(d), "HVI-HD1-anchor") / "gradient_diagnostics.jsonl"
        if path.exists():
            from hd_pipeline.runtime import read_jsonl_recover

            result = hd2.persistent_conflict(
                read_jsonl_recover(path),
                warm_start_steps=int(cfg["candidate_settings"]["HVI-HD1-anchor"]["warm_start_steps"]),
                median_threshold=float(cfg["persistent_conflict_median_cosine"]),
            )
            result["dimension"] = int(d)
            trigger_rows.append(result)
    from hd_pipeline.runtime import read_jsonl_recover

    hutch = read_jsonl_recover(
        hd2._job_dir(5, "HVI-HD1-anchor") / "hutchinson_gradient_validation.jsonl"
    )
    conflict = any(bool(row.get("triggered")) for row in trigger_rows)
    eight_probe = any(bool(row.get("conditional_eight_probe_trigger")) for row in hutch)
    payload = {
        "persistent_conflict": trigger_rows,
        "cagrad_triggered": conflict,
        "eight_probe_triggered": eight_probe,
        "status": "no_trigger" if not conflict and not eight_probe else "conditional_required",
    }
    atomic_json(ROOT / "results" / "hd2_hvi_objective_stabilization" / "conditional_methods.json", payload)
    if conflict or eight_probe:
        raise RuntimeError("conditional_hd2_method_required_but_not_yet_executed")
    return payload


def _run_hd3_dimension(d: int) -> list[dict[str, Any]]:
    """Current safe default; parallel execution is enabled by a later benchmark phase."""
    cfg = hd3.load_config()
    return [
        hd3.train_hd3_job(d, str(model_id), run_class="canonical")
        for model_id in cfg["models"]
    ]


def _run_stress_if_budget() -> dict[str, Any]:
    cfg = hd3.load_config()
    reserve_hours = float(cfg["finalization_reserve_hours"])
    d50_seconds = 0.0
    for model_id in cfg["models"]:
        path = hd3._job_dir(50, str(model_id), "canonical") / "validation_metrics.json"
        if path.exists():
            metrics = json.loads(path.read_text(encoding="utf-8"))
            d50_seconds += float(metrics.get("training_seconds", 0.0))
    if d50_seconds <= 0.0:
        return {"status": "skipped_no_50d_timing"}
    projected_seconds = d50_seconds * (float(cfg["stress_steps"]) / float(cfg["canonical_steps"])) * 2.0
    remaining_seconds = max(0.0, (_hours_remaining() - reserve_hours) * 3600.0)
    if projected_seconds > remaining_seconds:
        return {
            "status": "skipped_due_budget",
            "projected_seconds": projected_seconds,
            "remaining_seconds_after_reserve": remaining_seconds,
        }
    rows = hd3.run_stress()
    return {
        "status": "completed",
        "projected_seconds": projected_seconds,
        "rows": rows,
    }


def _finalize_pipeline() -> dict[str, Any]:
    hd3_rows = hd3.run_finalize()
    payload = {
        "status": "completed",
        "diagnostic_only": True,
        "candidate_supported": False,
        "git_commit": git_commit(),
        "combined_config_hash": combined_config_hash(),
        "active_elapsed_seconds": _active_elapsed_seconds(),
        "hd3_rows": len(hd3_rows),
    }
    atomic_json(PIPELINE_DIR / "final_status.json", payload)
    return payload


def real_jobs() -> dict[str, Callable[[], Any]]:
    return {
        "preflight": lambda: {"common": _common_preflight(), "hd2": hd2.run_preflight()},
        "hd2_phase_a": hd2.run_phase_a,
        "hd2_screen": hd2.run_screen,
        "hd2_conditional": _hd2_conditional_placeholder,
        "hd2_select": hd2.run_select,
        "hd2_finalize": hd2.run_finalize,
        "hd3_preflight": hd3.run_preflight,
        "hd3_10d": lambda: _run_hd3_dimension(10),
        "hd3_20d": lambda: _run_hd3_dimension(20),
        "hd3_50d": lambda: _run_hd3_dimension(50),
        "hd3_100d_stress": _run_stress_if_budget,
        "finalize": _finalize_pipeline,
    }


def _phase_subset(phase: str) -> dict[str, Callable[[], Any]]:
    jobs = real_jobs()
    if phase == "all":
        return jobs
    if phase == "preflight":
        return {"preflight": jobs["preflight"]}
    if phase == "hd2":
        names = ["hd2_phase_a", "hd2_screen", "hd2_conditional", "hd2_select", "hd2_finalize"]
        return {name: jobs[name] for name in names}
    if phase == "hd3":
        names = ["hd3_preflight", "hd3_10d", "hd3_20d", "hd3_50d", "hd3_100d_stress"]
        return {name: jobs[name] for name in names}
    if phase == "finalize":
        return {"finalize": jobs["finalize"]}
    raise ValueError(f"unsupported phase: {phase}")


def status_snapshot() -> dict[str, Any]:
    phase_rows: list[dict[str, Any]] = []
    for phase in DEFAULT_PHASES:
        status = _read_phase_status(_phase_path(PIPELINE_RUNTIME_DIR, phase))
        phase_rows.append(status or {"phase": phase, "status": "not_started"})
    return {
        "git_commit": git_commit(),
        "combined_config_hash": combined_config_hash(),
        "active_elapsed_seconds": _active_elapsed_seconds(),
        "hours_remaining": _hours_remaining(),
        "phases": phase_rows,
        "hd2": hd2.status_snapshot(),
        "hd3": hd3.status_snapshot(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["all", "preflight", "hd2", "hd3", "finalize", "status"],
        required=True,
    )
    args = parser.parse_args(argv)
    if args.phase == "status":
        print(json.dumps(status_snapshot(), indent=2, sort_keys=True, default=str))
        return 0
    PIPELINE_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    jobs = _phase_subset(args.phase)
    result = run_state_machine(
        PIPELINE_RUNTIME_DIR,
        jobs,
        implementation_sha=git_commit(),
        config_hash=combined_config_hash(),
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
