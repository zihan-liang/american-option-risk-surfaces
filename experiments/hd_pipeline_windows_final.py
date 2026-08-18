"""Final one-command Windows orchestrator for HD2 -> HD3.

Normal use is ``--phase all``.  Re-running the same command is the supported
resume path after Ctrl+C, shell closure, CUDA failure, or reboot.  Scientific
jobs own 250-step checkpoints; this orchestrator owns phase provenance,
conditional-method activation, the single-GPU parallelism benchmark, the
18-active-hour resource policy, and compact final evidence generation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import hd2_conditional_methods as hd2c
import hd2_hvi_objective_stabilization as hd2
import hd3_classical_vs_structured_high_dimensional_pinn as hd3
import hd3_parallelism_benchmark as hd3bench
from hd_pipeline.runtime import (
    atomic_json,
    load_checkpoint_verified,
    read_jsonl_recover,
    reported_completed_step,
    sha256_file,
)
from hd_pipeline_windows import run_state_machine, should_enable_parallel
from hd1_windows_high_dimensional_pinn_48h import environment_manifest, git_commit

PIPELINE_DIR = ROOT / "results" / "hd_pipeline"
RUNTIME_DIR = PIPELINE_DIR / "runtime"
LOG_DIR = RUNTIME_DIR / "logs"
STATE_ROOT = RUNTIME_DIR
HD3_BENCHMARK_PATH = hd3.RESULT_DIR / "parallelism_benchmark.json"
HD2_CONFIG = ROOT / "configs" / "hd2_hvi_objective_stabilization.json"
HD3_CONFIG = ROOT / "configs" / "hd3_classical_vs_structured_high_dimensional_pinn.json"

PHASE_ORDER = [
    "preflight",
    "hd2_phase_a",
    "hd2_screen",
    "hd2_conditional",
    "hd2_select",
    "hd2_finalize",
    "hd3_preflight",
    "hd3_parallelism_benchmark",
    "hd3_10d",
    "hd3_20d",
    "hd3_50d",
    "hd3_100d_stress",
    "finalize",
]


def combined_config_hash() -> str:
    digest = hashlib.sha256()
    for path in (HD2_CONFIG, HD3_CONFIG):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def project_canonical_seconds_final(
    benchmark: dict[str, Any],
    *,
    canonical_steps: int,
    parallel_enabled: bool,
) -> float:
    smoke_steps = int(benchmark["smoke_steps"])
    seq = benchmark["sequential"]
    fc10 = float(seq["d10_FC_HVI"]["wall_seconds"])
    ds10 = float(seq["d10_DEEPSETS_HVI"]["wall_seconds"])
    fc50 = float(seq["d50_FC_HVI"]["wall_seconds"])
    ds50 = float(seq["d50_DEEPSETS_HVI"]["wall_seconds"])
    fc20 = fc10 + 0.25 * (fc50 - fc10)
    ds20 = ds10 + 0.25 * (ds50 - ds10)
    if parallel_enabled:
        comparison = benchmark["comparison_sequential_d10"]
        comparison_sum = max(
            float(comparison["FC_HVI"]["wall_seconds"])
            + float(comparison["DEEPSETS_HVI"]["wall_seconds"]),
            1e-12,
        )
        concurrent10 = float(benchmark["concurrent_d10"]["wall_seconds"])
        ratio = concurrent10 / comparison_sum
        smoke_total = concurrent10 + (fc20 + ds20) * ratio + fc50 + ds50
    else:
        smoke_total = fc10 + ds10 + fc20 + ds20 + fc50 + ds50
    return smoke_total * float(canonical_steps) / float(smoke_steps)


def relobralo_budget_safe(
    *,
    hours_remaining: float,
    projected_relobralo_seconds: float,
    canonical_reserve_hours: float,
    finalization_reserve_hours: float,
) -> bool:
    required = (
        float(projected_relobralo_seconds) / 3600.0
        + float(canonical_reserve_hours)
        + float(finalization_reserve_hours)
    )
    return float(hours_remaining) > required


def stress_budget_decision(
    *,
    d50_pair_seconds: float,
    canonical_steps: int,
    stress_steps: int,
    hours_remaining: float,
    finalization_reserve_hours: float,
) -> dict[str, Any]:
    projected = (
        float(d50_pair_seconds)
        * float(stress_steps)
        / float(canonical_steps)
        * 2.0
    )
    available = max(
        0.0,
        (float(hours_remaining) - float(finalization_reserve_hours)) * 3600.0,
    )
    return {
        "run_stress": bool(projected <= available),
        "projected_stress_seconds": projected,
        "available_seconds_after_finalization_reserve": available,
    }


def _phase_statuses() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    directory = STATE_ROOT / "phase_status"
    for phase in PHASE_ORDER:
        path = directory / f"{phase}.json"
        if not path.exists():
            rows.append({"phase": phase, "status": "not_started"})
            continue
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            rows.append({"phase": phase, "status": "corrupt"})
    return rows


def _active_elapsed_seconds() -> float:
    total = 0.0
    for row in _phase_statuses():
        if row.get("status") in {"completed", "failed", "interrupted"}:
            total += float(row.get("elapsed_seconds", 0.0))
    return total


def _hours_remaining() -> float:
    return max(
        0.0,
        float(hd3.load_config()["pipeline_wall_clock_hours"])
        - _active_elapsed_seconds() / 3600.0,
    )


def _tracked_source_dirty() -> list[str]:
    raw = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    )
    dirty: list[str] = []
    for line in raw.splitlines():
        path = line[3:].strip().replace("\\", "/")
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path and not path.startswith("results/"):
            dirty.append(path)
    return dirty


def _run_focused_tests() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "experiments")])
    env["MPLBACKEND"] = "Agg"
    tests = [
        "tests/test_hd1_high_dimensional_pinn.py",
        "tests/test_hd2_hvi_objective_stabilization.py",
        "tests/test_hd2_conditional_methods.py",
        "tests/test_hd3_high_dimensional_pinn.py",
        "tests/test_hd3_parallelism_benchmark.py",
        "tests/test_hd_pipeline_resume.py",
        "tests/test_hd_pipeline_review_regressions.py",
        "tests/test_hd_objective_optimizers.py",
        "tests/test_hd_pipeline_final_contract.py",
    ]
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests],
        cwd=ROOT,
        env=env,
        check=True,
    )


def _common_preflight() -> dict[str, Any]:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    if branch != "exp/hd2-hvi-objective-stabilization":
        raise RuntimeError(f"wrong_branch:{branch}")
    dirty = _tracked_source_dirty()
    if dirty:
        raise RuntimeError("tracked_source_dirty:" + ",".join(dirty))
    free = shutil.disk_usage(ROOT).free
    if free < 5 * 1024**3:
        raise RuntimeError("disk_space_blocked:need_5_gib_free")
    try:
        nvidia_smi = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
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
        "disk_free_bytes": free,
        "environment": manifest,
    }
    atomic_json(PIPELINE_DIR / "environment_manifest.json", payload)
    atomic_json(
        PIPELINE_DIR / "config_manifest.json",
        {
            "git_commit": git_commit(),
            "combined_config_hash": combined_config_hash(),
            "hd2_config_sha256": sha256_file(HD2_CONFIG),
            "hd3_config_sha256": sha256_file(HD3_CONFIG),
        },
    )
    return payload


def _estimate_relobralo_seconds() -> float:
    total = 0.0
    for d in (2, 5):
        path = hd2._job_dir(d, "HVI-HD1-anchor") / "validation_metrics.json"
        if path.exists():
            total += float(json.loads(path.read_text(encoding="utf-8")).get("training_seconds", 0.0))
    return total * 1.25 if total > 0.0 else float("inf")


def _run_hd2_conditionals() -> dict[str, Any]:
    cfg = hd3.load_config()
    safe = relobralo_budget_safe(
        hours_remaining=_hours_remaining(),
        projected_relobralo_seconds=_estimate_relobralo_seconds(),
        canonical_reserve_hours=float(cfg["canonical_time_allocation_hours"]),
        finalization_reserve_hours=float(cfg["finalization_reserve_hours"]),
    )
    return hd2c.run_triggered_conditionals(relobralo_budget_safe=safe)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    temp.replace(path)


def _finalize_hd2() -> list[dict[str, Any]]:
    cfg = hd2.load_config()
    if not hd2.SELECTION_PATH.exists():
        raise RuntimeError("selection_required_before_hd2_test")
    decision = json.loads(hd2.SELECTION_PATH.read_text(encoding="utf-8"))
    if decision.get("implementation_sha") != git_commit() or decision.get("config_hash") != hd2.config_hash():
        raise RuntimeError("stale_selection_provenance")
    selected = str(decision["resolved_objective"]["candidate"])
    display_candidates: list[str] = []
    for candidate in ("SUP-HD2", "HVI-HD1-anchor", selected):
        if candidate not in display_candidates:
            display_candidates.append(candidate)
    rows: list[dict[str, Any]] = []
    for candidate in display_candidates:
        for d in cfg["dimensions"]:
            actual = (
                "HVI-HD1-anchor"
                if candidate == "HVI-HD1-anchor-8probe" and int(d) == 2
                else candidate
            )
            model, checkpoint = hd2._load_completed_model(int(d), actual, cfg)
            row = hd2.evaluate_model_metrics(
                d=int(d),
                candidate=candidate,
                model=model,
                cfg=cfg,
                oracle=hd2.load_oracle(int(d)),
                split="test",
                n=int(cfg["test_points"]),
                training_seconds=float(checkpoint.get("elapsed_training_seconds", 0.0)),
                batch_size=int(checkpoint.get("batch_size", cfg["initial_batch_size"])),
            )
            if actual != candidate:
                row["reused_anchor_d2"] = True
            row["selection_status"] = decision["selection_status"]
            rows.append(row)
    _write_csv(hd2.RESULT_DIR / "test_metrics.csv", rows)
    atomic_json(
        hd2.RESULT_DIR / "final_status.json",
        {
            "status": "completed",
            "diagnostic_only": True,
            "candidate_supported": False,
            "selection_status": decision["selection_status"],
            "resolved_objective": decision["resolved_objective"],
            "rows": len(rows),
        },
    )
    return rows


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "experiments")])
    env["MPLBACKEND"] = "Agg"
    return env


def _run_smoke_subprocess(model_id: str, d: int, batch_size: int, tag: str) -> dict[str, Any]:
    smoke_dir = hd3.RUNTIME_DIR / "parallelism_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    output = smoke_dir / f"{tag}.json"
    log = smoke_dir / f"{tag}.log"
    command = [
        sys.executable,
        str(ROOT / "experiments" / "hd3_parallelism_benchmark.py"),
        "--model",
        model_id,
        "--dimension",
        str(d),
        "--steps",
        str(hd3.load_config()["parallelism_smoke_steps"]),
        "--batch-size",
        str(batch_size),
        "--output",
        str(output),
    ]
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, env=_child_env(), stdout=handle, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - started
    if result.returncode != 0 or not output.exists():
        raise RuntimeError(f"parallelism_smoke_failed:{tag}:exit={result.returncode}")
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["training_wall_seconds"] = payload["wall_seconds"]
    payload["wall_seconds"] = wall
    return payload


def _query_child_memory(pids: set[int]) -> float:
    try:
        text = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        return float("inf")
    return hd3bench.parse_compute_memory_gib(text, pids)


def _concurrent_smoke_d10(batch_size: int) -> dict[str, Any]:
    smoke_dir = hd3.RUNTIME_DIR / "parallelism_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen[Any]] = []
    handles = []
    outputs: dict[str, Path] = {}
    started = time.perf_counter()
    try:
        for model_id in ("FC_HVI", "DEEPSETS_HVI"):
            output = smoke_dir / f"concurrent_d10_{model_id}.json"
            log = smoke_dir / f"concurrent_d10_{model_id}.log"
            outputs[model_id] = output
            handle = log.open("w", encoding="utf-8")
            handles.append(handle)
            command = [
                sys.executable,
                str(ROOT / "experiments" / "hd3_parallelism_benchmark.py"),
                "--model",
                model_id,
                "--dimension",
                "10",
                "--steps",
                str(hd3.load_config()["parallelism_smoke_steps"]),
                "--batch-size",
                str(batch_size),
                "--output",
                str(output),
            ]
            processes.append(
                subprocess.Popen(command, cwd=ROOT, env=_child_env(), stdout=handle, stderr=subprocess.STDOUT)
            )
        peak = 0.0
        pids = {process.pid for process in processes}
        while any(process.poll() is None for process in processes):
            peak = max(peak, _query_child_memory(pids))
            time.sleep(0.25)
        wall = time.perf_counter() - started
        return_codes = [process.returncode for process in processes]
        succeeded = all(code == 0 for code in return_codes) and all(path.exists() for path in outputs.values())
        child_metrics = {
            model: json.loads(path.read_text(encoding="utf-8"))
            for model, path in outputs.items()
            if path.exists()
        }
        return {
            "wall_seconds": wall,
            "peak_total_cuda_gib": peak,
            "jobs_succeeded": succeeded,
            "return_codes": return_codes,
            "child_metrics": child_metrics,
        }
    except KeyboardInterrupt:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for handle in handles:
            handle.close()


def _run_parallelism_benchmark() -> dict[str, Any]:
    cfg = hd3.load_config()
    smoke_steps = int(cfg["parallelism_smoke_steps"])
    sequential_batch = int(cfg["initial_batch_size"])
    concurrent_batch = int(cfg["concurrent_batch_size"])
    sequential = {}
    for d in (10, 50):
        for model_id in ("FC_HVI", "DEEPSETS_HVI"):
            key = f"d{d}_{model_id}"
            sequential[key] = _run_smoke_subprocess(
                model_id, d, sequential_batch, f"sequential_{key}"
            )
    comparison = {
        model_id: _run_smoke_subprocess(
            model_id,
            10,
            concurrent_batch,
            f"comparison_d10_{model_id}",
        )
        for model_id in ("FC_HVI", "DEEPSETS_HVI")
    }
    concurrent = _concurrent_smoke_d10(concurrent_batch)
    enabled = should_enable_parallel(
        fc_sequential_seconds=float(comparison["FC_HVI"]["wall_seconds"]),
        deepsets_sequential_seconds=float(comparison["DEEPSETS_HVI"]["wall_seconds"]),
        concurrent_wall_seconds=float(concurrent["wall_seconds"]),
        concurrent_peak_cuda_gib=float(concurrent["peak_total_cuda_gib"]),
        jobs_succeeded=bool(concurrent["jobs_succeeded"]),
        speed_ratio_max=float(cfg["parallelism_speed_ratio_max"]),
        memory_limit_gib=float(cfg["max_cuda_allocation_gib"]),
    )
    benchmark = {
        "smoke_steps": smoke_steps,
        "sequential_batch_size": sequential_batch,
        "concurrent_batch_size": concurrent_batch,
        "sequential": sequential,
        "comparison_sequential_d10": comparison,
        "concurrent_d10": concurrent,
        "parallel_enabled_10d_20d": enabled,
        "non_evidence": True,
    }
    projected = project_canonical_seconds_final(
        benchmark,
        canonical_steps=int(cfg["canonical_steps"]),
        parallel_enabled=enabled,
    )
    benchmark["projected_canonical_seconds"] = projected
    benchmark["hours_remaining_before_canonical"] = _hours_remaining()
    available = max(
        0.0,
        (_hours_remaining() - float(cfg["finalization_reserve_hours"])) * 3600.0,
    )
    benchmark["canonical_available_seconds_after_finalization_reserve"] = available
    benchmark["resource_blocked"] = bool(projected > available)
    atomic_json(HD3_BENCHMARK_PATH, benchmark)
    if projected > available:
        raise RuntimeError(
            f"resource_blocked:projected_canonical={projected:.1f}s available={available:.1f}s"
        )
    return benchmark


def _parallel_enabled() -> bool:
    if not HD3_BENCHMARK_PATH.exists():
        raise RuntimeError("parallelism_benchmark_required")
    return bool(
        json.loads(HD3_BENCHMARK_PATH.read_text(encoding="utf-8")).get(
            "parallel_enabled_10d_20d", False
        )
    )


def _run_concurrent_canonical(d: int) -> list[dict[str, Any]]:
    processes: list[subprocess.Popen[Any]] = []
    handles = []
    try:
        for model_id in ("FC_HVI", "DEEPSETS_HVI"):
            log_path = LOG_DIR / f"canonical_d{d}_{model_id}.log"
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8")
            handles.append(handle)
            command = [
                sys.executable,
                str(ROOT / "experiments" / "hd3_classical_vs_structured_high_dimensional_pinn.py"),
                "--phase",
                "canonical",
                "--dimension",
                str(d),
                "--model",
                model_id,
                "--execution-mode",
                "concurrent",
            ]
            processes.append(
                subprocess.Popen(command, cwd=ROOT, env=_child_env(), stdout=handle, stderr=subprocess.STDOUT)
            )
        while any(process.poll() is None for process in processes):
            time.sleep(1.0)
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError(
                "concurrent_canonical_failed:"
                + ",".join(str(process.returncode) for process in processes)
            )
    except KeyboardInterrupt:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for handle in handles:
            handle.close()
    return [
        json.loads(
            (hd3._job_dir(d, model_id, "canonical") / "validation_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        for model_id in ("FC_HVI", "DEEPSETS_HVI")
    ]


def _run_hd3_dimension(d: int) -> list[dict[str, Any]]:
    if d in (10, 20) and _parallel_enabled():
        return _run_concurrent_canonical(d)
    return [
        hd3.train_hd3_job(d, model_id, run_class="canonical", execution_mode="sequential")
        for model_id in ("FC_HVI", "DEEPSETS_HVI")
    ]


def _run_stress_if_budget() -> dict[str, Any]:
    cfg = hd3.load_config()
    d50_total = 0.0
    for model_id in cfg["models"]:
        path = hd3._job_dir(50, str(model_id), "canonical") / "validation_metrics.json"
        if path.exists():
            d50_total += float(
                json.loads(path.read_text(encoding="utf-8")).get("training_seconds", 0.0)
            )
    if d50_total <= 0.0:
        return {"status": "skipped_no_50d_timing"}
    decision = stress_budget_decision(
        d50_pair_seconds=d50_total,
        canonical_steps=int(cfg["canonical_steps"]),
        stress_steps=int(cfg["stress_steps"]),
        hours_remaining=_hours_remaining(),
        finalization_reserve_hours=float(cfg["finalization_reserve_hours"]),
    )
    if not decision["run_stress"]:
        return {"status": "skipped_due_budget", **decision}
    rows = hd3.run_stress()
    return {"status": "completed", **decision, "rows": rows}


def _read_checkpoint_step(path: Path, *, job_id: str, cfg_hash: str) -> int | None:
    if not path.exists():
        return None
    try:
        checkpoint = load_checkpoint_verified(
            path,
            implementation_sha=git_commit(),
            config_hash=cfg_hash,
            job_id=job_id,
        )
    except Exception:
        return None
    return int(checkpoint.get("completed_step", 0))


def _hydrate_hd2_status() -> dict[str, Any]:
    cfg = hd2.load_config()
    snapshot = hd2.status_snapshot()
    known = {(int(row["dimension"]), str(row["candidate"])) for row in snapshot["jobs"]}
    for candidate in hd2c.CONDITIONAL_CANDIDATES:
        dims = [5] if candidate == "HVI-HD1-anchor-8probe" else [2, 5]
        for d in dims:
            if (d, candidate) in known:
                continue
            status_path = hd2._status_path(d, candidate)
            snapshot["jobs"].append(
                json.loads(status_path.read_text(encoding="utf-8"))
                if status_path.exists()
                else {"dimension": d, "candidate": candidate, "status": "not_started"}
            )
    for row in snapshot["jobs"]:
        d, candidate = int(row["dimension"]), str(row["candidate"])
        step = _read_checkpoint_step(
            hd2._checkpoint_path(d, candidate),
            job_id=hd2._job_id(d, candidate),
            cfg_hash=hd2.config_hash(),
        )
        row["durable_checkpoint_step"] = step
        row["completed_step"] = reported_completed_step(
            status_step=row.get("completed_step"), checkpoint_step=step
        )
        row["resume_available"] = step is not None and row.get("status") != "completed"
    return snapshot


def _hydrate_hd3_status() -> dict[str, Any]:
    snapshot = hd3.status_snapshot()
    for row in snapshot["jobs"]:
        d, model_id, run_class = (
            int(row["dimension"]),
            str(row["model_id"]),
            str(row["run_class"]),
        )
        step = _read_checkpoint_step(
            hd3._checkpoint_path(d, model_id, run_class),
            job_id=hd3._job_id(d, model_id, run_class),
            cfg_hash=hd3.config_hash(),
        )
        row["durable_checkpoint_step"] = step
        row["completed_step"] = reported_completed_step(
            status_step=row.get("completed_step"), checkpoint_step=step
        )
        row["resume_available"] = step is not None and row.get("status") != "completed"
    return snapshot


def _aggregate_jsonl(paths: list[Path], output: Path, dedupe: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl_recover(path):
            key = tuple(row.get(field) for field in dedupe)
            seen[key] = row
    rows = sorted(seen.values(), key=lambda row: tuple(str(row.get(field, "")) for field in dedupe))
    _write_csv(output, rows)
    return rows


def _compact_results() -> dict[str, Any]:
    hd2_runtime_dirs = [path for path in hd2.RUNTIME_DIR.glob("d*_") if path.is_dir()]
    # glob above is intentionally supplemented by all actual job directories.
    hd2_runtime_dirs = [path for path in hd2.RUNTIME_DIR.iterdir() if path.is_dir()] if hd2.RUNTIME_DIR.exists() else []
    hd3_runtime_dirs = [
        path
        for path in hd3.RUNTIME_DIR.iterdir()
        if path.is_dir() and path.name != "parallelism_smoke"
    ] if hd3.RUNTIME_DIR.exists() else []
    _aggregate_jsonl(
        [path / "training_history.jsonl" for path in hd2_runtime_dirs],
        hd2.RESULT_DIR / "training_history.csv",
        ("candidate", "dimension", "step"),
    )
    _aggregate_jsonl(
        [path / "validation_history.jsonl" for path in hd2_runtime_dirs],
        hd2.RESULT_DIR / "validation_history.csv",
        ("candidate", "dimension", "step"),
    )
    _aggregate_jsonl(
        [path / "gradient_diagnostics.jsonl" for path in hd2_runtime_dirs],
        hd2.RESULT_DIR / "gradient_diagnostics.csv",
        ("candidate", "dimension", "step"),
    )
    _aggregate_jsonl(
        [path / "layerwise_gradient_diagnostics.jsonl" for path in hd2_runtime_dirs],
        hd2.RESULT_DIR / "layerwise_gradient_diagnostics.csv",
        ("candidate", "dimension", "step", "group"),
    )
    _aggregate_jsonl(
        [path / "hutchinson_gradient_validation.jsonl" for path in hd2_runtime_dirs],
        hd2.RESULT_DIR / "hutchinson_gradient_validation.csv",
        ("dimension", "step", "probe_count"),
    )
    _aggregate_jsonl(
        [path / "training_history.jsonl" for path in hd3_runtime_dirs],
        hd3.RESULT_DIR / "training_history.csv",
        ("model_id", "dimension", "run_class", "step"),
    )
    _aggregate_jsonl(
        [path / "validation_history.jsonl" for path in hd3_runtime_dirs],
        hd3.RESULT_DIR / "validation_history.csv",
        ("model_id", "dimension", "run_class", "step"),
    )
    if (hd3.RESULT_DIR / "run_metrics.csv").exists():
        shutil.copyfile(
            hd3.RESULT_DIR / "run_metrics.csv",
            hd3.RESULT_DIR / "dimension_summary.csv",
        )
    if HD3_BENCHMARK_PATH.exists():
        benchmark = json.loads(HD3_BENCHMARK_PATH.read_text(encoding="utf-8"))
        bench_rows: list[dict[str, Any]] = []
        for key, row in benchmark.get("sequential", {}).items():
            bench_rows.append({"mode": "sequential", "job": key, **row})
        for key, row in benchmark.get("comparison_sequential_d10", {}).items():
            bench_rows.append({"mode": "comparison_sequential_d10", "job": key, **row})
        concurrent = benchmark.get("concurrent_d10", {})
        bench_rows.append(
            {
                "mode": "concurrent_d10",
                "job": "pair",
                "wall_seconds": concurrent.get("wall_seconds"),
                "peak_total_cuda_gib": concurrent.get("peak_total_cuda_gib"),
                "jobs_succeeded": concurrent.get("jobs_succeeded"),
                "parallel_enabled": benchmark.get("parallel_enabled_10d_20d"),
            }
        )
        _write_csv(hd3.RESULT_DIR / "parallelism_benchmark.csv", bench_rows)

    ledger: list[dict[str, Any]] = []
    for row in _phase_statuses():
        ledger.append(
            {
                "source": "pipeline_phase",
                "phase": row.get("phase"),
                "status": row.get("status"),
                "elapsed_seconds": row.get("elapsed_seconds"),
                "implementation_sha": row.get("implementation_sha"),
                "config_hash": row.get("config_hash"),
            }
        )
    for row in _hydrate_hd2_status()["jobs"]:
        ledger.append({"source": "hd2_job", **row})
    for row in _hydrate_hd3_status()["jobs"]:
        ledger.append({"source": "hd3_job", **row})
    _write_csv(PIPELINE_DIR / "run_ledger.csv", ledger)

    compact_files: list[Path] = []
    for directory in (hd2.RESULT_DIR, hd3.RESULT_DIR, PIPELINE_DIR):
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if path.is_file() and path.name not in {"artifact_hashes.json", ".gitignore"}:
                compact_files.append(path)
    hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
        for path in sorted(compact_files)
    }
    atomic_json(PIPELINE_DIR / "artifact_hashes.json", hashes)
    for relative, expected in hashes.items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"artifact_hash_verification_failed:{relative}")
    return {"compact_files": len(compact_files), "hashes": len(hashes)}


def _finalize_pipeline() -> dict[str, Any]:
    hd3_rows = hd3.run_finalize()
    compact = _compact_results()
    payload = {
        "status": "completed",
        "diagnostic_only": True,
        "candidate_supported": False,
        "git_commit": git_commit(),
        "combined_config_hash": combined_config_hash(),
        "active_elapsed_seconds": _active_elapsed_seconds(),
        "hd3_rows": len(hd3_rows),
        **compact,
    }
    atomic_json(PIPELINE_DIR / "final_status.json", payload)
    return payload


def real_jobs() -> dict[str, Callable[[], Any]]:
    return {
        "preflight": lambda: {"common": _common_preflight(), "hd2": hd2.run_preflight()},
        "hd2_phase_a": hd2.run_phase_a,
        "hd2_screen": hd2.run_screen,
        "hd2_conditional": _run_hd2_conditionals,
        "hd2_select": hd2c.run_select_with_conditionals,
        "hd2_finalize": _finalize_hd2,
        "hd3_preflight": hd3.run_preflight,
        "hd3_parallelism_benchmark": _run_parallelism_benchmark,
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
        names = [
            "hd2_phase_a",
            "hd2_screen",
            "hd2_conditional",
            "hd2_select",
            "hd2_finalize",
        ]
        return {name: jobs[name] for name in names}
    if phase == "hd3":
        names = [
            "hd3_preflight",
            "hd3_parallelism_benchmark",
            "hd3_10d",
            "hd3_20d",
            "hd3_50d",
            "hd3_100d_stress",
        ]
        return {name: jobs[name] for name in names}
    if phase == "finalize":
        return {"finalize": jobs["finalize"]}
    raise ValueError(f"unsupported phase:{phase}")


def status_snapshot() -> dict[str, Any]:
    return {
        "git_commit": git_commit(),
        "combined_config_hash": combined_config_hash(),
        "active_elapsed_seconds": _active_elapsed_seconds(),
        "hours_remaining": _hours_remaining(),
        "parallelism_benchmark": (
            json.loads(HD3_BENCHMARK_PATH.read_text(encoding="utf-8"))
            if HD3_BENCHMARK_PATH.exists()
            else None
        ),
        "phases": _phase_statuses(),
        "hd2": _hydrate_hd2_status(),
        "hd3": _hydrate_hd3_status(),
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
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    result = run_state_machine(
        STATE_ROOT,
        _phase_subset(args.phase),
        implementation_sha=git_commit(),
        config_hash=combined_config_hash(),
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
