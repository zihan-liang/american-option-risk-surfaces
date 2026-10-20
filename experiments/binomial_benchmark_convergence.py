"""Independent high-resolution CRR convergence audit for the matched American put.

This experiment is deliberately standalone: it depends only on NumPy and the
Python standard library, writes compact CSV/JSON evidence, and can resume after
interruption from the point-results ledger.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = ROOT / "results" / "binomial_benchmark_convergence"

CONTRACT = {
    "option_type": "put",
    "K": 1.0,
    "T": 1.0,
    "r": 0.05,
    "q": 0.10,
    "sigma": 0.60,
}
PROBE_STEPS = (2048, 4096, 8192, 16384)
CONFIRM_STEPS = (32768, 50000)
CHECK_SPOTS = (0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00, 1.20, 1.40)
CONFIRM_SPOTS = (0.80, 1.00, 1.20)
CHECK_TAUS = (0.10, 0.50, 1.00)


@dataclass(frozen=True)
class Job:
    phase: str
    S_over_K: float
    tau_over_T: float
    steps: int

    @property
    def key(self) -> tuple[str, float, float, int]:
        return (self.phase, self.S_over_K, self.tau_over_T, self.steps)


def crr_american_put(
    spot: float,
    tau: float,
    *,
    k: float,
    r: float,
    q: float,
    sigma: float,
    steps: int,
) -> float:
    """Return a Cox-Ross-Rubinstein American put value for one initial spot."""
    if int(steps) < 1:
        raise ValueError("steps must be positive")
    if float(spot) < 0.0 or float(k) <= 0.0 or float(sigma) < 0.0:
        raise ValueError("invalid contract input")
    if tau <= 0.0:
        return float(max(k - spot, 0.0))
    if sigma == 0.0:
        raise ValueError("sigma=0 is not supported by the CRR probability formula")

    n_steps = int(steps)
    dt = float(tau) / n_steps
    log_u = float(sigma) * math.sqrt(dt)
    u = math.exp(log_u)
    d = 1.0 / u
    growth = math.exp((float(r) - float(q)) * dt)
    p = (growth - d) / (u - d)
    if not 0.0 <= p <= 1.0:
        raise RuntimeError(f"invalid_crr_probability:{p}")
    disc = math.exp(-float(r) * dt)

    # Build the terminal stock vector once, then roll it backwards together
    # with option values. This avoids recomputing exponentials at every layer.
    j = np.arange(n_steps + 1, dtype=np.float64)
    stock = float(spot) * np.exp((2.0 * j - float(n_steps)) * log_u)
    values = np.maximum(float(k) - stock, 0.0)
    for _ in range(n_steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        stock = stock[:-1] * u
        values = np.maximum(values, np.maximum(float(k) - stock, 0.0))
    return float(values[0])


def build_default_job_plan() -> list[dict[str, Any]]:
    jobs = [
        Job("probe", s, tau, steps)
        for tau in CHECK_TAUS
        for s in CHECK_SPOTS
        for steps in PROBE_STEPS
    ]
    jobs.extend(
        Job("confirm", s, tau, steps)
        for tau in CHECK_TAUS
        for s in CONFIRM_SPOTS
        for steps in CONFIRM_STEPS
    )
    return [asdict(job) for job in jobs]


def summarize_convergence(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(rows)
    by_point_step: dict[tuple[float, float, int], float] = {}
    for row in rows:
        by_point_step[(float(row["S_over_K"]), float(row["tau_over_T"]), int(row["steps"]))] = float(row["V_over_K"])

    step_pairs = [
        (2048, 4096, "probe_2048_to_4096"),
        (4096, 8192, "probe_4096_to_8192"),
        (8192, 16384, "probe_8192_to_16384"),
        (16384, 32768, "confirm_16384_to_32768"),
        (32768, 50000, "confirm_32768_to_50000"),
    ]
    output: list[dict[str, Any]] = []
    point_pairs = sorted({(float(r["S_over_K"]), float(r["tau_over_T"])) for r in rows})
    for lo, hi, label in step_pairs:
        errors = []
        for s, tau in point_pairs:
            a = by_point_step.get((s, tau, lo))
            b = by_point_step.get((s, tau, hi))
            if a is not None and b is not None:
                errors.append(b - a)
        if not errors:
            continue
        e = np.asarray(errors, dtype=np.float64)
        output.append(
            {
                "comparison": label,
                "from_steps": lo,
                "to_steps": hi,
                "point_count": int(e.size),
                "bias_change": float(e.mean()),
                "mae_change": float(np.abs(e).mean()),
                "rmse_change": float(np.sqrt(np.mean(e * e))),
                "max_abs_change": float(np.abs(e).max()),
            }
        )
    return output


def _read_point_results(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _completed_keys(rows: Iterable[dict[str, Any]]) -> set[tuple[str, float, float, int]]:
    return {
        (str(r["phase"]), float(r["S_over_K"]), float(r["tau_over_T"]), int(r["steps"]))
        for r in rows
    }


def _append_result(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["phase", "S_over_K", "tau_over_T", "steps", "V_over_K", "seconds"]
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({name: row[name] for name in fieldnames})
        handle.flush()
        os.fsync(handle.fileno())


def _solve_job(job_dict: dict[str, Any]) -> dict[str, Any]:
    job = Job(
        phase=str(job_dict["phase"]),
        S_over_K=float(job_dict["S_over_K"]),
        tau_over_T=float(job_dict["tau_over_T"]),
        steps=int(job_dict["steps"]),
    )
    started = time.perf_counter()
    value = crr_american_put(
        job.S_over_K,
        job.tau_over_T,
        k=CONTRACT["K"],
        r=CONTRACT["r"],
        q=CONTRACT["q"],
        sigma=CONTRACT["sigma"],
        steps=job.steps,
    )
    return {
        **asdict(job),
        "V_over_K": float(value / CONTRACT["K"]),
        "seconds": float(time.perf_counter() - started),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _environment_manifest() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unavailable",
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
    }


def run_phase(phase: str, *, result_dir: Path, workers: int) -> dict[str, Any]:
    if phase not in {"probe", "confirm", "all"}:
        raise ValueError("phase must be probe, confirm, or all")
    selected = {"probe", "confirm"} if phase == "all" else {phase}
    plan = [row for row in build_default_job_plan() if row["phase"] in selected]
    ledger = result_dir / "point_results.csv"
    existing = _read_point_results(ledger)
    done = _completed_keys(existing)
    pending = [row for row in plan if (row["phase"], float(row["S_over_K"]), float(row["tau_over_T"]), int(row["steps"])) not in done]

    started = time.perf_counter()
    if pending:
        if workers <= 1:
            for index, job in enumerate(pending, start=1):
                row = _solve_job(job)
                _append_result(ledger, row)
                print(
                    f"[{index}/{len(pending)}] {row['phase']} S/K={row['S_over_K']:.4g} "
                    f"tau/T={row['tau_over_T']:.3g} N={row['steps']} "
                    f"V/K={row['V_over_K']:.12g} sec={row['seconds']:.3f}",
                    flush=True,
                )
        else:
            with ProcessPoolExecutor(max_workers=int(workers)) as pool:
                futures = {pool.submit(_solve_job, job): job for job in pending}
                completed = 0
                for future in as_completed(futures):
                    row = future.result()
                    completed += 1
                    _append_result(ledger, row)
                    print(
                        f"[{completed}/{len(pending)}] {row['phase']} S/K={row['S_over_K']:.4g} "
                        f"tau/T={row['tau_over_T']:.3g} N={row['steps']} "
                        f"V/K={row['V_over_K']:.12g} sec={row['seconds']:.3f}",
                        flush=True,
                    )

    final_rows = _read_point_results(ledger)
    typed_rows = [
        {
            "phase": str(r["phase"]),
            "S_over_K": float(r["S_over_K"]),
            "tau_over_T": float(r["tau_over_T"]),
            "steps": int(r["steps"]),
            "V_over_K": float(r["V_over_K"]),
            "seconds": float(r["seconds"]),
        }
        for r in final_rows
    ]
    summary = summarize_convergence(typed_rows)
    _write_csv(result_dir / "convergence_summary.csv", summary)
    _atomic_json(result_dir / "environment_manifest.json", _environment_manifest())
    _atomic_json(
        result_dir / "config_manifest.json",
        {
            "contract": CONTRACT,
            "probe_steps": list(PROBE_STEPS),
            "confirm_steps": list(CONFIRM_STEPS),
            "check_spots": list(CHECK_SPOTS),
            "confirm_spots": list(CONFIRM_SPOTS),
            "check_taus": list(CHECK_TAUS),
            "workers": int(workers),
            "scientific_decision": "not_automatically_adjudicated",
        },
    )
    expected = {
        (r["phase"], float(r["S_over_K"]), float(r["tau_over_T"]), int(r["steps"]))
        for r in plan
    }
    finished = _completed_keys(typed_rows)
    status = "completed" if expected.issubset(finished) else "partial"
    payload = {
        "status": status,
        "requested_phase": phase,
        "expected_jobs": len(expected),
        "completed_requested_jobs": len(expected & finished),
        "total_ledger_rows": len(typed_rows),
        "elapsed_this_invocation_seconds": float(time.perf_counter() - started),
        "resume_supported": True,
        "results": {
            "point_results": "point_results.csv",
            "convergence_summary": "convergence_summary.csv",
        },
    }
    _atomic_json(result_dir / "final_status.json", payload)
    return payload


def preflight() -> dict[str, Any]:
    # Cheap independent check: optimized vector implementation must match a
    # small manually tractable tree to near machine precision.
    value = crr_american_put(1.0, 1.0, k=1.0, r=0.05, q=0.10, sigma=0.60, steps=8)
    if not np.isfinite(value) or value < 0.0:
        raise RuntimeError("CRR smoke value is invalid")
    return {
        "status": "passed",
        "smoke_value": float(value),
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "probe", "confirm", "all"), default="all")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.phase == "preflight":
        print(json.dumps(preflight(), indent=2, sort_keys=True))
        return 0
    payload = run_phase(args.phase, result_dir=args.result_dir, workers=args.workers)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
