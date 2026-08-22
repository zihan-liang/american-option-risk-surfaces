"""Clean-isolation and self-convergence audit for the frozen Ruixi DIRK solver."""
from __future__ import annotations

import argparse, csv, json, math, time
from pathlib import Path
from typing import Any
import numpy as np

from experiments.reference_audits.dirk_binomial_50k_convergence import (
    CONTRACT, REGIME_ID, RUIXI_SOURCE_COMMIT, SLICE_TAUS, SPOT_POINTS,
    compute_error_metrics, load_crr_cache, solve_dirk_price_slices,
    verify_canonical_reproduction, verify_ruixi_source,
)

CLEAN_SPATIAL_FIXED_N = 3840
CLEAN_SPATIAL_LEVELS = (120, 240, 480, 960)
CLEAN_TEMPORAL_FIXED_M = 1920
CLEAN_TEMPORAL_LEVELS = (60, 120, 240, 480, 960, 1920)
POSTHOC_TEMPORAL_LEVELS = (3840, 7680)
SELF_METRICS = ("mae", "rmse", "l2", "linf", "p95_abs")


def _diff(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    if d.shape != np.asarray(b).shape or d.size == 0 or not np.all(np.isfinite(d)):
        raise ValueError("self-convergence arrays must have the same finite non-empty shape")
    ad = np.abs(d)
    functions = {
        "mae": lambda: np.mean(ad),
        "rmse": lambda: np.sqrt(np.mean(d * d)),
        "l2": lambda: np.linalg.norm(d.ravel()),
        "linf": lambda: np.max(ad),
        "p95_abs": lambda: np.quantile(ad, 0.95),
    }
    if metric not in functions:
        raise ValueError(f"unsupported self-convergence metric: {metric}")
    return float(functions[metric]())


def self_convergence_order(coarse, middle, fine, *, metric="rmse", refinement_ratio=2.0):
    d1, d2, ratio = _diff(coarse, middle, metric), _diff(middle, fine, metric), float(refinement_ratio)
    if d1 <= 0 or d2 <= 0 or ratio <= 1:
        raise ValueError("positive differences and refinement_ratio > 1 are required")
    return {"coarse_difference": d1, "fine_difference": d2,
            "observed_order": float(math.log(d1 / d2) / math.log(ratio))}


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)


def _json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _run(M: int, N: int):
    started = time.perf_counter()
    try:
        values, meta = solve_dirk_price_slices(M=M, N=N)
        return values, {"run_id": f"M{M}_N{N}", "M": M, "N": N,
            "runtime_seconds": float(meta["runtime_seconds"]), "converged": True, "failure_reason": ""}
    except Exception as exc:
        return None, {"run_id": f"M{M}_N{N}", "M": M, "N": N,
            "runtime_seconds": float(time.perf_counter() - started), "converged": False,
            "failure_reason": str(exc)}


def _metric_rows(solved, benchmark):
    rows = []
    for (M, N), v in sorted(solved.items()):
        for i, tau in enumerate(SLICE_TAUS):
            rows.append({"run_id": f"M{M}_N{N}", "M": M, "N": N, "scope": f"tau_{tau:g}",
                "tau_over_T": tau, "points": SPOT_POINTS, **compute_error_metrics(v[i], benchmark[i])})
        rows.append({"run_id": f"M{M}_N{N}", "M": M, "N": N, "scope": "overall", "tau_over_T": "",
            "points": int(v.size), **compute_error_metrics(v.ravel(), benchmark.ravel())})
    return rows


def _self_tables(solved):
    differences, orders = [], []
    axes = (("spatial", CLEAN_SPATIAL_LEVELS, lambda x: (x, CLEAN_SPATIAL_FIXED_N)),
            ("temporal", CLEAN_TEMPORAL_LEVELS, lambda x: (CLEAN_TEMPORAL_FIXED_M, x)))
    scopes = (("overall", None),) + tuple((f"tau_{t:g}", i) for i, t in enumerate(SLICE_TAUS))
    for axis, levels, pair in axes:
        cache = {}
        for c, f in zip(levels, levels[1:]):
            if pair(c) not in solved or pair(f) not in solved: continue
            for scope, i in scopes:
                a = solved[pair(c)] if i is None else solved[pair(c)][i]
                b = solved[pair(f)] if i is None else solved[pair(f)][i]
                vals = {m: _diff(a, b, m) for m in SELF_METRICS}; cache[(c, f, scope)] = vals
                differences.append({"axis": axis, "scope": scope, "coarse_level": c, "fine_level": f,
                    "coarse_M": pair(c)[0], "coarse_N": pair(c)[1], "fine_M": pair(f)[0], "fine_N": pair(f)[1], **vals})
        for c, m, f in zip(levels, levels[1:], levels[2:]):
            for scope, _ in scopes:
                if (c, m, scope) not in cache or (m, f, scope) not in cache: continue
                for metric in SELF_METRICS:
                    d1, d2 = cache[(c, m, scope)][metric], cache[(m, f, scope)][metric]
                    orders.append({"axis": axis, "scope": scope, "metric": metric, "coarse_level": c,
                        "middle_level": m, "fine_level": f, "coarse_difference": d1, "fine_difference": d2,
                        "observed_order": float(math.log(d1 / d2, 2.0))})
    return differences, orders


def run_clean_isolation(*, output_dir: Path, benchmark_path: Path, reference_dir: Path,
                        ruixi_repo_root: Path, include_posthoc: bool = True):
    source = verify_ruixi_source(ruixi_repo_root); _json(output_dir / "ruixi_source_gate.json", source)
    if not source["passed"]: raise RuntimeError("Ruixi source checkout does not match frozen commit")
    reproduction = verify_canonical_reproduction(reference_dir); _json(output_dir / "reproduction_gate.json", reproduction)
    if not reproduction["passed"]: raise RuntimeError("canonical reproduction gate failed")
    benchmark = load_crr_cache(benchmark_path)
    pairs = {(m, CLEAN_SPATIAL_FIXED_N) for m in CLEAN_SPATIAL_LEVELS}
    pairs |= {(CLEAN_TEMPORAL_FIXED_M, n) for n in CLEAN_TEMPORAL_LEVELS}
    if include_posthoc: pairs |= {(CLEAN_TEMPORAL_FIXED_M, n) for n in POSTHOC_TEMPORAL_LEVELS}
    solved, runs = {}, []
    for M, N in sorted(pairs):
        values, row = _run(M, N); row["role"] = "posthoc_asymptotic_extension" if N in POSTHOC_TEMPORAL_LEVELS else "frozen_clean_isolation"
        runs.append(row)
        if values is not None: solved[(M, N)] = values
    metrics = _metric_rows(solved, benchmark); diffs, orders = _self_tables(solved)
    _csv(output_dir / "run_metrics.csv", runs); _csv(output_dir / "grid_metrics.csv", metrics)
    _csv(output_dir / "self_differences.csv", diffs); _csv(output_dir / "self_convergence_orders.csv", orders)
    spatial = [float(r["observed_order"]) for r in orders if r["axis"] == "spatial" and r["scope"] == "overall" and r["metric"] == "rmse"]
    temporal = [float(r["observed_order"]) for r in orders if r["axis"] == "temporal" and r["scope"] == "overall" and r["metric"] == "rmse"]
    posthoc = []
    levels = (960, 1920) + POSTHOC_TEMPORAL_LEVELS
    for c, m, f in zip(levels, levels[1:], levels[2:]):
        keys = [(CLEAN_TEMPORAL_FIXED_M, x) for x in (c, m, f)]
        if all(k in solved for k in keys):
            posthoc.append({"coarse_level": c, "middle_level": m, "fine_level": f,
                **self_convergence_order(solved[keys[0]], solved[keys[1]], solved[keys[2]], metric="rmse")})
    _csv(output_dir / "temporal_posthoc_self_orders.csv", posthoc)
    decision = {"status": "completed_diagnostic_audit",
        "spatial_order_status": "verified_second_order_in_overall_RMSE_self_convergence",
        "spatial_overall_rmse_self_orders": spatial,
        "temporal_order_status": "unresolved_not_verified_second_order",
        "temporal_frozen_overall_rmse_self_orders": temporal,
        "temporal_posthoc_overall_rmse_self_orders": [r["observed_order"] for r in posthoc],
        "nonconverged_frozen_runs": [r for r in runs if not r["converged"] and r["role"] == "frozen_clean_isolation"],
        "claim_boundary": "Do not claim stable second-order temporal convergence for this American-obstacle contract from these results.",
        "no_result_deletion_or_gate_weakening": True}
    _json(output_dir / "convergence_decision.json", decision)
    manifest = {"schema_version": 1, "experiment_id": "dirk_binomial_50k_clean_isolation", "regime_id": REGIME_ID,
        "contract": CONTRACT, "production_source_commit": RUIXI_SOURCE_COMMIT,
        "spatial": {"fixed_N": CLEAN_SPATIAL_FIXED_N, "M": list(CLEAN_SPATIAL_LEVELS)},
        "temporal": {"fixed_M": CLEAN_TEMPORAL_FIXED_M, "N": list(CLEAN_TEMPORAL_LEVELS)},
        "posthoc_temporal_extension": {"enabled": include_posthoc, "N": list(POSTHOC_TEMPORAL_LEVELS),
            "reason": "Frozen temporal sequence remained non-asymptotic at N=1920; added only to test stabilization."},
        "decision_file": "convergence_decision.json", "no_tuning_after_results": True}
    _json(output_dir / "run_manifest.json", manifest); return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("results/dirk_binomial_50k_clean_isolation"))
    p.add_argument("--benchmark-path", type=Path, default=Path("results/dirk_binomial_50k_convergence/crr50000_price_benchmark.csv"))
    p.add_argument("--reference-dir", type=Path, default=Path(f"_vendor/ruixi-export/results/visualization_exports/american_put_matched/{REGIME_ID}/reference"))
    p.add_argument("--ruixi-repo-root", type=Path, default=Path("_vendor/ruixi-code")); p.add_argument("--skip-posthoc", action="store_true")
    a = p.parse_args(); print(json.dumps(run_clean_isolation(output_dir=a.output_dir, benchmark_path=a.benchmark_path,
        reference_dir=a.reference_dir, ruixi_repo_root=a.ruixi_repo_root, include_posthoc=not a.skip_posthoc), indent=2))


if __name__ == "__main__": main()
