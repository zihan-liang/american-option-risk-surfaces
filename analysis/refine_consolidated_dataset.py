"""Refine and audit consolidated SURF experiment tables.

This module is a post-processing layer. It never trains/tunes models, changes
raw experiment artifacts, or recomputes the already-verified DIRK metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, Iterator

from analysis.consolidated_experiment_data import DATA_SUFFIXES, TABLE_HEADERS, flatten_json_scalars

ZIHAN_REPO = "zihan-liang/american-option-risk-surfaces"
RUIXI_REPO = "Ruixixu-hub/2026-surf-american-risk-surfaces"
REFERENCE_TYPE = "DIRK_policy_sinh_M480_N960"
REGIME_ID = "put_T100_s060_r005_q010"
SOURCE_META = {"repo", "branch", "source_commit", "branch_commit"}


def _git(repo: Path, *args: str, binary: bool = False):
    p = subprocess.run(["git", *args], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.stdout if binary else p.stdout.decode("utf-8", errors="replace")


def _ref(repo: Path, branch: str) -> str:
    remote = f"origin/{branch}"
    try:
        _git(repo, "rev-parse", "--verify", remote)
        return remote
    except subprocess.CalledProcessError:
        return branch


def _show(repo: Path, branch: str, path: str, binary: bool = False):
    return _git(repo, "show", f"{_ref(repo, branch)}:{path}", binary=binary)


def _safe_float(value: Any):
    try:
        if value in (None, ""):
            return None
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any):
    x = _safe_float(value)
    if x is None or not float(x).is_integer():
        return None
    return int(x)


def _boolish(value: Any):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "pass", "passed", "yes", "1", "success"}:
            return True
        if v in {"false", "fail", "failed", "no", "0"}:
            return False
    return None


def _csv_rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def _write_rows(path: Path, header: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(header), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row.get(k, "") for k in writer.fieldnames})


def real_remote_branches(repo: Path | str) -> list[str]:
    repo = Path(repo)
    fmt = "%(refname:short)%09%(symref)"
    try:
        lines = _git(repo, "for-each-ref", f"--format={fmt}", "refs/remotes/origin").splitlines()
    except subprocess.CalledProcessError:
        lines = []
    branches: list[str] = []
    for line in lines:
        short, _, symref = line.partition("\t")
        short = short.strip()
        if not short or symref.strip() or short in {"origin", "origin/HEAD"} or short.endswith("/HEAD"):
            continue
        branches.append(short[len("origin/"):] if short.startswith("origin/") else short)
    if not branches:
        branches = [x.strip() for x in _git(repo, "for-each-ref", "--format=%(refname:short)", "refs/heads").splitlines() if x.strip()]
    return sorted(dict.fromkeys(branches))


def refined_artifact_type(path: str) -> str:
    name = Path(path).name.lower()
    suffix = Path(name).suffix
    if suffix in {".pt", ".pth"}:
        return "checkpoint"
    if suffix in {".npz", ".npy"}:
        return "numeric_array"
    if suffix == ".csv":
        if "price_slice" in name:
            return "price_slice_csv"
        if name == "free_boundary.csv" or name.startswith("boundary_") or "boundary_metrics" in name:
            return "boundary_csv"
        if any(t in name for t in ("metric", "summary", "diagnostic")):
            return "metrics_csv"
        return "data_csv"
    if suffix == ".json":
        return "metadata_json"
    if suffix in {".txt", ".md"}:
        return "text_evidence"
    return "other"


def infer_method(path: str) -> str:
    p = path.replace("\\", "/").lower()
    if "/binomial/" in p:
        return "BINOMIAL_TREE"
    if "visualization_exports" in p and "/reference/" in p:
        return "DIRK_REFERENCE"
    if "arm_c_seed101" in p:
        return "ARM_C"
    if "arm_d_seed101" in p:
        return "ARM_D"
    if "deepsets_hvi" in p:
        return "DEEPSETS_HVI"
    if "fc_hvi" in p:
        return "FC_HVI"
    if "/oracle/" in p and "hd3" in p:
        return "HD_CNPSOR_ORACLE"
    for token, label in (
        ("cn_policy_iteration", "CN_POLICY_ITERATION"),
        ("policy_iteration", "POLICY_ITERATION"),
        ("projected_lu", "PROJECTED_LU"),
        ("cn_psor", "CN_PSOR"),
        ("dirk_psor", "DIRK_PSOR"),
    ):
        if token in p:
            return label
    return ""


def _path_hints(path: str) -> dict[str, Any]:
    p = path.lower()
    dim = None
    m = re.search(r"(?:^|[/_-])d(?:im)?(10|20|50|100|[1-9])(?:[/_.-]|$)", p)
    if m:
        dim = int(m.group(1))
    seed = None
    m = re.search(r"seed[_-]?(\d+)", p)
    if m:
        seed = int(m.group(1))
    regime = ""
    m = re.search(r"put_t\d+_s\d+_r\d+_q\d+", p)
    if m:
        regime = m.group(0)
    elif "matched_american_put" in p:
        regime = REGIME_ID
    family = ""
    if "hd1" in p:
        family = "HD1"
    elif "hd2" in p:
        family = "HD2"
    elif "hd3" in p:
        family = "HD3"
    else:
        m = re.search(r"(?:^|/)(m\d{2})(?:[/_.-]|$)", p)
        if m:
            family = m.group(1).upper()
    return {"method_id": infer_method(path), "dimension": dim, "seed": seed, "regime_id": regime, "experiment_family": family}


def is_metric_artifact(path: str) -> bool:
    name = Path(path).name.lower()
    suffix = Path(name).suffix
    excluded = (
        "price_slice", "training_history", "validation_history", "timing_samples",
        "surface_generation", "grid_refinement", "analytic_controls", "boundary_crosscheck",
        "selected_spot", "curve_sample", "profile", "boundary_paths", "sample_audit",
        "generation_log", "surface_hashes",
    )
    if any(token in name for token in excluded):
        return False
    if suffix == ".csv":
        return any(token in name for token in ("metric", "summary", "diagnostic", "comparison", "run_metrics", "selection", "scenario", "gate", "resource_accounting"))
    if suffix == ".json":
        return any(token in name for token in ("metric", "summary", "decision", "gate"))
    return False


def _existing_sha_map(output: Path) -> dict[tuple[str, str], dict[str, str]]:
    path = output / "source_artifact_inventory.csv"
    if not path.exists():
        return {}
    out: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[(row.get("repo", ""), row.get("git_blob_sha", ""))] = row
    return out


def enumerate_real_artifacts(repo: Path, repo_name: str, old_map: dict[tuple[str, str], dict[str, str]]) -> dict[str, Any]:
    provenance: list[dict[str, Any]] = []
    unique: dict[str, dict[str, Any]] = {}
    content_cache: dict[str, str] = {}
    for branch in real_remote_branches(repo):
        ref = _ref(repo, branch)
        commit = _git(repo, "rev-parse", ref).strip()
        for line in _git(repo, "ls-tree", "-r", "-l", ref).splitlines():
            if "\t" not in line:
                continue
            meta, path = line.split("\t", 1)
            parts = meta.split()
            if len(parts) < 4 or parts[1] != "blob" or path.startswith("results/consolidated_csv/"):
                continue
            suffix = Path(path).suffix.lower()
            if suffix not in DATA_SUFFIXES:
                continue
            blob = parts[2]
            size = int(parts[3]) if parts[3].isdigit() else None
            old = old_map.get((repo_name, blob), {})
            sha256 = old.get("content_sha256", "")
            if not sha256:
                if blob not in content_cache:
                    content_cache[blob] = hashlib.sha256(_git(repo, "cat-file", "blob", blob, binary=True)).hexdigest()
                sha256 = content_cache[blob]
            occurrence = {
                "repo": repo_name, "branch": branch, "branch_commit": commit, "path": path,
                "git_blob_sha": blob, "content_sha256": sha256, "duplicate_group": sha256[:16],
            }
            provenance.append(occurrence)
            if blob not in unique:
                hints = _path_hints(path)
                unique[blob] = {
                    "repo": repo_name, "representative_path": path, "extension": suffix,
                    "size_bytes": size, "git_blob_sha": blob, "content_sha256": sha256,
                    "artifact_type": refined_artifact_type(path), **hints,
                    "duplicate_group": sha256[:16], "canonical_source": f"{repo_name}@{branch}:{path}",
                    "_branch": branch, "_source_commit": commit, "_repo_path": repo,
                }
    provenance.sort(key=lambda r: (r["repo"], r["branch"], r["path"]))
    return {"branches": real_remote_branches(repo), "provenance": provenance, "artifacts": list(unique.values())}


def _public_artifact(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def _artifact_text(artifact: dict[str, Any]) -> str:
    return _show(artifact["_repo_path"], artifact["_branch"], artifact["representative_path"])


def _artifact_json(artifact: dict[str, Any]) -> Any:
    return json.loads(_artifact_text(artifact))


def _artifact_csv(artifact: dict[str, Any]) -> list[dict[str, str]]:
    return _csv_rows(_artifact_text(artifact))


def _source(artifact: dict[str, Any]) -> dict[str, Any]:
    return {"repo": artifact["repo"], "branch": artifact["_branch"], "source_commit": artifact["_source_commit"]}


def _rebuild_source_records(artifacts: list[dict[str, Any]], output: Path) -> int:
    header = TABLE_HEADERS["source_records.csv"]
    count = 0
    with (output / "source_records.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for art in artifacts:
            if art["extension"] not in {".csv", ".json"}:
                continue
            source = _source(art)
            base = {**source, "path": art["representative_path"], "git_blob_sha": art["git_blob_sha"]}
            try:
                if art["extension"] == ".csv":
                    for i, record in enumerate(_artifact_csv(art)):
                        for field, value in record.items():
                            writer.writerow({**base, "record_kind": "csv", "record_index": i, "field": field or "", "value": value, "value_type": "str"})
                            count += 1
                else:
                    obj = _artifact_json(art)
                    for field, value in flatten_json_scalars(obj):
                        if value is None:
                            typ = "null"
                        elif isinstance(value, bool):
                            typ = "bool"
                        elif isinstance(value, int):
                            typ = "int"
                        elif isinstance(value, float):
                            typ = "float"
                        else:
                            typ = "str"
                        writer.writerow({**base, "record_kind": "json", "record_index": 0, "field": field, "value": value, "value_type": typ})
                        count += 1
            except (json.JSONDecodeError, csv.Error, UnicodeDecodeError):
                continue
    return count


def parse_binomial_convergence(meta: dict[str, Any], source: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in meta.get("convergence", []):
        rmse = item.get("rmse_change_from_previous")
        maxerr = item.get("max_abs_change_from_previous")
        rows.append({
            **source, "steps": item.get("steps", ""), "S_over_K": "", "tau_over_T": item.get("tau_over_T", ""), "V_over_K": "",
            "comparison_metric": "rmse_change_from_previous;max_abs_change_from_previous",
            "comparison_value": f"rmse_change_from_previous={rmse};max_abs_change_from_previous={maxerr}",
        })
    return rows


def parse_binomial_slice(records: list[dict[str, str]], source: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        rows.append({
            **source, "steps": _safe_int(record.get("tree_steps") or record.get("steps")) or "",
            "S_over_K": record.get("S_over_K", ""), "tau_over_T": record.get("tau_over_T", ""),
            "V_over_K": record.get("V_over_K", record.get("V", "")), "comparison_metric": "", "comparison_value": "",
        })
    return rows


def parse_hd2_selection(selection: dict[str, Any], gates: dict[str, Any], source: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    status = selection.get("selection_status", "")
    mapping = {
        "price_ratio": "price_ratio_max", "gamma_ratio": "gamma_ratio_max", "boundary_ratio": "boundary_ratio_max",
        "pde_p95_ratio": "pde_p95_ratio_max", "fb_p95_ratio": "fb_p95_ratio_max",
    }
    for candidate, audit in selection.get("candidate_audit", {}).items():
        physics = _safe_float(audit.get("physics_score"))
        if physics is not None:
            metrics.append({**source, "candidate_id": candidate, "dimension": "", "seed": 101, "metric_name": "physics_score", "metric_value": physics, "gate_name": "", "gate_pass": "", "selection_status": status})
        eligible = _boolish(audit.get("eligible"))
        if eligible is not None:
            metrics.append({**source, "candidate_id": candidate, "dimension": "", "seed": 101, "metric_name": "eligible", "metric_value": int(eligible), "gate_name": "", "gate_pass": eligible, "selection_status": status})
        for dim, values in audit.get("dimensions", {}).items():
            overall = _boolish(values.get("passes"))
            for metric, gate_key in mapping.items():
                actual = _safe_float(values.get(metric))
                threshold = _safe_float(gates.get(gate_key))
                if actual is None:
                    continue
                passed = actual <= threshold if threshold is not None else None
                metrics.append({**source, "candidate_id": candidate, "dimension": _safe_int(dim) or dim, "seed": 101, "metric_name": metric, "metric_value": actual, "gate_name": gate_key, "gate_pass": passed, "selection_status": status})
                gate_rows.append({
                    **source, "method_id": candidate, "experiment_id": "hd2_hvi_objective_stabilization", "gate": gate_key,
                    "metric": metric, "operator": "<=", "threshold": threshold if threshold is not None else "", "actual": actual,
                    "passed": passed if passed is not None else "", "decision": status,
                })
            if overall is not None:
                gate_rows.append({**source, "method_id": candidate, "experiment_id": "hd2_hvi_objective_stabilization", "gate": "all_selection_gates", "metric": "passes", "operator": "", "threshold": "", "actual": overall, "passed": overall, "decision": status})
    return metrics, gate_rows


def _metric_row(base: dict[str, Any], name: str, value: Any, **extra) -> dict[str, Any] | None:
    x = _safe_float(value)
    if x is None:
        b = _boolish(value)
        if b is None:
            return None
        x = int(b)
    return {**base, "metric_name": name, "metric_value": x, **extra}


def _dedup_rows(rows: list[dict[str, Any]], ignore: set[str] | None = None) -> list[dict[str, Any]]:
    ignore = ignore or set()
    seen = set()
    out = []
    for row in rows:
        key = tuple(sorted((k, str(v)) for k, v in row.items() if k not in ignore))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _rebuild_price_slices(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for art in artifacts:
        name = Path(art["representative_path"]).name.lower()
        if art["extension"] != ".csv" or "price_slice" not in name:
            continue
        hints = _path_hints(art["representative_path"])
        method = hints["method_id"]
        if not method:
            continue
        source = _source(art)
        try:
            records = _artifact_csv(art)
        except Exception:
            continue
        for r in records:
            out.append({
                **source, "method_id": method, "dimension": hints["dimension"] or "", "seed": hints["seed"] or (101 if method in {"ARM_C", "ARM_D"} else ""),
                "regime_id": hints["regime_id"], "S_over_K": r.get("S_over_K", r.get("G_over_K", "")), "tau_over_T": r.get("tau_over_T", ""),
                "V_over_K": r.get("V_over_K", r.get("V", "")), "payoff_over_K": r.get("payoff_over_K", r.get("payoff", "")),
                "premium_over_K": r.get("premium_over_K", r.get("premium", "")),
                "reference_type": REFERENCE_TYPE if method == "DIRK_REFERENCE" else ("HD_CN_PSOR_internal" if method == "HD_CNPSOR_ORACLE" else ""),
            })
    return _dedup_rows(out, SOURCE_META)


def _rebuild_free_boundaries(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for art in artifacts:
        if Path(art["representative_path"]).name.lower() != "free_boundary.csv":
            continue
        hints = _path_hints(art["representative_path"])
        method = hints["method_id"]
        if not method:
            continue
        source = _source(art)
        try:
            records = _artifact_csv(art)
        except Exception:
            continue
        for r in records:
            boundary = r.get("S_star_over_K", r.get("boundary_over_K", r.get("predicted_boundary_over_K", "")))
            found = r.get("boundary_found", "")
            if found == "" and boundary not in ("", None):
                found = _safe_float(boundary) is not None
            out.append({
                **source, "method_id": method, "dimension": hints["dimension"] or "", "seed": hints["seed"] or (101 if method in {"ARM_C", "ARM_D", "FC_HVI", "DEEPSETS_HVI"} else ""),
                "regime_id": hints["regime_id"], "tau_over_T": r.get("tau_over_T", r.get("tau", "")), "boundary_over_K": boundary,
                "boundary_found": found, "threshold": r.get("boundary_tolerance", r.get("threshold", "")), "missing_reason": r.get("missing_reason", ""),
            })
    return _dedup_rows(out, SOURCE_META)


def _rebuild_binomial(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for art in artifacts:
        path = art["representative_path"].replace("\\", "/").lower()
        source = _source(art)
        try:
            if Path(path).name == "binomial_convergence.json":
                out.extend(parse_binomial_convergence(_artifact_json(art), source))
            elif "/binomial/" in path and "price_slice" in Path(path).name and art["extension"] == ".csv":
                out.extend(parse_binomial_slice(_artifact_csv(art), source))
        except Exception:
            continue
    return _dedup_rows(out, SOURCE_META)


def _rebuild_hd_metrics(artifacts: list[dict[str, Any]], dirk_price_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    tables = {k: [] for k in ("experiment_runs.csv", "price_metrics.csv", "greek_metrics.csv", "boundary_metrics.csv", "structural_metrics.csv", "compute_metrics.csv", "hd1_metrics.csv", "hd2_candidate_metrics.csv", "hd3_metrics.csv")}
    tables["price_metrics.csv"].extend(dirk_price_rows)
    hd2_selection = None
    hd2_gates: dict[str, Any] = {}
    hd2_source = None
    for art in artifacts:
        p = art["representative_path"].lower()
        if p.endswith("results/hd2_hvi_objective_stabilization/selection_decision.json"):
            hd2_selection = _artifact_json(art); hd2_source = _source(art)
        elif p.endswith("results/hd2_hvi_objective_stabilization/config_manifest.json"):
            obj = _artifact_json(art); hd2_gates = obj.get("config", {}).get("selection_gates", {})
    for art in artifacts:
        path = art["representative_path"]
        name = Path(path).name.lower()
        hints = _path_hints(path)
        family = hints["experiment_family"]
        if art["extension"] != ".csv":
            continue
        if not ((name == "run_metrics.csv" and family in {"HD1", "HD3"}) or (name == "test_metrics.csv" and family == "HD2")):
            continue
        try:
            records = _artifact_csv(art)
        except Exception:
            continue
        for r in records:
            method = r.get("model_id") or r.get("model") or r.get("candidate") or r.get("method_id") or hints["method_id"]
            if not method:
                continue
            dim = _safe_int(r.get("dimension")) or hints["dimension"] or ""
            seed = _safe_int(r.get("seed")) or hints["seed"] or 101
            source = _source(art)
            experiment_id = Path(path).parent.name
            base = {**source, "experiment_id": experiment_id, "method_id": method, "dimension": dim, "seed": seed}
            reference = "HD_CN_PSOR_internal"
            tables["experiment_runs.csv"].append({**base, "experiment_family": family, "variant": r.get("variant", ""), "status": r.get("status", r.get("selection_status", "")), "diagnostic_only": r.get("diagnostic_only", ""), "candidate_supported": r.get("candidate_supported", ""), "reference_type": reference, "scoring_protocol": "internal_native"})
            for field, raw in r.items():
                value = _safe_float(raw)
                if value is None:
                    continue
                fl = field.lower()
                if "price" in fl:
                    tables["price_metrics.csv"].append({**base, "metric_name": field, "metric_value": value, "reference_type": reference, "metric_scope": "internal_diagnostic", "regime_id": hints["regime_id"]})
                if "delta" in fl or "gamma" in fl:
                    greek_def = "covariance_weighted_hessian_contraction" if "gamma_contraction" in fl else ("reduced_aggregate" if "reduced" in fl else "component_or_scalar")
                    tables["greek_metrics.csv"].append({**base, "metric_name": field, "metric_value": value, "greek_definition": greek_def, "greek_source": "stored_metric", "reference_type": reference})
                if field in {"permutation_consistency_rmse", "transverse_invariance_rmse", "parameter_count"}:
                    tables["structural_metrics.csv"].append({**source, "method_id": method, "dimension": dim, "seed": seed, "metric_name": field, "metric_value": value})
                target = "hd1_metrics.csv" if family == "HD1" else ("hd2_candidate_metrics.csv" if family == "HD2" else "hd3_metrics.csv")
                if target == "hd1_metrics.csv":
                    tables[target].append({**source, "model_id": method, "dimension": dim, "seed": seed, "metric_name": field, "metric_value": value, "gate_name": "", "gate_pass": ""})
                elif target == "hd2_candidate_metrics.csv":
                    tables[target].append({**source, "candidate_id": method, "dimension": dim, "seed": seed, "metric_name": field, "metric_value": value, "gate_name": "", "gate_pass": "", "selection_status": r.get("selection_status", "")})
                else:
                    tables[target].append({**source, "model_id": method, "dimension": dim, "seed": seed, "metric_name": field, "metric_value": value, "reference_type": reference, "regime_id": hints["regime_id"]})
            boundary_fields = {"threshold": r.get("primary_boundary_tolerance", ""), "reference_count": r.get("reference_boundary_count", ""), "predicted_count": r.get("predicted_boundary_count", ""), "matched_count": r.get("matched_boundary_count", ""), "coverage": r.get("boundary_coverage", ""), "found_rate": r.get("boundary_coverage", ""), "conditional_rmse": r.get("boundary_conditional_rmse", ""), "full_rmse": r.get("boundary_full_rmse", r.get("boundary_rmse", "")), "exercise_f1": r.get("exercise_f1", ""), "boundary_tube_f1": r.get("boundary_tube_f1", ""), "missing_reason": "", "boundary_extraction_method": ""}
            if any(v not in ("", None) for v in boundary_fields.values()):
                tables["boundary_metrics.csv"].append({**base, **boundary_fields})
            tables["compute_metrics.csv"].append({**source, "method_id": method, "dimension": dim, "seed": seed, "training_seconds": r.get("training_seconds", ""), "evaluation_seconds": r.get("evaluation_seconds", r.get("inference_seconds", "")), "solver_seconds": "", "oracle_seconds": r.get("oracle_seconds", ""), "screening_seconds": "", "confirmation_seconds": "", "wall_clock_seconds": "", "hardware": "", "protocol": "oracle_assisted"})
    if hd2_selection is not None and hd2_source is not None:
        selection_metrics, _ = parse_hd2_selection(hd2_selection, hd2_gates, hd2_source)
        tables["hd2_candidate_metrics.csv"].extend(selection_metrics)
    for key in tables:
        tables[key] = _dedup_rows(tables[key], SOURCE_META)
    return tables


def _rebuild_arm_metrics(artifacts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    arm_metrics: list[dict[str, Any]] = []
    price: list[dict[str, Any]] = []
    greek: list[dict[str, Any]] = []
    boundary: list[dict[str, Any]] = []
    for art in artifacts:
        p = art["representative_path"].lower()
        if not p.endswith("scoring/pinn/pinn_metrics_by_regime_seed.csv") or ("arm_c_seed101" not in p and "arm_d_seed101" not in p):
            continue
        method = "ARM_C" if "arm_c_seed101" in p else "ARM_D"
        arm = "C" if method == "ARM_C" else "D"
        source = _source(art)
        for r in _artifact_csv(art):
            regime = r.get("regime_id", "")
            seed = _safe_int(r.get("seed")) or 101
            scope = f"split={r.get('split','')};region={r.get('region','all')}"
            base = {**source, "experiment_id": "ruixi_heldout_pilot", "method_id": method, "dimension": "", "seed": seed}
            for field, raw in r.items():
                value = _safe_float(raw)
                if value is None:
                    continue
                metric_name = f"{scope}:{field}"
                arm_metrics.append({**source, "arm": arm, "method_id": method, "regime_id": regime, "seed": seed, "metric_name": metric_name, "metric_value": value, "reference_type": r.get("reference_source", "")})
                fl = field.lower()
                if fl in {"rmse", "mae", "max_abs_error", "relative_l2"}:
                    price.append({**base, "metric_name": metric_name, "metric_value": value, "reference_type": r.get("reference_source", ""), "metric_scope": scope, "regime_id": regime})
                if "delta" in fl or "gamma" in fl:
                    greek.append({**base, "metric_name": metric_name, "metric_value": value, "greek_definition": "scaled_gamma" if "scaled_gamma" in fl else "scalar_delta", "greek_source": "stored_scoring", "reference_type": r.get("reference_source", "")})
            if r.get("region", "") == "all":
                boundary.append({**base, "threshold": "", "reference_count": r.get("boundary_reference_found", ""), "predicted_count": r.get("boundary_predicted_found", ""), "matched_count": "", "coverage": r.get("boundary_found_rate", ""), "found_rate": r.get("boundary_found_rate", ""), "conditional_rmse": r.get("boundary_conditional_rmse", ""), "full_rmse": "", "exercise_f1": "", "boundary_tube_f1": "", "missing_reason": "", "boundary_extraction_method": "stored_scoring"})
    return (_dedup_rows(arm_metrics, SOURCE_META), _dedup_rows(price, SOURCE_META), _dedup_rows(greek, SOURCE_META), _dedup_rows(boundary, SOURCE_META))


def _rebuild_classical(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for art in artifacts:
        p = art["representative_path"].replace("\\", "/").lower()
        source = _source(art)
        if p.endswith("results/independent_method_campaign/m01/screen/run_metrics.csv"):
            for r in _artifact_csv(art):
                method = r.get("config_id") or r.get("method_id") or "M01"
                base = {**source, "method_id": method, "regime_id": "", "reference_type": ""}
                for field in ("primary_metric_value", "baseline_metric_value", "scenario_count", "memory_mb"):
                    row = _metric_row(base, field, r.get(field))
                    if row: out.append(row)
                for field in ("optional_metrics_json", "timings_seconds_json"):
                    try:
                        obj = json.loads(r.get(field, "") or "{}")
                    except json.JSONDecodeError:
                        obj = {}
                    for key, value in flatten_json_scalars(obj):
                        row = _metric_row(base, f"{field}.{key}", value)
                        if row: out.append(row)
        elif p.endswith("scoring/classical/classical_baseline_metrics.csv"):
            for r in _artifact_csv(art):
                method = r.get("variant", "CLASSICAL")
                scope = f"split={r.get('split','')};region={r.get('region','all')}"
                base = {**source, "method_id": method, "regime_id": r.get("regime_id", ""), "reference_type": r.get("reference_source", "")}
                for field, raw in r.items():
                    row = _metric_row(base, f"{scope}:{field}", raw)
                    if row: out.append(row)
    return _dedup_rows(out, SOURCE_META)


def _campaign_family(path: str) -> str:
    m = re.search(r"(?:^|/)(m\d{2})(?:/|[_.-])", path.lower())
    return m.group(1).upper() if m else ""


def _rebuild_campaign(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for art in artifacts:
        family = _campaign_family(art["representative_path"])
        if not family or not is_metric_artifact(art["representative_path"]):
            continue
        source = _source(art)
        name = Path(art["representative_path"]).name
        try:
            if art["extension"] == ".csv":
                for i, r in enumerate(_artifact_csv(art)):
                    method_name = r.get("config_id") or r.get("candidate") or r.get("method_name") or r.get("model") or ""
                    status = r.get("status", r.get("run_status", ""))
                    decision = r.get("decision", r.get("claim_status", ""))
                    for field, raw in r.items():
                        if field.endswith("_json") and isinstance(raw, str):
                            try:
                                obj = json.loads(raw)
                            except json.JSONDecodeError:
                                obj = None
                            if obj is not None:
                                for key, value in flatten_json_scalars(obj):
                                    row = _metric_row({**source, "method_id": family, "method_name": method_name, "run_status": status, "gate_name": "", "gate_threshold": "", "gate_pass": "", "final_decision": decision}, f"{name}[{i}].{field}.{key}", value)
                                    if row: out.append(row)
                                continue
                        row = _metric_row({**source, "method_id": family, "method_name": method_name, "run_status": status, "gate_name": r.get("gate_name", ""), "gate_threshold": r.get("gate_threshold", ""), "gate_pass": r.get("gate_pass", ""), "final_decision": decision}, f"{name}[{i}].{field}", raw)
                        if row: out.append(row)
            elif art["extension"] == ".json":
                obj = _artifact_json(art)
                decision = obj.get("selection_status", obj.get("decision", obj.get("outcome", ""))) if isinstance(obj, dict) else ""
                for key, value in flatten_json_scalars(obj):
                    row = _metric_row({**source, "method_id": family, "method_name": "", "run_status": "", "gate_name": "", "gate_threshold": "", "gate_pass": "", "final_decision": decision}, f"{name}.{key}", value)
                    if row: out.append(row)
        except Exception:
            continue
    return _dedup_rows(out, SOURCE_META)


def _rebuild_gates(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    hd2_selection = None; hd2_gates = {}; hd2_source = None
    for art in artifacts:
        p = art["representative_path"].lower()
        if p.endswith("results/hd2_hvi_objective_stabilization/selection_decision.json"):
            hd2_selection = _artifact_json(art); hd2_source = _source(art)
        if p.endswith("results/hd2_hvi_objective_stabilization/config_manifest.json"):
            hd2_gates = _artifact_json(art).get("config", {}).get("selection_gates", {})
    if hd2_selection is not None and hd2_source is not None:
        _, rows = parse_hd2_selection(hd2_selection, hd2_gates, hd2_source)
        out.extend(rows)
    for art in artifacts:
        name = Path(art["representative_path"]).name.lower()
        if "gate" not in name or art["extension"] not in {".csv", ".json"}:
            continue
        source = _source(art); hints = _path_hints(art["representative_path"])
        try:
            if art["extension"] == ".csv":
                for r in _artifact_csv(art):
                    method = r.get("method_id") or r.get("config_id") or r.get("variant") or hints["method_id"] or hints["experiment_family"]
                    gate = r.get("gate") or r.get("gate_name") or r.get("name") or name
                    metric = r.get("metric") or r.get("metric_name") or r.get("check") or ""
                    threshold = r.get("threshold") or r.get("gate_threshold") or r.get("limit") or ""
                    actual = r.get("actual") or r.get("metric_value") or r.get("value") or ""
                    passed = r.get("passed") or r.get("pass") or r.get("gate_pass") or ""
                    decision = r.get("decision") or r.get("status") or ""
                    if any(x not in ("", None) for x in (threshold, actual, passed, decision)):
                        out.append({**source, "method_id": method, "experiment_id": Path(art["representative_path"]).parent.name, "gate": gate, "metric": metric, "operator": r.get("operator", ""), "threshold": threshold, "actual": actual, "passed": passed, "decision": decision})
            else:
                obj = _artifact_json(art)
                method = hints["method_id"] or hints["experiment_family"]
                for key, value in flatten_json_scalars(obj):
                    low = key.lower()
                    if not any(t in low for t in ("threshold", "_max", "_min", "pass", "reject", "eligible")):
                        continue
                    b = _boolish(value)
                    out.append({**source, "method_id": method, "experiment_id": Path(art["representative_path"]).parent.name, "gate": key, "metric": key, "operator": "", "threshold": value if ("threshold" in low or low.endswith("_max") or low.endswith("_min")) else "", "actual": "", "passed": b if b is not None else "", "decision": ""})
        except Exception:
            continue
    return _dedup_rows(out, SOURCE_META)


def _rebuild_environments(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for art in artifacts:
        if Path(art["representative_path"]).name.lower() != "environment_manifest.json":
            continue
        try:
            obj = _artifact_json(art)
        except Exception:
            continue
        hints = _path_hints(art["representative_path"]); source = _source(art)
        cuda_available = obj.get("cuda_available")
        device = obj.get("device", "")
        if not device and cuda_available is not None:
            device = "cuda" if bool(cuda_available) else "cpu"
        out.append({
            **source, "experiment_id": Path(art["representative_path"]).parent.name, "method_id": hints["method_id"], "seed": hints["seed"] or "",
            "os": obj.get("platform", obj.get("os", "")), "python": obj.get("python", ""), "pytorch": obj.get("torch", obj.get("pytorch", "")),
            "cuda": obj.get("torch_cuda", obj.get("cuda", obj.get("cuda_driver", ""))), "gpu": obj.get("gpu_name", obj.get("gpu", "")),
            "cpu": obj.get("cpu", ""), "ram": obj.get("ram", obj.get("system_ram_bytes", "")), "device": device,
        })
    return _dedup_rows(out, SOURCE_META)


def _canonicalize_existing(output: Path, filename: str, require_method: bool = False) -> list[dict[str, Any]]:
    path = output / filename
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if require_method:
        rows = [r for r in rows if r.get("method_id") or r.get("model_id") or r.get("candidate_id")]
    return _dedup_rows(rows, SOURCE_META)


def audit_rows(table: str, rows: list[dict[str, Any]], key_columns: list[str]) -> dict[str, Any]:
    seen = set(); duplicates = 0; missing = 0
    for row in rows:
        key = tuple(sorted((k, str(v)) for k, v in row.items()))
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
        if any(row.get(k) in (None, "") for k in key_columns):
            missing += 1
    if not rows:
        status = "empty"
    elif duplicates or missing:
        status = "needs_review"
    else:
        status = "ok"
    return {"table": table, "status": status, "row_count": len(rows), "duplicate_rows": duplicates, "missing_key_rows": missing, "key_columns": ";".join(key_columns), "source_evidence": "refined_unique_git_artifacts", "notes": ""}


AUDIT_KEYS = {
    "source_artifact_inventory.csv": ["repo", "git_blob_sha"], "provenance.csv": ["repo", "branch", "path"],
    "source_records.csv": ["git_blob_sha", "record_kind", "record_index", "field"], "array_artifacts.csv": ["content_sha256", "array_name"],
    "experiment_runs.csv": ["experiment_id", "method_id"], "price_metrics.csv": ["method_id", "metric_name"], "greek_metrics.csv": ["method_id", "metric_name"],
    "boundary_metrics.csv": ["method_id"], "free_boundary_points.csv": ["method_id", "tau_over_T"], "price_slices.csv": ["method_id", "S_over_K", "tau_over_T"],
    "structural_metrics.csv": ["method_id", "metric_name"], "compute_metrics.csv": ["method_id"], "classical_solver_metrics.csv": ["method_id", "metric_name"],
    "binomial_convergence.csv": ["steps", "tau_over_T"], "ruixi_arm_metrics.csv": ["arm", "regime_id", "metric_name"],
    "hd1_metrics.csv": ["model_id", "dimension", "metric_name"], "hd2_candidate_metrics.csv": ["candidate_id", "metric_name"], "hd3_metrics.csv": ["model_id", "dimension", "metric_name"],
    "campaign_method_results.csv": ["method_id", "metric_name"], "experiment_gates.csv": ["gate"], "environment_runs.csv": ["experiment_id"],
    "surface_artifacts.csv": ["content_sha256"], "dirk_common_reference_metrics.csv": ["method_id", "metric_scope"],
}

SUSPICIOUS_MAX = {"ruixi_arm_metrics.csv": 100_000, "campaign_method_results.csv": 100_000, "classical_solver_metrics.csv": 100_000, "hd1_metrics.csv": 10_000, "hd2_candidate_metrics.csv": 20_000, "hd3_metrics.csv": 30_000}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return sum(1 for _ in reader)


def refine_dataset(zihan_repo: Path, ruixi_repo: Path, output: Path) -> dict[str, Any]:
    output = Path(output)
    before_dirk = (output / "dirk_common_reference_metrics.csv").read_bytes()
    old_map = _existing_sha_map(output)
    z = enumerate_real_artifacts(Path(zihan_repo), ZIHAN_REPO, old_map)
    r = enumerate_real_artifacts(Path(ruixi_repo), RUIXI_REPO, old_map)
    artifacts = z["artifacts"] + r["artifacts"]
    provenance = z["provenance"] + r["provenance"]

    _write_rows(output / "source_artifact_inventory.csv", TABLE_HEADERS["source_artifact_inventory.csv"], [_public_artifact(a) for a in artifacts])
    _write_rows(output / "provenance.csv", TABLE_HEADERS["provenance.csv"], provenance)
    source_record_count = _rebuild_source_records(artifacts, output)

    # Preserve the verified common-reference metric table exactly.
    dirk_price_rows = [r for r in _canonicalize_existing(output, "price_metrics.csv", require_method=True) if str(r.get("metric_name", "")).startswith(("dirk_", "internal_cnpsor_price_rmse")) and r.get("regime_id") == REGIME_ID]
    hd = _rebuild_hd_metrics(artifacts, dirk_price_rows)
    arm_metrics, arm_price, arm_greek, arm_boundary = _rebuild_arm_metrics(artifacts)
    hd["price_metrics.csv"].extend(arm_price); hd["greek_metrics.csv"].extend(arm_greek); hd["boundary_metrics.csv"].extend(arm_boundary)

    price_slices = _rebuild_price_slices(artifacts)
    free_boundaries = _rebuild_free_boundaries(artifacts)
    binomial = _rebuild_binomial(artifacts)
    classical = _rebuild_classical(artifacts)
    campaign = _rebuild_campaign(artifacts)
    gates = _rebuild_gates(artifacts)
    environments = _rebuild_environments(artifacts)

    # Add concise Arm experiment-run rows from the scoring evidence.
    for method in ("ARM_C", "ARM_D"):
        matching = [x for x in arm_metrics if x["method_id"] == method]
        if matching:
            first = matching[0]
            hd["experiment_runs.csv"].append({"repo": first["repo"], "branch": first["branch"], "source_commit": first["source_commit"], "experiment_family": "RUIXI_PINN", "experiment_id": "ruixi_heldout_pilot", "method_id": method, "variant": "soft_lcp" if method == "ARM_C" else "etc_fb_mixture", "dimension": 1, "seed": 101, "status": "completed", "diagnostic_only": True, "candidate_supported": "", "reference_type": REFERENCE_TYPE, "scoring_protocol": "physics_only_heldout"})

    # Campaign run rows, one per method family represented by refined aggregate evidence.
    for family in sorted({x["method_id"] for x in campaign}):
        row = next(x for x in campaign if x["method_id"] == family)
        hd["experiment_runs.csv"].append({"repo": row["repo"], "branch": row["branch"], "source_commit": row["source_commit"], "experiment_family": "CAMPAIGN", "experiment_id": family, "method_id": family, "variant": "", "dimension": "", "seed": "", "status": "recorded", "diagnostic_only": "", "candidate_supported": "", "reference_type": "", "scoring_protocol": "campaign"})

    specialized = {
        **hd,
        "price_slices.csv": price_slices,
        "free_boundary_points.csv": free_boundaries,
        "binomial_convergence.csv": binomial,
        "classical_solver_metrics.csv": classical,
        "ruixi_arm_metrics.csv": arm_metrics,
        "campaign_method_results.csv": campaign,
        "experiment_gates.csv": gates,
        "environment_runs.csv": environments,
    }
    for name, rows in specialized.items():
        if name in TABLE_HEADERS:
            rows = _dedup_rows(rows, SOURCE_META)
            _write_rows(output / name, TABLE_HEADERS[name], rows)

    # Base builder already exports arrays/surfaces from unique blobs. Canonicalize
    # those and any table not rebuilt above by scientific content, not branch copy.
    for name in ("array_artifacts.csv", "surface_artifacts.csv"):
        rows = _canonicalize_existing(output, name)
        _write_rows(output / name, TABLE_HEADERS[name], rows)

    if (output / "dirk_common_reference_metrics.csv").read_bytes() != before_dirk:
        raise RuntimeError("refinement changed verified DIRK common-reference metrics")

    audits: list[dict[str, Any]] = []
    for name in TABLE_HEADERS:
        path = output / name
        if name == "source_records.csv":
            audit = {"table": name, "status": "ok" if source_record_count else "empty", "row_count": source_record_count, "duplicate_rows": 0, "missing_key_rows": 0, "key_columns": ";".join(AUDIT_KEYS[name]), "source_evidence": "one_parse_per_unique_git_blob", "notes": "lossless CSV/JSON scalar fallback"}
        else:
            rows = _read_rows(path)
            audit = audit_rows(name, rows, AUDIT_KEYS.get(name, []))
            if name in SUSPICIOUS_MAX and len(rows) > SUSPICIOUS_MAX[name]:
                audit["status"] = "needs_review"; audit["notes"] = f"row count exceeds specialized-table sanity limit {SUSPICIOUS_MAX[name]}"
        audits.append(audit)

    audit_header = ["table", "status", "row_count", "duplicate_rows", "missing_key_rows", "key_columns", "source_evidence", "notes"]
    _write_rows(output / "table_audit.csv", audit_header, audits)
    (output / "table_audit.json").write_text(json.dumps(audits, indent=2) + "\n", encoding="utf-8")

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["repositories"][ZIHAN_REPO] = {"branches_scanned": len(z["branches"]), "branches": z["branches"]}
    manifest["repositories"][RUIXI_REPO] = {"branches_scanned": len(r["branches"]), "branches": r["branches"]}
    counts = {}
    for name in TABLE_HEADERS:
        counts[name] = source_record_count if name == "source_records.csv" else _row_count(output / name)
    manifest["counts"] = counts
    manifest["refinement"] = {
        "real_branch_counts": {ZIHAN_REPO: len(z["branches"]), RUIXI_REPO: len(r["branches"])},
        "symbolic_remote_head_excluded": True,
        "source_records_deduplicated_by_unique_blob": True,
        "specialized_tables_rebuilt_from_canonical_artifacts": sorted(specialized),
        "table_audit": "table_audit.csv",
        "dirk_common_reference_metrics_preserved_byte_for_byte": True,
    }
    notes = list(manifest.get("notes", []))
    for note in (
        "Refinement excludes symbolic remote HEAD refs from branch counts/provenance.",
        "Specialized metric tables are rebuilt from canonical aggregate artifacts; pointwise/history data remain in the lossless source_records/array exports instead of being mislabeled as metrics.",
        "Every logical table is audited in table_audit.csv/json; empty is never interpreted as zero.",
    ):
        if note not in notes:
            notes.append(note)
    manifest["notes"] = notes
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    readme = output / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Consolidated experiment CSV data\n"
    section = "\n## Refinement and table audit\n\n`table_audit.csv` and `table_audit.json` report row counts, duplicate rows, and required-key gaps for every logical table. Specialized metric tables use canonical aggregate artifacts only; histories, pointwise curves, and other raw numerical evidence remain available through `source_records.csv`, `array_artifacts.csv`, and the array exports. Symbolic Git remote HEAD refs are excluded from branch counts.\n"
    if "## Refinement and table audit" not in text:
        text += section
    readme.write_text(text, encoding="utf-8")
    return manifest


def verify_refinement(output: Path) -> dict[str, Any]:
    output = Path(output)
    audits = json.loads((output / "table_audit.json").read_text(encoding="utf-8"))
    bad = [r for r in audits if r["status"] != "ok"]
    if bad:
        raise ValueError(f"table audit has unresolved issues: {bad}")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    ruixi = manifest["repositories"][RUIXI_REPO]
    if "origin" in ruixi["branches"] or any(str(x).endswith("/HEAD") for x in ruixi["branches"]):
        raise ValueError(f"symbolic remote HEAD leaked into branch inventory: {ruixi['branches']}")
    required_nonempty = ["binomial_convergence.csv", "hd2_candidate_metrics.csv", "experiment_gates.csv", "environment_runs.csv", "ruixi_arm_metrics.csv", "campaign_method_results.csv", "classical_solver_metrics.csv"]
    for name in required_nonempty:
        if manifest["counts"].get(name, 0) <= 0:
            raise ValueError(f"required refined table is empty: {name}")
    if not manifest.get("refinement", {}).get("dirk_common_reference_metrics_preserved_byte_for_byte"):
        raise ValueError("DIRK preservation invariant missing")
    return {"tables": len(audits), "all_tables_ok": True, "zihan_branches": manifest["repositories"][ZIHAN_REPO]["branches_scanned"], "ruixi_branches": ruixi["branches_scanned"], "counts": {k: manifest["counts"][k] for k in required_nonempty}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zihan-repo", type=Path)
    parser.add_argument("--ruixi-repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_refinement(args.output), indent=2))
        return
    if args.zihan_repo is None or args.ruixi_repo is None:
        parser.error("--zihan-repo and --ruixi-repo are required unless --verify-only is used")
    manifest = refine_dataset(args.zihan_repo, args.ruixi_repo, args.output)
    print(json.dumps({"refinement": manifest["refinement"], "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
