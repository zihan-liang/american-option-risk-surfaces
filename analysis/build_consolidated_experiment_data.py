"""Build branch-complete consolidated experiment CSVs from immutable Git artifacts."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from analysis.consolidated_experiment_data import (
    HD_DOMAIN,
    SHARED_DOMAIN,
    TABLE_HEADERS,
    flatten_json_scalars,
    inventory_repository,
    normalize_surface_arrays,
    price_error_metrics,
    price_rmse,
    rectilinear_interpolate,
    write_empty_output_contract,
)

ZIHAN_REPO_NAME = "zihan-liang/american-option-risk-surfaces"
RUIXI_REPO_NAME = "Ruixixu-hub/2026-surf-american-risk-surfaces"
HD_BRANCH = "origin/exp/hd3-matched-american-put-final-boundary-complete"
RUIXI_VIZ_BRANCH = "origin/viz/american-put-pointwise-export"
REGIME_ID = "put_T100_s060_r005_q010"
HD_RESULT_ROOT = "results/hd3_matched_american_put_boundary_complete"
RUIXI_COMPACT_ROOT = f"results/visualization_exports/american_put_matched/{REGIME_ID}"
REFERENCE_TYPE = "DIRK_policy_sinh_M480_N960"
REFERENCE_SOURCE = "DIRK + Policy Iteration + sinh grid (M=480,N=960)"
EXPECTED_ARM_D_BYTES = 677_973_093
EXPECTED_ARM_D_SHA256 = "237d670ccb889564d94a9f553cf9528082f71c2a6891687585c68e56dedcbc5c"


def _git(repo: Path, *args: str, binary: bool = False):
    p = subprocess.run(["git", *args], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.stdout if binary else p.stdout.decode("utf-8", errors="replace")


def _show_bytes(repo: Path, ref: str, path: str) -> bytes:
    return _git(repo, "show", f"{ref}:{path}", binary=True)


def _show_text(repo: Path, ref: str, path: str) -> str:
    return _show_bytes(repo, ref, path).decode("utf-8", errors="replace")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_float(v: Any):
    try:
        if v is None or v == "":
            return None
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any):
    x = _safe_float(v)
    return int(x) if x is not None and float(x).is_integer() else None


def _value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    return "str"


def _branch_ref(repo: Path, branch: str) -> str:
    remote = f"origin/{branch}"
    try:
        _git(repo, "rev-parse", "--verify", remote)
        return remote
    except subprocess.CalledProcessError:
        return branch


def _branch_tip(repo: Path, branch_ref: str) -> str:
    return _git(repo, "rev-parse", branch_ref).strip()


def _parse_csv_text(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def _write_dict_rows(path: Path, header: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") if row.get(k, "") is not None else "" for k in header})


def _infer_path(path: str) -> dict[str, Any]:
    lower = path.lower()
    method = ""
    for token, label in [
        ("deepsets_hvi", "DEEPSETS_HVI"), ("fc_hvi", "FC_HVI"),
        ("arm_c_seed101", "ARM_C"), ("arm_d_seed101", "ARM_D"),
        ("cn_psor", "CN_PSOR"), ("projected_lu", "PROJECTED_LU"),
        ("policy_iteration", "POLICY_ITERATION")]:
        if token in lower:
            method = label
            break
    dim = None
    m = re.search(r"(?:^|[/_-])d(?:im)?(10|20|50|100|[1-9])(?:[/_.-]|$)", lower)
    if m:
        dim = int(m.group(1))
    seed = None
    m = re.search(r"seed[_-]?(\d+)", lower)
    if m:
        seed = int(m.group(1))
    regime = ""
    m = re.search(r"put_t\d+_s\d+_r\d+_q\d+", lower)
    if m:
        regime = m.group(0)
    family = ""
    for name in ("hd1", "hd2", "hd3"):
        if name in lower:
            family = name.upper()
            break
    if not family:
        m = re.search(r"(?:^|/)m(\d{2})(?:[/_.-]|$)", lower)
        if m:
            family = f"M{m.group(1)}"
    return {"method_id": method, "dimension": dim, "seed": seed, "regime_id": regime, "experiment_family": family}


def _load_npz_bytes(data: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _load_npy_bytes(data: bytes) -> dict[str, np.ndarray]:
    return {"array": np.asarray(np.load(io.BytesIO(data), allow_pickle=False))}


def _export_array_csv_gz(array: np.ndarray, destination: Path) -> None:
    arr = np.asarray(array)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destination, "wt", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"i{i}" for i in range(arr.ndim)] + ["value"])
        if arr.ndim == 0:
            writer.writerow([arr.item()])
            return
        for index in np.ndindex(arr.shape):
            value = arr[index]
            if hasattr(value, "item"):
                value = value.item()
            writer.writerow([*index, value])


def _generic_records(repo: Path, inventory: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for occ in inventory["provenance"]:
        path = occ["path"]
        suffix = Path(path).suffix.lower()
        if suffix not in {".csv", ".json"}:
            continue
        ref = _branch_ref(repo, occ["branch"])
        try:
            text = _show_text(repo, ref, path)
        except subprocess.CalledProcessError:
            continue
        if suffix == ".csv":
            try:
                records = _parse_csv_text(text)
            except csv.Error:
                continue
            for i, record in enumerate(records):
                for field, value in record.items():
                    rows.append({**occ, "source_commit": occ["branch_commit"], "record_kind": "csv", "record_index": i, "field": field or "", "value": value, "value_type": "str"})
        else:
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                continue
            for field, value in flatten_json_scalars(obj):
                rows.append({**occ, "source_commit": occ["branch_commit"], "record_kind": "json", "record_index": 0, "field": field, "value": value, "value_type": _value_type(value)})
    return rows


def _array_exports(repo: Path, inventory: dict[str, Any], output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    array_rows: list[dict[str, Any]] = []
    surface_rows: list[dict[str, Any]] = []
    occurrence_by_blob = {}
    for occ in inventory["provenance"]:
        occurrence_by_blob.setdefault(occ["git_blob_sha"], occ)
    for artifact in inventory["artifacts"]:
        if artifact["extension"] not in {".npz", ".npy"}:
            continue
        occ = occurrence_by_blob[artifact["git_blob_sha"]]
        ref = _branch_ref(repo, occ["branch"])
        try:
            data = _show_bytes(repo, ref, occ["path"])
            arrays = _load_npz_bytes(data) if artifact["extension"] == ".npz" else _load_npy_bytes(data)
        except Exception:
            continue
        hints = _infer_path(occ["path"])
        for name, arr in arrays.items():
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
            rel = Path("arrays") / f"{artifact['content_sha256'][:16]}__{safe_name}.csv.gz"
            target = output / rel
            try:
                _export_array_csv_gz(arr, target)
                rel_str = rel.as_posix()
            except Exception:
                rel_str = ""
            array_rows.append({**occ, "source_commit": occ["branch_commit"], "array_name": name, "shape": "x".join(map(str, arr.shape)), "dtype": str(arr.dtype), "csv_gz_path": rel_str, "content_sha256": artifact["content_sha256"]})
        try:
            s, tau, v = normalize_surface_arrays(arrays)
            surface_rows.append({**occ, "source_commit": occ["branch_commit"], **hints, "shape": f"{tau.size}x{s.size}", "s_min": float(s.min()), "s_max": float(s.max()), "tau_min": float(tau.min()), "tau_max": float(tau.max()), "fields": ";".join(sorted(arrays)), "point_count": int(v.size), "content_sha256": artifact["content_sha256"]})
        except Exception:
            pass
    return array_rows, surface_rows


def _specialized_from_run_metrics(repo_name: str, occ: dict[str, Any], records: list[dict[str, str]], tables: dict[str, list[dict[str, Any]]]) -> None:
    hints = _infer_path(occ["path"])
    family = hints["experiment_family"]
    for record in records:
        method = record.get("model_id") or record.get("method_id") or hints["method_id"]
        dim = _safe_int(record.get("dimension")) or hints["dimension"]
        seed = _safe_int(record.get("seed")) or hints["seed"] or (101 if family.startswith("HD") else None)
        experiment_id = Path(occ["path"]).parent.name
        base = {"repo": repo_name, "branch": occ["branch"], "source_commit": occ["branch_commit"], "experiment_id": experiment_id, "method_id": method, "dimension": dim, "seed": seed, "regime_id": hints["regime_id"]}
        tables["experiment_runs.csv"].append({**base, "experiment_family": family, "variant": record.get("variant", ""), "status": record.get("status", record.get("run_class", "")), "diagnostic_only": record.get("diagnostic_only", ""), "candidate_supported": record.get("candidate_supported", ""), "reference_type": "HD_CN_PSOR_internal" if family == "HD3" else "", "scoring_protocol": "internal_native" if family == "HD3" else ""})
        for field, raw in record.items():
            val = _safe_float(raw)
            if val is None:
                continue
            fl = field.lower()
            if "price" in fl:
                tables["price_metrics.csv"].append({**base, "metric_name": field, "metric_value": val, "reference_type": "HD_CN_PSOR_internal" if family == "HD3" else "", "metric_scope": "internal_diagnostic" if family == "HD3" else ""})
            if "delta" in fl or "gamma" in fl:
                greek_definition = "covariance_weighted_hessian_contraction" if "gamma_contraction" in fl else ("reduced_aggregate" if "reduced" in fl else "component_or_scalar")
                tables["greek_metrics.csv"].append({**base, "metric_name": field, "metric_value": val, "greek_definition": greek_definition, "greek_source": "stored_metric", "reference_type": "HD_CN_PSOR_internal" if family == "HD3" else ""})
            if field in {"permutation_consistency_rmse", "transverse_invariance_rmse", "parameter_count"}:
                tables["structural_metrics.csv"].append({**base, "metric_name": field, "metric_value": val})
            if family == "HD1":
                tables["hd1_metrics.csv"].append({**base, "model_id": method, "metric_name": field, "metric_value": val, "gate_name": "", "gate_pass": ""})
            elif family == "HD2":
                tables["hd2_candidate_metrics.csv"].append({**base, "candidate_id": method, "metric_name": field, "metric_value": val, "gate_name": "", "gate_pass": "", "selection_status": ""})
            elif family == "HD3":
                tables["hd3_metrics.csv"].append({**base, "model_id": method, "metric_name": field, "metric_value": val, "reference_type": "HD_CN_PSOR_internal"})
        tables["boundary_metrics.csv"].append({**base, "threshold": record.get("primary_boundary_tolerance", ""), "reference_count": record.get("reference_boundary_count", ""), "predicted_count": record.get("predicted_boundary_count", ""), "matched_count": record.get("matched_boundary_count", ""), "coverage": record.get("boundary_coverage", ""), "found_rate": record.get("boundary_coverage", ""), "conditional_rmse": record.get("boundary_conditional_rmse", ""), "full_rmse": record.get("boundary_full_rmse", ""), "exercise_f1": record.get("exercise_f1", ""), "boundary_tube_f1": record.get("boundary_tube_f1", ""), "missing_reason": "", "boundary_extraction_method": ""})
        tables["compute_metrics.csv"].append({**base, "training_seconds": record.get("training_seconds", ""), "evaluation_seconds": record.get("evaluation_seconds", ""), "solver_seconds": "", "oracle_seconds": record.get("oracle_seconds", ""), "screening_seconds": "", "confirmation_seconds": "", "wall_clock_seconds": "", "hardware": "", "protocol": "oracle_assisted" if family == "HD3" else ""})


def _specialized_tables(repo: Path, repo_name: str, inventory: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    tables = {name: [] for name in TABLE_HEADERS}
    for occ in inventory["provenance"]:
        path = occ["path"]
        if Path(path).suffix.lower() != ".csv":
            continue
        ref = _branch_ref(repo, occ["branch"])
        try:
            records = _parse_csv_text(_show_text(repo, ref, path))
        except Exception:
            continue
        name = Path(path).name.lower()
        hints = _infer_path(path)
        if name == "run_metrics.csv":
            _specialized_from_run_metrics(repo_name, occ, records, tables)
        if name == "free_boundary.csv":
            for record in records:
                boundary = record.get("S_star_over_K", record.get("boundary_over_K", record.get("predicted_boundary_over_K", "")))
                tables["free_boundary_points.csv"].append({"repo": repo_name, "branch": occ["branch"], "source_commit": occ["branch_commit"], **hints, "tau_over_T": record.get("tau_over_T", record.get("tau", "")), "boundary_over_K": boundary, "boundary_found": record.get("boundary_found", ""), "threshold": record.get("boundary_tolerance", record.get("threshold", "")), "missing_reason": record.get("missing_reason", "")})
        if "price_slice" in name:
            for record in records:
                tables["price_slices.csv"].append({"repo": repo_name, "branch": occ["branch"], "source_commit": occ["branch_commit"], **hints, "S_over_K": record.get("S_over_K", record.get("G_over_K", "")), "tau_over_T": record.get("tau_over_T", ""), "V_over_K": record.get("V_over_K", record.get("V", "")), "payoff_over_K": record.get("payoff_over_K", record.get("payoff", "")), "premium_over_K": record.get("premium_over_K", record.get("premium", "")), "reference_type": REFERENCE_TYPE if "reference" in path.lower() else ""})
        if hints["experiment_family"].startswith("M"):
            method_id = hints["experiment_family"]
            for i, record in enumerate(records):
                for field, raw in record.items():
                    val = _safe_float(raw)
                    if val is not None:
                        row = {"repo": repo_name, "branch": occ["branch"], "source_commit": occ["branch_commit"], "method_id": method_id, "method_name": "", "run_status": "", "metric_name": f"{name}[{i}].{field}", "metric_value": val, "gate_name": "", "gate_threshold": "", "gate_pass": "", "final_decision": ""}
                        tables["campaign_method_results.csv"].append(row)
                        if method_id == "M01":
                            tables["classical_solver_metrics.csv"].append({"repo": repo_name, "branch": occ["branch"], "source_commit": occ["branch_commit"], "method_id": method_id, "regime_id": hints["regime_id"], "metric_name": row["metric_name"], "metric_value": val, "reference_type": ""})
        if "arm_c" in path.lower() or "arm_d" in path.lower():
            arm = "C" if "arm_c" in path.lower() else "D"
            for i, record in enumerate(records):
                for field, raw in record.items():
                    val = _safe_float(raw)
                    if val is not None:
                        tables["ruixi_arm_metrics.csv"].append({"repo": repo_name, "branch": occ["branch"], "source_commit": occ["branch_commit"], "arm": arm, "method_id": f"ARM_{arm}", "regime_id": hints["regime_id"], "seed": hints["seed"] or 101, "metric_name": f"{name}[{i}].{field}", "metric_value": val, "reference_type": REFERENCE_TYPE if "scor" in path.lower() or "metric" in name else ""})
    return tables


def _surface_from_git(repo: Path, ref: str, path: str):
    arrays = _load_npz_bytes(_show_bytes(repo, ref, path))
    return (*normalize_surface_arrays(arrays), arrays)


def _surface_from_bytes(data: bytes):
    arrays = _load_npz_bytes(data)
    return (*normalize_surface_arrays(arrays), arrays)


def _extract_full_reference(archive: Path) -> tuple[bytes, str]:
    if archive.stat().st_size != EXPECTED_ARM_D_BYTES:
        raise ValueError(f"Arm D archive size mismatch: {archive.stat().st_size} != {EXPECTED_ARM_D_BYTES}")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    if archive_sha != EXPECTED_ARM_D_SHA256:
        raise ValueError(f"Arm D archive SHA-256 mismatch: {archive_sha}")
    with zipfile.ZipFile(archive) as zf:
        matches = [n for n in zf.namelist() if n.endswith(f"high_accuracy_reference/{REGIME_ID}.npz")]
        if len(matches) != 1:
            raise ValueError(f"expected one full reference member, found {matches}")
        data = zf.read(matches[0])
    return data, _sha256_bytes(data)


def _internal_hd_rmse(zihan: Path) -> dict[tuple[int, str], float]:
    text = _show_text(zihan, HD_BRANCH, f"{HD_RESULT_ROOT}/run_metrics.csv")
    out = {}
    for row in _parse_csv_text(text):
        d = _safe_int(row.get("dimension")); model = row.get("model_id", "")
        val = _safe_float(row.get("normalized_price_rmse"))
        if d and model and val is not None:
            out[(d, model)] = val
    return out


def _common_price_metric_rows(
    predicted,
    reference,
    *,
    method_id: str,
    dimension,
    seed: int,
    regime_id: str,
    domain_scope: str,
    s_min: float,
    s_max: float,
    tau_min: float,
    tau_max: float,
    grid_points: int,
    interpolation_method: str,
    source_repo: str,
    source_branch: str,
    source_commit: str,
    reference_commit: str,
    reference_artifact_sha256: str,
) -> list[dict[str, Any]]:
    values = price_error_metrics(predicted, reference)
    order = ("rmse", "mae", "p95_abs", "max_abs", "relative_l2")
    base = {
        "method_id": method_id,
        "dimension": "" if dimension in (None, "") else dimension,
        "seed": seed,
        "regime_id": regime_id,
        "domain_scope": domain_scope,
        "reference_type": REFERENCE_TYPE,
        "s_min": s_min,
        "s_max": s_max,
        "tau_min": tau_min,
        "tau_max": tau_max,
        "grid_points": grid_points,
        "interpolation_method": interpolation_method,
        "source_repo": source_repo,
        "source_branch": source_branch,
        "source_commit": source_commit,
        "reference_commit": reference_commit,
        "reference_artifact_sha256": reference_artifact_sha256,
    }
    return [{**base, "metric_name": name, "metric_value": values[name]} for name in order]


def _dirk_metric_rows(zihan: Path, ruixi: Path, arm_d_archive: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    hd_commit = _branch_tip(zihan, HD_BRANCH); ruixi_commit = _branch_tip(ruixi, RUIXI_VIZ_BRANCH)
    compact_ref_path = f"{RUIXI_COMPACT_ROOT}/reference/surface.npz"
    compact_ref_bytes = _show_bytes(ruixi, RUIXI_VIZ_BRANCH, compact_ref_path)
    compact_ref_sha = _sha256_bytes(compact_ref_bytes)
    shared_s, shared_tau, shared_ref, _ = _surface_from_bytes(compact_ref_bytes)
    full_ref_bytes, full_ref_sha = _extract_full_reference(arm_d_archive)
    full_s, full_tau, full_ref, _ = _surface_from_bytes(full_ref_bytes)
    internal = _internal_hd_rmse(zihan)
    metrics: list[dict[str, Any]] = []
    long_price: list[dict[str, Any]] = []
    common_price: list[dict[str, Any]] = []

    def add(method, dimension, seed, scope, internal_rmse="", hd_rmse="", shared_rmse="", s_min="", s_max="", tau_min="", tau_max="", points="", interp="", source_repo=ZIHAN_REPO_NAME, source_branch="exp/hd3-matched-american-put-final-boundary-complete", source_commit=hd_commit, ref_sha=""):
        metrics.append({"method_id": method, "dimension": dimension or "", "seed": seed, "regime_id": REGIME_ID, "internal_cnpsor_price_rmse": internal_rmse, "dirk_price_rmse_hd_domain": hd_rmse, "dirk_price_rmse_shared_domain": shared_rmse, "reference_type": "HD_CN_PSOR_internal" if scope == "internal_diagnostic" else REFERENCE_TYPE, "metric_scope": scope, "s_min": s_min, "s_max": s_max, "tau_min": tau_min, "tau_max": tau_max, "grid_points": points, "interpolation_method": interp, "source_repo": source_repo, "source_branch": source_branch, "source_commit": source_commit, "reference_source": "HD_CN_PSOR_internal" if scope == "internal_diagnostic" else REFERENCE_SOURCE, "reference_commit": "" if scope == "internal_diagnostic" else ruixi_commit, "reference_artifact_sha256": ref_sha})

    common_price.extend(_common_price_metric_rows(
        shared_ref.ravel(), shared_ref.ravel(), method_id="DIRK_REFERENCE", dimension="", seed=101,
        regime_id=REGIME_ID, domain_scope="shared_domain", s_min=float(shared_s[0]), s_max=float(shared_s[-1]),
        tau_min=float(shared_tau[0]), tau_max=float(shared_tau[-1]), grid_points=int(shared_ref.size),
        interpolation_method="self_reference", source_repo=RUIXI_REPO_NAME, source_branch="viz/american-put-pointwise-export",
        source_commit=ruixi_commit, reference_commit=ruixi_commit, reference_artifact_sha256=compact_ref_sha,
    ))

    for arm_dir, method in [("arm_c_seed101", "ARM_C"), ("arm_d_seed101", "ARM_D")]:
        s, tau, v, _ = _surface_from_git(ruixi, RUIXI_VIZ_BRANCH, f"{RUIXI_COMPACT_ROOT}/{arm_dir}/surface.npz")
        interpolated = not (np.array_equal(s, shared_s) and np.array_equal(tau, shared_tau))
        if interpolated:
            v = rectilinear_interpolate(s, tau, v, shared_s, shared_tau)
        rmse = price_rmse(v.ravel().tolist(), shared_ref.ravel().tolist())
        interp = "bilinear_rectilinear" if interpolated else "stored_common_grid"
        add(method, "", 101, "cross_model_shared_domain", shared_rmse=rmse, s_min=shared_s[0], s_max=shared_s[-1], tau_min=shared_tau[0], tau_max=shared_tau[-1], points=shared_ref.size, interp=interp, source_repo=RUIXI_REPO_NAME, source_branch="viz/american-put-pointwise-export", source_commit=ruixi_commit, ref_sha=compact_ref_sha)
        common_price.extend(_common_price_metric_rows(
            v.ravel(), shared_ref.ravel(), method_id=method, dimension="", seed=101, regime_id=REGIME_ID,
            domain_scope="shared_domain", s_min=float(shared_s[0]), s_max=float(shared_s[-1]),
            tau_min=float(shared_tau[0]), tau_max=float(shared_tau[-1]), grid_points=int(shared_ref.size),
            interpolation_method=interp, source_repo=RUIXI_REPO_NAME, source_branch="viz/american-put-pointwise-export",
            source_commit=ruixi_commit, reference_commit=ruixi_commit, reference_artifact_sha256=compact_ref_sha,
        ))
        long_price.append({"repo": RUIXI_REPO_NAME, "branch": "viz/american-put-pointwise-export", "source_commit": ruixi_commit, "experiment_id": "matched_american_put_common_reference", "method_id": method, "dimension": "", "seed": 101, "metric_name": "dirk_price_rmse_shared_domain", "metric_value": rmse, "reference_type": REFERENCE_TYPE, "metric_scope": "cross_model", "regime_id": REGIME_ID})

    for d in (10, 20, 50):
        for model in ("FC_HVI", "DEEPSETS_HVI"):
            path = f"{HD_RESULT_ROOT}/visualization/d{d}/{model}/surface.npz"
            s, tau, v, _ = _surface_from_git(zihan, HD_BRANCH, path)
            if s[0] > HD_DOMAIN[0] + 1e-12 or s[-1] < HD_DOMAIN[1] - 1e-12 or tau[0] > HD_DOMAIN[2] + 1e-12 or tau[-1] < HD_DOMAIN[3] - 1e-12:
                raise ValueError(f"HD surface domain incomplete for {d} {model}: S={s[[0,-1]]}, tau={tau[[0,-1]]}")
            ref_on_hd = rectilinear_interpolate(full_s, full_tau, full_ref, s, tau)
            hd_rmse = price_rmse(v.ravel().tolist(), ref_on_hd.ravel().tolist())
            common_price.extend(_common_price_metric_rows(
                v.ravel(), ref_on_hd.ravel(), method_id=model, dimension=d, seed=101, regime_id=REGIME_ID,
                domain_scope="hd_domain", s_min=float(s[0]), s_max=float(s[-1]), tau_min=float(tau[0]), tau_max=float(tau[-1]),
                grid_points=int(v.size), interpolation_method="DIRK_reference_bilinear_to_native_HD_grid",
                source_repo=ZIHAN_REPO_NAME, source_branch="exp/hd3-matched-american-put-final-boundary-complete",
                source_commit=hd_commit, reference_commit=ruixi_commit, reference_artifact_sha256=full_ref_sha,
            ))
            v_shared = rectilinear_interpolate(s, tau, v, shared_s, shared_tau)
            shared_rmse = price_rmse(v_shared.ravel().tolist(), shared_ref.ravel().tolist())
            common_price.extend(_common_price_metric_rows(
                v_shared.ravel(), shared_ref.ravel(), method_id=model, dimension=d, seed=101, regime_id=REGIME_ID,
                domain_scope="shared_domain", s_min=float(shared_s[0]), s_max=float(shared_s[-1]),
                tau_min=float(shared_tau[0]), tau_max=float(shared_tau[-1]), grid_points=int(shared_ref.size),
                interpolation_method="HD_bilinear_to_verified_Ruixi_compact_grid",
                source_repo=ZIHAN_REPO_NAME, source_branch="exp/hd3-matched-american-put-final-boundary-complete",
                source_commit=hd_commit, reference_commit=ruixi_commit, reference_artifact_sha256=compact_ref_sha,
            ))
            old = internal[(d, model)]
            add(model, d, 101, "internal_diagnostic", internal_rmse=old, s_min=s[0], s_max=s[-1], tau_min=tau[0], tau_max=tau[-1], points=v.size, interp="none")
            add(model, d, 101, "hd_scaling_hd_domain", hd_rmse=hd_rmse, s_min=s[0], s_max=s[-1], tau_min=tau[0], tau_max=tau[-1], points=v.size, interp="DIRK_reference_bilinear_to_native_HD_grid", ref_sha=full_ref_sha)
            add(model, d, 101, "cross_model_shared_domain", shared_rmse=shared_rmse, s_min=shared_s[0], s_max=shared_s[-1], tau_min=shared_tau[0], tau_max=shared_tau[-1], points=shared_ref.size, interp="HD_bilinear_to_verified_Ruixi_compact_grid", ref_sha=compact_ref_sha)
            for metric_name, metric_value, ref_type, metric_scope in [("internal_cnpsor_price_rmse", old, "HD_CN_PSOR_internal", "internal_diagnostic"), ("dirk_price_rmse_hd_domain", hd_rmse, REFERENCE_TYPE, "hd_scaling"), ("dirk_price_rmse_shared_domain", shared_rmse, REFERENCE_TYPE, "cross_model")]:
                long_price.append({"repo": ZIHAN_REPO_NAME, "branch": "exp/hd3-matched-american-put-final-boundary-complete", "source_commit": hd_commit, "experiment_id": "hd3_matched_american_put_boundary_complete", "method_id": model, "dimension": d, "seed": 101, "metric_name": metric_name, "metric_value": metric_value, "reference_type": ref_type, "metric_scope": metric_scope, "regime_id": REGIME_ID})
    return metrics, long_price, common_price


def _merge_tables(target: dict[str, list[dict[str, Any]]], source: dict[str, list[dict[str, Any]]]) -> None:
    for key, rows in source.items():
        if key in target:
            target[key].extend(rows)


def build_consolidated_dataset(zihan_repo: Path, ruixi_repo: Path, arm_d_archive: Path, output: Path) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        shutil.rmtree(output)
    write_empty_output_contract(output)
    z_inv = inventory_repository(zihan_repo, ZIHAN_REPO_NAME)
    r_inv = inventory_repository(ruixi_repo, RUIXI_REPO_NAME)
    source_records = _generic_records(zihan_repo, z_inv) + _generic_records(ruixi_repo, r_inv)
    z_arrays, z_surfaces = _array_exports(zihan_repo, z_inv, output)
    r_arrays, r_surfaces = _array_exports(ruixi_repo, r_inv, output)
    tables = {name: [] for name in TABLE_HEADERS}
    _merge_tables(tables, _specialized_tables(zihan_repo, ZIHAN_REPO_NAME, z_inv))
    _merge_tables(tables, _specialized_tables(ruixi_repo, RUIXI_REPO_NAME, r_inv))
    dirk_rows, dirk_price_rows, common_price_rows = _dirk_metric_rows(zihan_repo, ruixi_repo, arm_d_archive)
    tables["dirk_common_reference_metrics.csv"].extend(dirk_rows)
    tables["dirk_common_reference_price_metrics.csv"].extend(common_price_rows)
    tables["price_metrics.csv"].extend(dirk_price_rows)
    tables["source_artifact_inventory.csv"] = z_inv["artifacts"] + r_inv["artifacts"]
    tables["provenance.csv"] = z_inv["provenance"] + r_inv["provenance"]
    tables["source_records.csv"] = source_records
    tables["array_artifacts.csv"] = z_arrays + r_arrays
    tables["surface_artifacts.csv"] = z_surfaces + r_surfaces
    for filename, header in TABLE_HEADERS.items():
        _write_dict_rows(output / filename, header, tables.get(filename, []))
    manifest = {
        "schema_version": 1,
        "headline_cross_model_metric": "dirk_price_rmse_shared_domain",
        "headline_hd_scaling_metric": "dirk_price_rmse_hd_domain",
        "common_reference_price_metric_set": ["rmse", "mae", "p95_abs", "max_abs", "relative_l2"],
        "shared_domain": list(SHARED_DOMAIN),
        "hd_domain": list(HD_DOMAIN),
        "repositories": {ZIHAN_REPO_NAME: {"branches_scanned": z_inv["branch_count"], "branches": z_inv["branches"]}, RUIXI_REPO_NAME: {"branches_scanned": r_inv["branch_count"], "branches": r_inv["branches"]}},
        "counts": {name: len(rows) for name, rows in tables.items()},
        "reference": {"type": REFERENCE_TYPE, "source": REFERENCE_SOURCE, "arm_d_archive_sha256": EXPECTED_ARM_D_SHA256},
        "notes": ["Raw accepted experiment artifacts were not modified.", "Checkpoints are inventoried but never converted to model weights in CSV.", "CSV/JSON scalars are preserved in source_records.csv as a generic lossless fallback.", "NPZ/NPY numeric arrays are registered in array_artifacts.csv and exported to arrays/*.csv.gz when readable.", "Specialized tables may be empty when no supporting source schema was found; empty never means zero."],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Consolidated experiment CSV data\n\n"
        "This directory is generated from immutable Git/release artifacts across every branch of both SURF project repositories. Identical blobs are deduplicated in `source_artifact_inventory.csv`, while every branch/path occurrence remains in `provenance.csv`.\n\n"
        "## Reference policy\n\n"
        "HD internal CN/PSOR RMSE is retained only as `internal_cnpsor_price_rmse`. Final cross-model price comparison uses `dirk_price_rmse_shared_domain` on S/K=[0.5,1.5], tau/T=[0.01,1]. HD-only scaling uses `dirk_price_rmse_hd_domain` on S/K=[0.2,1.5], tau/T=[0.01,1], both against Ruixi's stored DIRK + Policy Iteration + sinh-grid M=480,N=960 reference. `dirk_common_reference_price_metrics.csv` extends the common-reference scoring to RMSE, MAE, P95 absolute error, maximum absolute error, and relative L2.\n\n"
        "## Generic lossless fallback\n\n"
        "All parseable CSV/JSON scalar fields are preserved in `source_records.csv`. Numeric arrays are indexed in `array_artifacts.csv` and exported under `arrays/` when readable. Checkpoints are inventory-only.\n",
        encoding="utf-8",
    )
    return manifest


def validate_output(output: Path) -> dict[str, Any]:
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["repositories"][ZIHAN_REPO_NAME]["branches_scanned"] < 14:
        raise ValueError("Zihan branch inventory is incomplete")
    if manifest["repositories"][RUIXI_REPO_NAME]["branches_scanned"] < 5:
        raise ValueError("Ruixi branch inventory is incomplete")
    rows = _parse_csv_text((output / "dirk_common_reference_metrics.csv").read_text())
    hd_shared = [r for r in rows if r["metric_scope"] == "cross_model_shared_domain" and r["method_id"] in {"FC_HVI", "DEEPSETS_HVI"}]
    hd_domain = [r for r in rows if r["metric_scope"] == "hd_scaling_hd_domain"]
    low_shared = [r for r in rows if r["metric_scope"] == "cross_model_shared_domain" and r["method_id"] in {"ARM_C", "ARM_D"}]
    if len(hd_shared) != 6 or len(hd_domain) != 6 or len(low_shared) != 2:
        raise ValueError(f"unexpected DIRK metric row counts: hd_shared={len(hd_shared)}, hd_domain={len(hd_domain)}, low_shared={len(low_shared)}")
    for row in hd_shared + hd_domain + low_shared:
        field = "dirk_price_rmse_hd_domain" if row["metric_scope"] == "hd_scaling_hd_domain" else "dirk_price_rmse_shared_domain"
        val = _safe_float(row[field])
        if val is None or val < 0:
            raise ValueError(f"invalid {field} for {row['method_id']} d={row['dimension']}: {row[field]}")

    common = _parse_csv_text((output / "dirk_common_reference_price_metrics.csv").read_text())
    expected_metrics = {"rmse", "mae", "p95_abs", "max_abs", "relative_l2"}
    shared_groups: dict[tuple[str, str], set[str]] = {}
    hd_groups: dict[tuple[str, str], set[str]] = {}
    for row in common:
        value = _safe_float(row.get("metric_value"))
        if value is None or value < 0:
            raise ValueError(f"invalid common-reference price metric: {row}")
        key = (row.get("method_id", ""), row.get("dimension", ""))
        if row.get("domain_scope") == "shared_domain":
            shared_groups.setdefault(key, set()).add(row.get("metric_name", ""))
        elif row.get("domain_scope") == "hd_domain":
            hd_groups.setdefault(key, set()).add(row.get("metric_name", ""))
    expected_shared = {("DIRK_REFERENCE", ""), ("ARM_C", ""), ("ARM_D", ""), *( (model, str(d)) for d in (10,20,50) for model in ("FC_HVI","DEEPSETS_HVI") )}
    expected_hd = {(model, str(d)) for d in (10,20,50) for model in ("FC_HVI","DEEPSETS_HVI")}
    if set(shared_groups) != expected_shared or set(hd_groups) != expected_hd:
        raise ValueError(f"unexpected common-reference metric groups: shared={set(shared_groups)}, hd={set(hd_groups)}")
    if any(names != expected_metrics for names in shared_groups.values()) or any(names != expected_metrics for names in hd_groups.values()):
        raise ValueError("common-reference price metric set is incomplete")
    old_shared_rmse = {(r["method_id"], r["dimension"]): float(r["dirk_price_rmse_shared_domain"]) for r in hd_shared + low_shared}
    new_shared_rmse = {(r["method_id"], r["dimension"]): float(r["metric_value"]) for r in common if r["domain_scope"] == "shared_domain" and r["metric_name"] == "rmse" and r["method_id"] != "DIRK_REFERENCE"}
    for key, old in old_shared_rmse.items():
        if not math.isclose(old, new_shared_rmse[key], rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(f"RMSE compatibility failure for {key}: old={old}, new={new_shared_rmse[key]}")
    return {"dirk_rows": len(rows), "common_price_rows": len(common), "zihan_branches": manifest["repositories"][ZIHAN_REPO_NAME]["branches_scanned"], "ruixi_branches": manifest["repositories"][RUIXI_REPO_NAME]["branches_scanned"]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zihan-repo", type=Path, required=True)
    p.add_argument("--ruixi-repo", type=Path, required=True)
    p.add_argument("--arm-d-archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    if args.verify_only:
        print(json.dumps(validate_output(args.output), indent=2))
        return
    manifest = build_consolidated_dataset(args.zihan_repo, args.ruixi_repo, args.arm_d_archive, args.output)
    print(json.dumps({"counts": manifest["counts"], "validation": validate_output(args.output)}, indent=2))


if __name__ == "__main__":
    main()
