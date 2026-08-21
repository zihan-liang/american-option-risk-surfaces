"""Semantic cleanup for consolidated experiment tables.

This layer operates only on generated consolidated CSVs. It does not modify raw
experiment artifacts or recompute scientific results.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from analysis.consolidated_experiment_data import TABLE_HEADERS


IDENTIFIER_METRIC_NAMES = {
    "dimension",
    "seed",
    "batch_size",
    "steps",
    "oom_attempt_count",
    "points_processed",
}

SEMANTIC_METRIC_TABLES = {
    "hd1_metrics.csv",
    "hd2_candidate_metrics.csv",
    "hd3_metrics.csv",
}

ENVIRONMENT_SCIENTIFIC_FIELDS = (
    "os",
    "python",
    "pytorch",
    "cuda",
    "gpu",
    "cpu",
    "ram",
    "device",
)

_RESULT_FILENAME_TOKEN = re.compile(
    r"(?:^|[_-])(?:metric|metrics|summary|decision|gate|gates)(?:$|[_-])",
    re.IGNORECASE,
)
_CAMPAIGN_CONFIG_PREFIX = re.compile(r"^(m\d{2}_[^.]+\.json)(?:\.|$)", re.IGNORECASE)


def _blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _boolish(value: Any):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "cuda"}:
            return True
        if v in {"false", "0", "no", "cpu"}:
            return False
    return None


def is_identifier_metric(metric_name: Any) -> bool:
    return str(metric_name or "").strip().lower() in IDENTIFIER_METRIC_NAMES


def filter_identifier_metrics(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    removed = 0
    for row in rows:
        if is_identifier_metric(row.get("metric_name")):
            removed += 1
            continue
        kept.append(dict(row))
    return kept, removed


def _campaign_row_is_config_false_positive(row: dict[str, Any]) -> bool:
    metric_name = str(row.get("metric_name") or "")
    match = _CAMPAIGN_CONFIG_PREFIX.match(metric_name)
    if not match:
        return False
    filename = match.group(1)
    stem = filename[:-5] if filename.lower().endswith(".json") else filename
    return _RESULT_FILENAME_TOKEN.search(stem) is None


def filter_campaign_nonresult_rows(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Remove config/spec JSON scalars accidentally classified as campaign results.

    Result-like filenames are recognized by whole filename tokens such as
    metrics, summary, decision, or gate. This deliberately does not treat the
    substring ``gate`` inside ``surrogate`` as a result marker.
    """
    kept: list[dict[str, Any]] = []
    removed = 0
    for row in rows:
        if _campaign_row_is_config_false_positive(row):
            removed += 1
            continue
        kept.append(dict(row))
    return kept, removed


def parse_environment_manifest(
    obj: dict[str, Any],
    source: dict[str, Any],
    path: str,
    hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hints = hints or {}
    nested = obj.get("environment") if isinstance(obj.get("environment"), dict) else {}

    def pick(*names: str):
        for name in names:
            if name in nested and not _blank(nested.get(name)):
                return nested.get(name)
            if name in obj and not _blank(obj.get(name)):
                return obj.get(name)
        return ""

    cuda_available = pick("cuda_available")
    device = pick("device")
    if _blank(device):
        parsed = _boolish(cuda_available)
        if parsed is not None:
            device = "cuda" if parsed else "cpu"

    return {
        **source,
        "experiment_id": Path(path).parent.name,
        "method_id": hints.get("method_id", "") or "",
        "seed": hints.get("seed", "") or "",
        "os": pick("platform", "os"),
        "python": pick("python"),
        "pytorch": pick("torch", "pytorch"),
        "cuda": pick("torch_cuda", "cuda", "cuda_driver"),
        "gpu": pick("gpu_name", "gpu"),
        "cpu": pick("cpu"),
        "ram": pick("ram", "system_ram_bytes"),
        "device": device,
    }


def _field_value(fields: dict[str, str], *names: str) -> str:
    for name in names:
        nested = f"environment.{name}"
        if nested in fields and not _blank(fields[nested]):
            return fields[nested]
        if name in fields and not _blank(fields[name]):
            return fields[name]
    return ""


def _path_hints(path: str) -> dict[str, Any]:
    low = path.lower()
    seed = "101" if any(token in low for token in ("hd1", "hd2", "hd3", "seed101")) else ""
    method = ""
    if "deepsets_hvi" in low:
        method = "DEEPSETS_HVI"
    elif "fc_hvi" in low:
        method = "FC_HVI"
    return {"seed": seed, "method_id": method}


def rebuild_environment_rows_from_source_records(source_records_path: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, str]] = defaultdict(dict)
    with Path(source_records_path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            path = row.get("path", "")
            if not path.endswith("environment_manifest.json") or row.get("record_kind") != "json":
                continue
            key = (
                row.get("repo", ""),
                row.get("branch", ""),
                row.get("source_commit", ""),
                path,
            )
            grouped[key][row.get("field", "")] = row.get("value", "")

    out: list[dict[str, Any]] = []
    for (repo, branch, source_commit, path), fields in sorted(grouped.items()):
        cuda_available = _field_value(fields, "cuda_available")
        device = _field_value(fields, "device")
        if _blank(device):
            parsed = _boolish(cuda_available)
            if parsed is not None:
                device = "cuda" if parsed else "cpu"
        hints = _path_hints(path)
        out.append({
            "repo": repo,
            "branch": branch,
            "source_commit": source_commit,
            "experiment_id": Path(path).parent.name,
            "method_id": hints["method_id"],
            "seed": hints["seed"],
            "os": _field_value(fields, "platform", "os"),
            "python": _field_value(fields, "python"),
            "pytorch": _field_value(fields, "torch", "pytorch"),
            "cuda": _field_value(fields, "torch_cuda", "cuda", "cuda_driver"),
            "gpu": _field_value(fields, "gpu_name", "gpu"),
            "cpu": _field_value(fields, "cpu"),
            "ram": _field_value(fields, "ram", "system_ram_bytes"),
            "device": device,
        })
    return out


def collect_final_status_evidence(source_records_path: Path) -> dict[Any, dict[str, str]]:
    """Collect top-level final-status evidence without inventing missing fields."""
    grouped: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    wanted = {"status", "diagnostic_only", "candidate_supported", "selection_status"}
    with Path(source_records_path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            path = row.get("path", "")
            if not path.endswith("final_status.json") or row.get("record_kind") != "json":
                continue
            field = row.get("field", "")
            if field not in wanted:
                continue
            experiment_id = Path(path).parent.name
            grouped[(row.get("branch", ""), experiment_id)][field] = row.get("value", "")
    return dict(grouped)


def repair_experiment_runs(
    runs: Iterable[dict[str, Any]], evidence: dict[Any, dict[str, Any]]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in runs:
        item = dict(row)
        experiment_id = str(item.get("experiment_id") or "")
        branch = str(item.get("branch") or "")
        ev = evidence.get((branch, experiment_id)) or evidence.get(experiment_id) or {}
        if ev:
            for field in ("status", "diagnostic_only", "candidate_supported"):
                if not _blank(ev.get(field)):
                    item[field] = ev[field]
        out.append(item)
    return out


def enrich_compute_hardware(
    compute_rows: Iterable[dict[str, Any]], environment_rows: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_branch: dict[str, str] = {}
    for env in environment_rows:
        branch = str(env.get("branch") or "")
        gpu = str(env.get("gpu") or "")
        if branch and gpu and branch not in by_branch:
            by_branch[branch] = gpu
    out: list[dict[str, Any]] = []
    for row in compute_rows:
        item = dict(row)
        if _blank(item.get("hardware")):
            gpu = by_branch.get(str(item.get("branch") or ""), "")
            if gpu:
                item["hardware"] = gpu
        out.append(item)
    return out


def semantic_audit_rows(table: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[str] = []
    if table == "environment_runs.csv":
        bad = sum(1 for row in rows if not any(not _blank(row.get(field)) for field in ENVIRONMENT_SCIENTIFIC_FIELDS))
        if bad:
            issues.append(f"{bad} environment rows contain no scientific environment fields")
    if table in SEMANTIC_METRIC_TABLES:
        bad = sum(1 for row in rows if is_identifier_metric(row.get("metric_name")))
        if bad:
            issues.append(f"{bad} identifier fields are mislabeled as metrics")
    if table == "campaign_method_results.csv":
        bad = sum(1 for row in rows if _campaign_row_is_config_false_positive(row))
        if bad:
            issues.append(f"{bad} configuration/spec rows are mislabeled as campaign results")
    if table == "experiment_runs.csv":
        bad = sum(1 for row in rows if _blank(row.get("status")))
        if bad:
            issues.append(f"{bad} experiment rows have blank run status")
    if table == "compute_metrics.csv":
        bad = sum(1 for row in rows if _blank(row.get("hardware")))
        if bad:
            issues.append(f"{bad} compute rows have blank hardware")
    return {
        "status": "needs_review" if issues else "ok",
        "notes": "; ".join(issues),
    }


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write(path: Path, header: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row.get(k, "") for k in header})


def apply_semantic_refinement(output: Path) -> dict[str, Any]:
    output = Path(output)
    removed: dict[str, int] = {}
    for name in sorted(SEMANTIC_METRIC_TABLES):
        path = output / name
        rows, count = filter_identifier_metrics(_read(path))
        _write(path, TABLE_HEADERS[name], rows)
        removed[name] = count

    campaign_rows, campaign_removed = filter_campaign_nonresult_rows(
        _read(output / "campaign_method_results.csv")
    )
    _write(
        output / "campaign_method_results.csv",
        TABLE_HEADERS["campaign_method_results.csv"],
        campaign_rows,
    )

    environment_rows = rebuild_environment_rows_from_source_records(output / "source_records.csv")
    _write(output / "environment_runs.csv", TABLE_HEADERS["environment_runs.csv"], environment_rows)

    final_status_evidence = collect_final_status_evidence(output / "source_records.csv")
    original_runs = _read(output / "experiment_runs.csv")
    repaired_runs = repair_experiment_runs(original_runs, final_status_evidence)
    repaired_count = sum(1 for before, after in zip(original_runs, repaired_runs) if before != after)
    _write(output / "experiment_runs.csv", TABLE_HEADERS["experiment_runs.csv"], repaired_runs)

    original_compute = _read(output / "compute_metrics.csv")
    enriched_compute = enrich_compute_hardware(original_compute, environment_rows)
    hardware_enriched = sum(1 for before, after in zip(original_compute, enriched_compute) if before != after)
    _write(output / "compute_metrics.csv", TABLE_HEADERS["compute_metrics.csv"], enriched_compute)

    return {
        "identifier_metrics_removed": removed,
        "campaign_nonresult_rows_removed": campaign_removed,
        "environment_rows_rebuilt": len(environment_rows),
        "experiment_runs_repaired": repaired_count,
        "compute_hardware_rows_enriched": hardware_enriched,
    }
