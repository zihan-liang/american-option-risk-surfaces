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

METADATA_METRIC_TERMINALS = {
    "seed",
    "seeds",
    "training_seed",
    "training_seeds",
    "sample_count",
    "scenario_count",
    "batch_size",
    "steps",
    "dimension",
    "points_processed",
    "oom_attempt_count",
}

SEMANTIC_METRIC_TABLES = {
    "hd1_metrics.csv",
    "hd2_candidate_metrics.csv",
    "hd3_metrics.csv",
}

FINAL_HEADER_OVERRIDES = {
    "greek_metrics.csv": [
        "repo", "branch", "source_commit", "experiment_id", "method_id",
        "dimension", "seed", "regime_id", "metric_name", "metric_value",
        "greek_definition", "greek_source", "reference_type",
    ],
    "boundary_metrics.csv": [
        "repo", "branch", "source_commit", "experiment_id", "method_id",
        "dimension", "seed", "regime_id", "threshold", "reference_count",
        "predicted_count", "matched_count", "coverage", "found_rate",
        "conditional_rmse", "full_rmse", "exercise_f1", "boundary_tube_f1",
        "missing_reason", "boundary_extraction_method",
    ],
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
_INDEX_SUFFIX = re.compile(r"\[\d+\]$")


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


def final_table_header(name: str) -> list[str]:
    """Return the final user-facing schema for a consolidated table."""
    return list(FINAL_HEADER_OVERRIDES.get(name, TABLE_HEADERS[name]))


def metric_terminal_name(metric_name: Any) -> str:
    """Return the semantic terminal field encoded in a flattened metric name."""
    text = str(metric_name or "").strip()
    if ":" in text:
        text = text.rsplit(":", 1)[-1]
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    text = _INDEX_SUFFIX.sub("", text)
    return text.lower()


def is_identifier_metric(metric_name: Any) -> bool:
    return str(metric_name or "").strip().lower() in IDENTIFIER_METRIC_NAMES


def is_metadata_metric(metric_name: Any) -> bool:
    return metric_terminal_name(metric_name) in METADATA_METRIC_TERMINALS


def filter_identifier_metrics(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    removed = 0
    for row in rows:
        if is_identifier_metric(row.get("metric_name")):
            removed += 1
            continue
        kept.append(dict(row))
    return kept, removed


def filter_metadata_metric_rows(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop context/identifier scalars that are not scientific result metrics."""
    kept: list[dict[str, Any]] = []
    removed = 0
    for row in rows:
        if is_metadata_metric(row.get("metric_name")):
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


def rebuild_campaign_experiment_runs(
    existing_runs: Iterable[dict[str, Any]],
    campaign_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rebuild CAMPAIGN run provenance only from filtered result evidence."""
    retained = [dict(row) for row in existing_runs if row.get("experiment_family") != "CAMPAIGN"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in campaign_rows:
        method = str(row.get("method_id") or "").strip()
        if method:
            grouped[method].append(dict(row))

    def priority(row: dict[str, Any]) -> tuple[int, str, str]:
        branch = str(row.get("branch") or "")
        if branch == "campaign/independent-methods":
            rank = 0
        elif branch.startswith("tmp/"):
            rank = 2
        else:
            rank = 1
        return rank, branch, str(row.get("source_commit") or "")

    for method in sorted(grouped):
        source = sorted(grouped[method], key=priority)[0]
        retained.append({
            "repo": source.get("repo", ""),
            "branch": source.get("branch", ""),
            "source_commit": source.get("source_commit", ""),
            "experiment_family": "CAMPAIGN",
            "experiment_id": method,
            "method_id": method,
            "variant": "",
            "dimension": "",
            "seed": "",
            "status": source.get("run_status") or "recorded",
            "diagnostic_only": "",
            "candidate_supported": "",
            "reference_type": "",
            "scoring_protocol": "campaign",
        })
    return retained


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


def _metric_scope_and_terminal(metric_name: Any) -> tuple[str, str]:
    text = str(metric_name or "")
    if ":" in text:
        scope, terminal = text.rsplit(":", 1)
        return scope, terminal
    return "", text


def rebuild_ruixi_greek_rows(arm_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild Arm C/D Greek rows while retaining per-regime identity."""
    out: list[dict[str, Any]] = []
    for row in arm_rows:
        method = str(row.get("method_id") or "")
        if method not in {"ARM_C", "ARM_D"}:
            continue
        _, terminal = _metric_scope_and_terminal(row.get("metric_name"))
        field = terminal.lower()
        if "delta" not in field and "gamma" not in field:
            continue
        out.append({
            "repo": row.get("repo", ""),
            "branch": row.get("branch", ""),
            "source_commit": row.get("source_commit", ""),
            "experiment_id": "ruixi_heldout_pilot",
            "method_id": method,
            "dimension": "",
            "seed": row.get("seed", ""),
            "regime_id": row.get("regime_id", ""),
            "metric_name": row.get("metric_name", ""),
            "metric_value": row.get("metric_value", ""),
            "greek_definition": "scaled_gamma" if "scaled_gamma" in field else "scalar_delta",
            "greek_source": "stored_scoring",
            "reference_type": row.get("reference_type", ""),
        })
    return out


def rebuild_ruixi_boundary_rows(arm_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild one boundary summary per Arm/method/regime from region=all rows."""
    grouped: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in arm_rows:
        method = str(row.get("method_id") or "")
        if method not in {"ARM_C", "ARM_D"}:
            continue
        scope, terminal = _metric_scope_and_terminal(row.get("metric_name"))
        if "region=all" not in scope:
            continue
        field = terminal.lower()
        if field not in {
            "boundary_reference_found",
            "boundary_predicted_found",
            "boundary_found_rate",
            "boundary_conditional_rmse",
        }:
            continue
        key = (
            str(row.get("repo") or ""),
            str(row.get("branch") or ""),
            str(row.get("source_commit") or ""),
            method,
            str(row.get("regime_id") or ""),
            str(row.get("seed") or ""),
            scope,
        )
        item = grouped.setdefault(key, {
            "repo": row.get("repo", ""),
            "branch": row.get("branch", ""),
            "source_commit": row.get("source_commit", ""),
            "experiment_id": "ruixi_heldout_pilot",
            "method_id": method,
            "dimension": "",
            "seed": row.get("seed", ""),
            "regime_id": row.get("regime_id", ""),
            "threshold": "",
            "reference_count": "",
            "predicted_count": "",
            "matched_count": "",
            "coverage": "",
            "found_rate": "",
            "conditional_rmse": "",
            "full_rmse": "",
            "exercise_f1": "",
            "boundary_tube_f1": "",
            "missing_reason": "",
            "boundary_extraction_method": "stored_scoring",
        })
        value = row.get("metric_value", "")
        if field == "boundary_reference_found":
            item["reference_count"] = value
        elif field == "boundary_predicted_found":
            item["predicted_count"] = value
        elif field == "boundary_found_rate":
            item["coverage"] = value
            item["found_rate"] = value
        elif field == "boundary_conditional_rmse":
            item["conditional_rmse"] = value
    return list(grouped.values())


def _price_regime_map(price_rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str, str], str]:
    mapping: dict[tuple[str, str, str, str], str] = {}
    for row in price_rows:
        regime = str(row.get("regime_id") or "")
        if not regime:
            continue
        key = (
            str(row.get("experiment_id") or ""),
            str(row.get("method_id") or ""),
            str(row.get("dimension") or ""),
            str(row.get("seed") or ""),
        )
        mapping.setdefault(key, regime)
    return mapping


def enrich_regime_from_price_evidence(
    rows: Iterable[dict[str, Any]], price_rows: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    mapping = _price_regime_map(price_rows)
    out: list[dict[str, Any]] = []
    filled = 0
    for row in rows:
        item = dict(row)
        if _blank(item.get("regime_id")):
            key = (
                str(item.get("experiment_id") or ""),
                str(item.get("method_id") or ""),
                str(item.get("dimension") or ""),
                str(item.get("seed") or ""),
            )
            regime = mapping.get(key, "")
            if regime:
                item["regime_id"] = regime
                filled += 1
        out.append(item)
    return out, filled


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
    if table in {"ruixi_arm_metrics.csv", "classical_solver_metrics.csv", "campaign_method_results.csv"}:
        bad = sum(1 for row in rows if is_metadata_metric(row.get("metric_name")))
        if bad:
            issues.append(f"{bad} context/metadata fields are mislabeled as result metrics")
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
    if table in {"greek_metrics.csv", "boundary_metrics.csv"}:
        bad = sum(
            1 for row in rows
            if row.get("method_id") in {"ARM_C", "ARM_D"} and _blank(row.get("regime_id"))
        )
        if bad:
            issues.append(f"{bad} Arm C/D rows have lost per-regime identity")
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
        _write(path, final_table_header(name), rows)
        removed[name] = count

    metadata_removed: dict[str, int] = {}

    arm_rows, arm_removed = filter_metadata_metric_rows(_read(output / "ruixi_arm_metrics.csv"))
    metadata_removed["ruixi_arm_metrics.csv"] = arm_removed
    _write(output / "ruixi_arm_metrics.csv", final_table_header("ruixi_arm_metrics.csv"), arm_rows)

    classical_rows, classical_removed = filter_metadata_metric_rows(_read(output / "classical_solver_metrics.csv"))
    metadata_removed["classical_solver_metrics.csv"] = classical_removed
    _write(output / "classical_solver_metrics.csv", final_table_header("classical_solver_metrics.csv"), classical_rows)

    campaign_rows, campaign_config_removed = filter_campaign_nonresult_rows(
        _read(output / "campaign_method_results.csv")
    )
    campaign_rows, campaign_metadata_removed = filter_metadata_metric_rows(campaign_rows)
    metadata_removed["campaign_method_results.csv"] = campaign_metadata_removed
    _write(
        output / "campaign_method_results.csv",
        final_table_header("campaign_method_results.csv"),
        campaign_rows,
    )

    price_rows = _read(output / "price_metrics.csv")

    existing_greek = [row for row in _read(output / "greek_metrics.csv") if row.get("method_id") not in {"ARM_C", "ARM_D"}]
    existing_greek, greek_regime_filled = enrich_regime_from_price_evidence(existing_greek, price_rows)
    ruixi_greek = rebuild_ruixi_greek_rows(arm_rows)
    final_greek = existing_greek + ruixi_greek
    _write(output / "greek_metrics.csv", final_table_header("greek_metrics.csv"), final_greek)

    existing_boundary = [row for row in _read(output / "boundary_metrics.csv") if row.get("method_id") not in {"ARM_C", "ARM_D"}]
    existing_boundary, boundary_regime_filled = enrich_regime_from_price_evidence(existing_boundary, price_rows)
    ruixi_boundary = rebuild_ruixi_boundary_rows(arm_rows)
    final_boundary = existing_boundary + ruixi_boundary
    _write(output / "boundary_metrics.csv", final_table_header("boundary_metrics.csv"), final_boundary)

    environment_rows = rebuild_environment_rows_from_source_records(output / "source_records.csv")
    _write(output / "environment_runs.csv", final_table_header("environment_runs.csv"), environment_rows)

    original_runs = _read(output / "experiment_runs.csv")
    campaign_rebuilt_runs = rebuild_campaign_experiment_runs(original_runs, campaign_rows)
    campaign_run_count = sum(1 for row in campaign_rebuilt_runs if row.get("experiment_family") == "CAMPAIGN")
    final_status_evidence = collect_final_status_evidence(output / "source_records.csv")
    repaired_runs = repair_experiment_runs(campaign_rebuilt_runs, final_status_evidence)
    repaired_count = sum(1 for before, after in zip(campaign_rebuilt_runs, repaired_runs) if before != after)
    _write(output / "experiment_runs.csv", final_table_header("experiment_runs.csv"), repaired_runs)

    original_compute = _read(output / "compute_metrics.csv")
    enriched_compute = enrich_compute_hardware(original_compute, environment_rows)
    hardware_enriched = sum(1 for before, after in zip(original_compute, enriched_compute) if before != after)
    _write(output / "compute_metrics.csv", final_table_header("compute_metrics.csv"), enriched_compute)

    return {
        "identifier_metrics_removed": removed,
        "metadata_metric_rows_removed": metadata_removed,
        "campaign_nonresult_rows_removed": campaign_config_removed,
        "campaign_experiment_runs_rebuilt": campaign_run_count,
        "environment_rows_rebuilt": len(environment_rows),
        "experiment_runs_repaired": repaired_count,
        "compute_hardware_rows_enriched": hardware_enriched,
        "ruixi_greek_rows_rebuilt": len(ruixi_greek),
        "ruixi_boundary_rows_rebuilt": len(ruixi_boundary),
        "greek_regime_ids_enriched_from_price_evidence": greek_regime_filled,
        "boundary_regime_ids_enriched_from_price_evidence": boundary_regime_filled,
    }
