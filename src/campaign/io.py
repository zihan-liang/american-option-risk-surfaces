"""Deterministic output writer for independent campaign evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .records import METHOD_RUN_CSV_FIELDS, MethodRunRecord, RunStatus
from .registry import MethodSpec


def _stable_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json(path: Path, data: Any) -> None:
    path.write_text(_stable_json(data), encoding="utf-8")


def _config_sha256(config: Mapping[str, Any]) -> str:
    canonical = json.dumps(dict(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: tuple[str, ...] | None = None) -> None:
    materialized_rows = list(rows)
    if fieldnames is None:
        fieldnames = tuple(sorted({key for row in materialized_rows for key in row}))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized_rows)


def _csv_row(record: MethodRunRecord) -> dict[str, Any]:
    payload = record.to_dict()
    return {
        "method_id": payload["method_id"],
        "phase": payload["phase"],
        "status": payload["status"],
        "config_id": payload["config_id"],
        "primary_metric_value": payload["primary_metric_value"],
        "baseline_metric_value": payload["baseline_metric_value"],
        "optional_metrics_json": json.dumps(payload["optional_metrics"], sort_keys=True, allow_nan=False),
        "scenario_count": payload["scenario_count"],
        "seed": payload["seed"],
        "split": payload["split"],
        "branch": payload["branch"],
        "commit": payload["commit"],
        "timings_seconds_json": json.dumps(payload["timings_seconds"], sort_keys=True, allow_nan=False),
        "memory_mb": payload["memory_mb"],
        "claim_status": payload["claim_status"],
        "notes": payload["notes"],
    }


def _normalize_records(
    records: MethodRunRecord | Sequence[MethodRunRecord] | None,
    compatibility_record: MethodRunRecord | None,
    spec: MethodSpec,
    decision_record: MethodRunRecord | None,
) -> tuple[tuple[MethodRunRecord, ...], MethodRunRecord]:
    if records is not None and compatibility_record is not None:
        raise ValueError("pass either records or record, not both")
    supplied = records if records is not None else compatibility_record
    if supplied is None:
        raise ValueError("at least one MethodRunRecord is required")
    if isinstance(supplied, MethodRunRecord):
        normalized = (supplied,)
    elif isinstance(supplied, Sequence):
        normalized = tuple(supplied)
    else:
        raise ValueError("records must be a MethodRunRecord or a sequence of MethodRunRecord values")
    if not normalized:
        raise ValueError("records must not be empty")
    if any(not isinstance(current, MethodRunRecord) for current in normalized):
        raise ValueError("records must contain only MethodRunRecord values")
    if any(current.method_id != spec.method_id for current in normalized):
        raise ValueError("every record method_id must match the MethodSpec")
    phase = normalized[0].phase
    if any(current.phase != phase for current in normalized):
        raise ValueError("every record must use the same phase")

    if decision_record is None:
        if len(normalized) != 1:
            raise ValueError("decision_record is required when writing multiple records")
        selected = normalized[0]
    else:
        if not isinstance(decision_record, MethodRunRecord):
            raise ValueError("decision_record must be a MethodRunRecord")
        if decision_record not in normalized:
            raise ValueError("decision_record must be one of the supplied records")
        selected = decision_record
    return normalized, selected


def write_method_outputs(
    method_dir: str | Path,
    spec: MethodSpec,
    records: MethodRunRecord | Sequence[MethodRunRecord] | None = None,
    *,
    config: Mapping[str, Any],
    decision_record: MethodRunRecord | None = None,
    record: MethodRunRecord | None = None,
    training_history: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Path]:
    """Write the standard evidence set and hash every generated artifact but the hash manifest."""
    run_records, selected_record = _normalize_records(records, record, spec, decision_record)

    output_dir = Path(method_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_rows = None if training_history is None else list(training_history)
    requires_history = "training_history.csv" in spec.required_outputs
    if history_rows == []:
        raise ValueError("training_history must be non-empty when supplied")
    if any(current.status is RunStatus.COMPLETED for current in run_records) and requires_history and history_rows is None:
        raise ValueError("training_history is required for completed runs of this method")

    paths = {
        "config_manifest": output_dir / "config_manifest.json",
        "run_metrics": output_dir / "run_metrics.csv",
        "decision": output_dir / "decision.json",
    }
    _write_json(
        paths["config_manifest"],
        {
            "config": dict(config),
            "config_sha256": _config_sha256(config),
            "method_spec": spec.to_dict(),
            "provenance": {"branch": selected_record.branch, "commit": selected_record.commit},
        },
    )
    _write_csv(paths["run_metrics"], [_csv_row(current) for current in run_records], METHOD_RUN_CSV_FIELDS)
    _write_json(
        paths["decision"],
        {
            "baseline_metric_value": selected_record.baseline_metric_value,
            "claim_status": selected_record.claim_status,
            "decision_record": selected_record.to_dict(),
            "method_id": selected_record.method_id,
            "phase": selected_record.phase.value,
            "primary_metric": spec.primary_metric,
            "primary_metric_value": selected_record.primary_metric_value,
            "record_count": len(run_records),
            "status": selected_record.status.value,
        },
    )
    history_path = output_dir / "training_history.csv"
    if history_rows is not None:
        paths["training_history"] = output_dir / "training_history.csv"
        _write_csv(paths["training_history"], history_rows)
    elif history_path.exists():
        history_path.unlink()

    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths.values(), key=lambda item: item.name)
    }
    paths["artifact_hashes"] = output_dir / "artifact_hashes.json"
    _write_json(paths["artifact_hashes"], {"algorithm": "sha256", "files": hashes})
    return paths
