"""Deterministic output writer for independent campaign evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .records import METHOD_RUN_CSV_FIELDS, MethodRunRecord
from .registry import MethodSpec


def _stable_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json(path: Path, data: Any) -> None:
    path.write_text(_stable_json(data), encoding="utf-8")


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


def write_method_outputs(
    method_dir: str | Path,
    spec: MethodSpec,
    record: MethodRunRecord,
    *,
    config: Mapping[str, Any],
    training_history: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Path]:
    """Write the standard evidence set and hash every generated artifact but the hash manifest."""
    if record.method_id != spec.method_id:
        raise ValueError("record method_id must match the MethodSpec")

    output_dir = Path(method_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "config_manifest": output_dir / "config_manifest.json",
        "run_metrics": output_dir / "run_metrics.csv",
        "decision": output_dir / "decision.json",
    }
    _write_json(paths["config_manifest"], {"config": dict(config), "method_spec": spec.to_dict()})
    _write_csv(paths["run_metrics"], [_csv_row(record)], METHOD_RUN_CSV_FIELDS)
    _write_json(
        paths["decision"],
        {
            "baseline_metric_value": record.baseline_metric_value,
            "claim_status": record.claim_status,
            "method_id": record.method_id,
            "phase": record.phase.value,
            "primary_metric": spec.primary_metric,
            "primary_metric_value": record.primary_metric_value,
            "status": record.status.value,
        },
    )
    if training_history is not None:
        paths["training_history"] = output_dir / "training_history.csv"
        _write_csv(paths["training_history"], training_history)

    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths.values(), key=lambda item: item.name)
    }
    paths["artifact_hashes"] = output_dir / "artifact_hashes.json"
    _write_json(paths["artifact_hashes"], {"algorithm": "sha256", "files": hashes})
    return paths

