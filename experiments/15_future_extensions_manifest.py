"""Stage 13: emit a future-extension manifest without making result claims."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from common import ensure_dir, load_config, write_csv, write_json


DEFAULT_CONFIG = "configs/future_extensions.yaml"
REQUIRED_FIELDS = [
    "extension_id",
    "extension_name",
    "motivation",
    "current_limitation_addressed",
    "required_new_data_or_solver_work",
    "required_validation",
    "expected_outputs",
    "risk_level",
    "claim_status",
]


def run(config_path: str = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg = load_config(config_path)
    out_dir = ensure_dir(cfg.get("output_dir", "results/experiment_15_future_extensions"))
    extensions = [normalize_extension(row) for row in cfg.get("extensions", [])]
    validation_rows = build_validation_matrix(extensions)
    scan_rows = scan_future_claim_language([Path(path) for path in cfg.get("scan_files", [])])
    summary = validate_extensions(extensions)
    summary["language_violations"] = sum(1 for row in scan_rows if row["status"] == "violation")
    summary["no_completed_extension_claims"] = summary["invalid_extensions"] == 0
    summary["output_dir"] = str(out_dir)

    write_csv_with_header(out_dir / "future_extensions.csv", extensions, REQUIRED_FIELDS)
    write_csv_with_header(out_dir / "future_extension_validation_matrix.csv", validation_rows, VALIDATION_FIELDNAMES)
    write_csv_with_header(out_dir / "future_extension_language_scan.csv", scan_rows, LANGUAGE_FIELDNAMES)
    write_json(out_dir / "future_extensions_summary.json", summary)
    return summary


def normalize_extension(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {field: str(row.get(field, "")).strip() for field in REQUIRED_FIELDS}
    return normalized


def validate_extensions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    invalid = 0
    for row in rows:
        missing_required = any(not str(row.get(field, "")).strip() for field in REQUIRED_FIELDS)
        wrong_status = row.get("claim_status") != "future_work_only"
        completed_language = row.get("claim_status", "").lower() in {"completed", "validated", "implemented"}
        if missing_required or wrong_status or completed_language:
            invalid += 1
    return {
        "extensions": len(rows),
        "claim_status": "future_work_only" if invalid == 0 else "future_work_only_with_invalid_rows",
        "invalid_extensions": invalid,
    }


def build_validation_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    for row in rows:
        matrix.append(
            {
                "extension_id": row["extension_id"],
                "extension_name": row["extension_name"],
                "current_limitation_addressed": row["current_limitation_addressed"],
                "required_new_data_or_solver_work": row["required_new_data_or_solver_work"],
                "required_validation": row["required_validation"],
                "expected_outputs": row["expected_outputs"],
                "risk_level": row["risk_level"],
                "claim_status": row["claim_status"],
                "implementation_status": "not_started",
            }
        )
    return matrix


def scan_future_claim_language(files: list[Path]) -> list[dict[str, Any]]:
    risky_phrases = [
        "has validated production performance",
        "improves the surrogate",
        "is implemented",
        "is validated",
        "achieves",
        "solves",
    ]
    safe_markers = [
        "future work",
        "future-work",
        "requires",
        "would",
        "should",
        "not completed",
        "not implemented",
        "not validated",
        "before any claim",
        "avoid",
        "forbidden",
        "restricted",
    ]
    rows: list[dict[str, Any]] = []
    for path in files:
        if not path.exists():
            rows.append({"file": str(path), "line": "", "phrase": "", "status": "missing_file", "text": ""})
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            lower = line.lower()
            for phrase in risky_phrases:
                if phrase in lower:
                    status = "allowed_future_context" if any(marker in lower for marker in safe_markers) else "violation"
                    rows.append(
                        {
                            "file": str(path),
                            "line": line_no,
                            "phrase": phrase,
                            "status": status,
                            "text": line.strip(),
                        }
                    )
                    break
    return rows


def write_csv_with_header(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(Path(path).parent)
    if rows:
        write_csv(path, rows)
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()


VALIDATION_FIELDNAMES = [
    "extension_id",
    "extension_name",
    "current_limitation_addressed",
    "required_new_data_or_solver_work",
    "required_validation",
    "expected_outputs",
    "risk_level",
    "claim_status",
    "implementation_status",
]
LANGUAGE_FIELDNAMES = ["file", "line", "phrase", "status", "text"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, sort_keys=True))
