"""Stage 12: final claim audit and report-support artifact generation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from common import ensure_dir, load_config, write_csv, write_json


DEFAULT_CONFIG = "configs/claim_audit.yaml"


def run(config_path: str = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg = load_config(config_path)
    out_dir = ensure_dir(cfg.get("output_dir", "results/experiment_14_claim_audit"))
    final_matrix = Path(cfg.get("final_claim_matrix", "results/tables/final_claim_matrix.csv"))
    ensure_dir(final_matrix.parent)

    claim_rows = build_claim_matrix(cfg.get("claims", []))
    write_csv_with_header(final_matrix, claim_rows, CLAIM_FIELDNAMES)
    write_csv_with_header(out_dir / "claim_matrix.csv", claim_rows, CLAIM_FIELDNAMES)

    scan_rows = scan_forbidden_language(
        [Path(path) for path in cfg.get("scan_files", [])],
        cfg.get("forbidden_phrases", []),
    )
    write_csv_with_header(out_dir / "forbidden_language_scan.csv", scan_rows, SCAN_FIELDNAMES)

    artifact_rows = check_required_artifacts([Path(path) for path in cfg.get("required_artifacts", [])])
    write_csv_with_header(out_dir / "missing_artifacts.csv", artifact_rows, ARTIFACT_FIELDNAMES)

    summary = {
        "claims": len(claim_rows),
        "allowed_claims": sum(1 for row in claim_rows if row.get("status", "").startswith(("allowed", "scoped", "passed"))),
        "forbidden_language_violations": sum(1 for row in scan_rows if row["status"] == "violation"),
        "missing_artifacts": sum(1 for row in artifact_rows if row["status"] == "missing"),
        "decision": "manuscript_safe_claims_only",
        "no_new_training_or_solver_runs": True,
    }
    write_json(out_dir / "claim_audit_summary.json", summary)
    return summary


def build_claim_matrix(claim_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in claim_specs:
        evidence_path = Path(spec.get("evidence", ""))
        evidence = load_json_or_status(evidence_path)
        row = {
            "claim_id": spec.get("claim_id", ""),
            "claim": spec.get("claim", ""),
            "evidence": str(evidence_path),
            "evidence_status": "present" if evidence_path.exists() else "missing",
            "status": spec.get("status", ""),
            "allowed_wording": spec.get("allowed_wording", ""),
            "restricted_wording": spec.get("restricted_wording", ""),
            "limitations": spec.get("limitations", ""),
        }
        for metric_name, metric_path in spec.get("metrics", {}).items():
            row[f"metric_{metric_name}"] = value_at_path(evidence, str(metric_path))
        rows.append(row)
    return rows


def scan_forbidden_language(files: list[Path], phrases: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in files:
        if not path.exists():
            rows.append({"file": str(path), "line": "", "phrase": "", "status": "missing_file", "text": ""})
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            lower_line = line.lower()
            for phrase in phrases:
                if phrase.lower() not in lower_line:
                    continue
                status = "allowed_restricted_context" if is_restricted_context(lower_line) else "violation"
                rows.append(
                    {
                        "file": str(path),
                        "line": line_no,
                        "phrase": phrase,
                        "status": status,
                        "text": line.strip(),
                    }
                )
    return rows


def check_required_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.append(
            {
                "artifact": str(path),
                "status": "present" if path.exists() else "missing",
                "bytes": path.stat().st_size if path.exists() and path.is_file() else "",
            }
        )
    return rows


def load_json_or_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing"}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def value_at_path(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return "missing"
    if isinstance(current, (dict, list)):
        return json.dumps(current, sort_keys=True)
    return current


def is_restricted_context(lower_line: str) -> bool:
    markers = [
        "do not",
        "does not",
        "not ",
        "not a",
        "not as",
        "no ",
        "forbidden",
        "restricted",
        "cannot",
        "must not",
        "is not",
    ]
    return any(marker in lower_line for marker in markers)


def write_csv_with_header(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(Path(path).parent)
    if rows:
        write_csv(path, rows)
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()


CLAIM_FIELDNAMES = [
    "claim_id",
    "claim",
    "evidence",
    "evidence_status",
    "status",
    "allowed_wording",
    "restricted_wording",
    "limitations",
]
SCAN_FIELDNAMES = ["file", "line", "phrase", "status", "text"]
ARTIFACT_FIELDNAMES = ["artifact", "status", "bytes"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(run(args.config))
