"""Stage 14: lightweight repository release verification."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

from common import ensure_dir, write_csv, write_json


DEFAULT_REQUIRED_FILES = [
    "README.md",
    "docs/release_handoff.md",
    "docs/artifact_manifest.md",
    "docs/claim_audit.md",
    "docs/reproducibility_checklist.md",
    "reports/final_report.tex",
    "reports/stage13_future_extensions.tex",
    "paper/manuscript_outline.tex",
    "paper/references.bib",
    "results/experiment_14_claim_audit/claim_audit_summary.json",
    "results/experiment_15_future_extensions/future_extensions_summary.json",
]
DEFAULT_FORBIDDEN_TRACKED_FILES = [
    "manuscript_outline.aux",
    "manuscript_outline.log",
    "manuscript_outline.pdf",
    "paper/manuscript_outline.pdf",
]


def run(
    *,
    output_root: str | Path = "results/experiment_16_release_verification",
    required_files: list[str | Path] | None = None,
    tracked_files: list[str] | None = None,
    forbidden_tracked_files: list[str] | None = None,
    stage12_summary: str | Path = "results/experiment_14_claim_audit/claim_audit_summary.json",
    stage13_summary: str | Path = "results/experiment_15_future_extensions/future_extensions_summary.json",
) -> dict[str, Any]:
    out_dir = ensure_dir(output_root)
    required_rows = check_required_files(required_files or DEFAULT_REQUIRED_FILES)
    tracked = tracked_files if tracked_files is not None else git_tracked_files()
    tracked_rows = check_forbidden_tracked_files(
        tracked,
        forbidden_tracked_files or DEFAULT_FORBIDDEN_TRACKED_FILES,
    )
    stage_rows = check_stage_summaries(stage12_summary, stage13_summary)
    summary = {
        "missing_required_files": sum(1 for row in required_rows if row["status"] == "missing"),
        "forbidden_tracked_files": sum(1 for row in tracked_rows if row["status"] == "tracked_forbidden"),
        "stage12_release_gate": _gate_status(stage_rows, "stage12_claim_audit"),
        "stage13_release_gate": _gate_status(stage_rows, "stage13_future_extensions"),
    }
    summary["release_gate"] = "pass" if all(
        [
            summary["missing_required_files"] == 0,
            summary["forbidden_tracked_files"] == 0,
            summary["stage12_release_gate"] == "pass",
            summary["stage13_release_gate"] == "pass",
        ]
    ) else "fail"
    write_csv_with_header(out_dir / "release_required_files.csv", required_rows, ["path", "status"])
    write_csv_with_header(
        out_dir / "release_tracked_generated_files.csv",
        tracked_rows,
        ["path", "status"],
    )
    write_csv_with_header(
        out_dir / "release_stage_summary_checks.csv",
        stage_rows,
        ["stage", "check", "value", "status"],
    )
    write_json(out_dir / "release_verification_summary.json", summary)
    return summary


def check_required_files(paths: list[str | Path]) -> list[dict[str, str]]:
    return [{"path": str(path), "status": "present" if Path(path).exists() else "missing"} for path in paths]


def check_forbidden_tracked_files(tracked_files: list[str], forbidden_files: list[str]) -> list[dict[str, str]]:
    tracked = {path.replace("\\", "/") for path in tracked_files}
    rows = []
    for path in forbidden_files:
        normalized = path.replace("\\", "/")
        rows.append(
            {
                "path": path,
                "status": "tracked_forbidden" if normalized in tracked else "not_tracked",
            }
        )
    return rows


def check_stage_summaries(stage12_path: str | Path, stage13_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stage12 = load_json(stage12_path)
    rows.extend(
        [
            _summary_row("stage12_claim_audit", "forbidden_language_violations", stage12.get("forbidden_language_violations"), 0),
            _summary_row("stage12_claim_audit", "missing_artifacts", stage12.get("missing_artifacts"), 0),
        ]
    )
    stage13 = load_json(stage13_path)
    rows.extend(
        [
            _summary_row("stage13_future_extensions", "invalid_extensions", stage13.get("invalid_extensions"), 0),
            _summary_row("stage13_future_extensions", "language_violations", stage13.get("language_violations"), 0),
            {
                "stage": "stage13_future_extensions",
                "check": "claim_status",
                "value": stage13.get("claim_status"),
                "status": "pass" if stage13.get("claim_status") == "future_work_only" else "fail",
            },
        ]
    )
    return rows


def git_tracked_files() -> list[str]:
    try:
        completed = subprocess.run(
            ["git", "ls-files"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def load_json(path: str | Path) -> dict[str, Any]:
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _summary_row(stage: str, check: str, value: Any, expected: Any) -> dict[str, Any]:
    return {"stage": stage, "check": check, "value": value, "status": "pass" if value == expected else "fail"}


def _gate_status(rows: list[dict[str, Any]], stage: str) -> str:
    stage_rows = [row for row in rows if row["stage"] == stage]
    return "pass" if stage_rows and all(row["status"] == "pass" for row in stage_rows) else "fail"


def write_csv_with_header(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(Path(path).parent)
    if rows:
        write_csv(path, rows)
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
