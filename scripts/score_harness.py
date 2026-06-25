#!/usr/bin/env python3
"""Score the project harness across course-inspired dimensions."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DIMENSIONS = {
    "instructions": [
        "AGENTS.md",
        "docs/harness/startup-readiness-checklist.md",
        "docs/harness/claim-safety-rules.md",
    ],
    "state": [
        "feature_list.json",
        "docs/harness/feature-list.schema.json",
        "progress.md",
        "session-handoff.md",
    ],
    "verification": [
        "init.sh",
        "scripts/init_harness.sh",
        "scripts/verify_harness.sh",
        "scripts/verify_round2_evidence.sh",
        "scripts/verify_report_artifacts.sh",
        "scripts/check_result_schema.py",
        "scripts/validate_feature_list.py",
    ],
    "scope": [
        "docs/harness/worktree-policy.md",
        "docs/harness/subagent-protocol.md",
        "docs/harness/result-schema-policy.md",
        "docs/harness/claim-safety-rules.md",
    ],
    "lifecycle": [
        "docs/harness/clean-state-checklist.md",
        "scripts/check_clean_session.sh",
        "progress.md",
        "session-handoff.md",
        "docs/harness/decision-log.md",
    ],
    "observability": [
        "docs/harness/evaluator-rubric.md",
        "docs/harness/quality-document.md",
        "scripts/score_harness.py",
        "docs/harness/harness_changelog.md",
    ],
}


def score_dimension(name: str, files: list[str]) -> dict:
    present = [path for path in files if (ROOT / path).exists()]
    missing = [path for path in files if not (ROOT / path).exists()]
    score = round(100.0 * len(present) / len(files), 2)
    return {
        "dimension": name,
        "score": score,
        "present": present,
        "missing": missing,
    }


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Harness Score",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Overall score: `{payload['overall_score']}`",
        "",
        "| Dimension | Score | Missing |",
        "|---|---:|---|",
    ]
    for item in payload["dimensions"]:
        missing = ", ".join(item["missing"]) if item["missing"] else "none"
        lines.append(f"| {item['dimension']} | {item['score']} | {missing} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/harness", type=Path)
    args = parser.parse_args(argv)

    dimensions = [score_dimension(name, files) for name, files in DIMENSIONS.items()]
    overall = round(sum(item["score"] for item in dimensions) / len(dimensions), 2)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_score": overall,
        "dimensions": dimensions,
    }

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "harness_score.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(output_dir / "harness_score.md", payload)
    print(f"harness score: {overall} -> {output_dir / 'harness_score.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
