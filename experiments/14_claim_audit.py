"""Stage 13: create a claim audit table from available experiment summaries."""

from __future__ import annotations

import json
from pathlib import Path

from common import ensure_dir, write_csv, write_json


def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"status": "missing"}
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def run() -> dict:
    out_dir = ensure_dir("results/experiment_14_claim_audit")
    rows = [
        {
            "claim": "validated CN/PSOR benchmark",
            "evidence": "results/experiment_01_solver_validation/validation_summary.json",
            "status": "ready_after_metrics_review",
            "notes": str(_load("results/experiment_01_solver_validation/validation_summary.json")),
        },
        {
            "claim": "positive-premium price surface",
            "evidence": "results/experiment_07_positive_premium_tuning/positive_premium_summary.json",
            "status": "scoped_to_tested_regimes",
            "notes": str(_load("results/experiment_07_positive_premium_tuning/positive_premium_summary.json")),
        },
        {
            "claim": "direct boundary diagnostics",
            "evidence": "results/experiment_09_boundary_head/boundary_summary.json",
            "status": "scoped_diagnostic",
            "notes": str(_load("results/experiment_09_boundary_head/boundary_summary.json")),
        },
        {
            "claim": "bounded supervised Delta diagnostics",
            "evidence": "results/experiment_12_supervised_delta_head/delta_head_summary.json",
            "status": "separate_head_only",
            "notes": str(_load("results/experiment_12_supervised_delta_head/delta_head_summary.json")),
        },
    ]
    write_csv("results/tables/final_claim_matrix.csv", rows)
    write_csv(out_dir / "claim_matrix.csv", rows)
    summary = {"claims": len(rows), "forbidden_claims": ["one network solves all", "price surrogate is a Greek engine", "surrogate replaces PSOR"]}
    write_json(out_dir / "claim_audit_summary.json", summary)
    return summary


if __name__ == "__main__":
    print(run())
