"""Stage 15: publication-readiness evidence and claim-scope audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from common import ensure_dir, load_config, write_csv, write_json


DEFAULT_CONFIG = "configs/publication_evidence.yaml"


def run(config_path: str = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg = load_config(config_path)
    out_dir = ensure_dir(cfg.get("output_dir", "results/experiment_17_publication_audit"))

    claim_rows = _publication_gap_matrix(_read_csv(cfg["claim_matrix"]), cfg)
    integrated_rows = _integrated_scope_rows(cfg)
    baseline_rows = _baseline_comparison_rows(cfg)
    boundary_rows = _boundary_failure_rows(cfg)
    benchmark_rows = _numerical_benchmark_rows(cfg)

    write_csv(out_dir / "publication_gap_matrix.csv", claim_rows)
    write_csv(out_dir / "integrated_scope_summary.csv", integrated_rows)
    write_csv(out_dir / "baseline_comparison.csv", baseline_rows)
    write_csv(out_dir / "boundary_failure_analysis.csv", boundary_rows)
    write_csv(out_dir / "numerical_benchmark_summary.csv", benchmark_rows)
    write_json(out_dir / "publication_config_manifest.json", cfg)

    demoted = [
        row["claim_id"]
        for row in claim_rows
        if row["readiness"] in {"supplement_only", "needs_more_evidence", "reject_or_remove"}
    ]
    integrated_summary = _read_json(cfg["integrated_summary"])
    target = int(cfg.get("target_integrated_test_regimes", 34))
    summary = {
        "decision": "journal_style_with_scoped_claims",
        "claims": len(claim_rows),
        "main_claim_ready": sum(1 for row in claim_rows if row["readiness"] == "main_claim_ready"),
        "supplement_only": sum(1 for row in claim_rows if row["readiness"] == "supplement_only"),
        "needs_more_evidence": sum(1 for row in claim_rows if row["readiness"] == "needs_more_evidence"),
        "demoted_or_limited_claims": demoted,
        "integrated_audited_cases": int(integrated_summary.get("cases", 0)),
        "target_integrated_test_regimes": target,
        "manuscript_claim_policy": "main text uses scoped claims; weak boundary and six-case integrated audit are explicit limitations",
        "no_new_model_architectures": True,
    }
    write_json(out_dir / "publication_audit_summary.json", summary)
    return summary


def scan_claim_language(text_by_file: dict[str, str], phrases: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_name, text in text_by_file.items():
        for line_no, line in enumerate(text.splitlines(), start=1):
            lower_line = line.lower()
            for phrase in phrases:
                if phrase.lower() in lower_line:
                    rows.append(
                        {
                            "file": file_name,
                            "line": line_no,
                            "phrase": phrase,
                            "status": "allowed_restricted_context" if _restricted_context(lower_line) else "violation",
                            "text": line.strip(),
                        }
                    )
    return rows


def scan_files_for_claim_language(paths: list[Path], phrases: list[str]) -> list[dict[str, Any]]:
    texts = {str(path): path.read_text(encoding="utf-8") for path in paths if path.exists()}
    return scan_claim_language(texts, phrases)


def _publication_gap_matrix(rows: list[dict[str, str]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    readiness = {
        "cn_psor_benchmark": ("main_claim_ready", "Validated engineering-scale benchmark; avoid universal accuracy language."),
        "grid_domain_runtime": ("main_claim_ready", "Supports tested-grid stability; keep analytic-reference limits explicit."),
        "application_surfaces": ("main_claim_ready", "Broad audited sweep with delicate regimes retained."),
        "surrogate_dataset": ("supplement_only", "Dataset governance supports methods and supplement, not a standalone result claim."),
        "direct_price_baselines": ("supplement_only", "Negative evidence; use to motivate positive-premium construction."),
        "positive_premium_price": ("main_claim_ready", "Hard obstacle construction and held-out metrics support main result."),
        "boundary_head": ("supplement_only", "Boundary errors remain materially weaker; present as diagnostic limitation."),
        "price_delta_audit": ("supplement_only", "Derivative audit motivates separate Delta head; not final contribution."),
        "bounded_delta_head": ("main_claim_ready", "Strict-mask supervised Delta evidence supports scoped main result."),
        "integrated_workflow": ("supplement_only", "Six fresh cases are not enough for a top-journal primary claim."),
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        claim_id = row.get("claim_id", "")
        state, rationale = readiness.get(claim_id, ("needs_more_evidence", "No publication-readiness rule configured."))
        out.append(
            {
                "claim_id": claim_id,
                "claim": row.get("claim", ""),
                "readiness": state,
                "recommended_manuscript_placement": "main_text" if state == "main_claim_ready" else "supplement_or_limitations",
                "evidence": row.get("evidence", ""),
                "allowed_wording": row.get("allowed_wording", ""),
                "restricted_wording": row.get("restricted_wording", ""),
                "publication_rationale": rationale,
                "limitation": row.get("limitations", ""),
            }
        )
    return out


def _integrated_scope_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    integrated = _read_json(cfg["integrated_summary"])
    by_regime = _read_csv(cfg["integrated_by_regime"])
    split = _read_csv(cfg["split_table"])
    target = int(cfg.get("target_integrated_test_regimes", 34))
    test_regimes = [row for row in split if row.get("split") == "test"]
    audited_cases = int(integrated.get("cases", len(by_regime)))
    skipped = max(min(target, len(test_regimes)) - audited_cases, 0)
    rows: list[dict[str, Any]] = [
        {
            "scope": "fresh_psor_integrated_audit",
            "target_test_regimes": target,
            "available_test_regimes": len(test_regimes),
            "audited_cases": audited_cases,
            "skipped_cases": skipped,
            "disposition": "supplement_only" if skipped else "main_text_candidate",
            "reason": "Fresh integrated audit remains six cases; manuscript must not make broad workflow-generalization claims."
            if skipped
            else "All requested test regimes were audited.",
        }
    ]
    for option_type in sorted({row.get("option_type", "") for row in by_regime if row.get("option_type")}):
        option_rows = [row for row in by_regime if row.get("option_type") == option_type]
        rows.append(_metric_summary_row(option_type, option_rows))
    if by_regime:
        rows.append(_metric_summary_row("all", by_regime))
    return rows


def _baseline_comparison_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_name, path_key in [
        ("direct_price_baseline", "baseline_metrics"),
        ("positive_premium", "positive_premium_metrics"),
    ]:
        for row in _read_csv(cfg[path_key]):
            if row.get("split") in {"validation", "test"} and row.get("option_type") == "all" and row.get("rmse"):
                rows.append(
                    {
                        "evidence_family": source_name,
                        "model": row.get("model"),
                        "split": row.get("split"),
                        "metric": "price_rmse",
                        "value": row.get("rmse"),
                        "max_obstacle_violation": row.get("max_obstacle_violation", ""),
                    }
                )
    delta_validation = _read_json(cfg["delta_validation_summary"])
    delta_head = _read_json(cfg["delta_head_summary"])
    rows.extend(
        [
            {
                "evidence_family": "delta",
                "model": "price_model_autograd_delta",
                "split": "test",
                "metric": "delta_rmse",
                "value": delta_validation.get("autograd_delta_rmse"),
                "max_obstacle_violation": "",
            },
            {
                "evidence_family": "delta",
                "model": "bounded_delta_head_mlp",
                "split": "test",
                "metric": "delta_rmse",
                "value": delta_head.get("selected_test_delta_rmse"),
                "max_obstacle_violation": "",
            },
        ]
    )
    return rows


def _boundary_failure_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    for source, path_key in [("pooled_boundary", "boundary_by_regime"), ("call_refinement", "call_boundary_by_regime")]:
        for row in _read_csv(cfg[path_key]):
            if row.get("split") != "test":
                continue
            mae = _float(row.get("boundary_mae"))
            combined.append(
                {
                    "source": source,
                    "model": row.get("model"),
                    "regime_key": row.get("regime_key"),
                    "option_type": row.get("option_type"),
                    "samples": row.get("samples"),
                    "boundary_rmse": row.get("boundary_rmse"),
                    "boundary_mae": row.get("boundary_mae"),
                    "boundary_max_error": row.get("boundary_max_error"),
                    "publication_disposition": "worst_case_boundary_limit",
                    "_sort_mae": mae,
                }
            )
    combined.sort(key=lambda row: row["_sort_mae"], reverse=True)
    return [{k: v for k, v in row.items() if not k.startswith("_")} for row in combined[:10]]


def _numerical_benchmark_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    convergence = _read_json(cfg["convergence_summary"])
    application = _read_json(cfg["application_summary"])
    return [
        {
            "evidence": "grid_domain_runtime",
            "metric": "reference_grid",
            "value": convergence.get("reference_grid"),
            "publication_interpretation": "Internal reference grid; not analytic publication-grade convergence.",
        },
        {
            "evidence": "grid_domain_runtime",
            "metric": "max_obstacle_violation",
            "value": convergence.get("max_obstacle_violation"),
            "publication_interpretation": "Obstacle enforcement remains robust in the tested grid/domain audit.",
        },
        {
            "evidence": "application_sweep",
            "metric": "diagnostic_failures",
            "value": application.get("diagnostic_failures"),
            "publication_interpretation": "Weak regimes must remain visible in manuscript and supplement.",
        },
    ]


def _metric_summary_row(scope: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "scope": scope,
        "target_test_regimes": "",
        "available_test_regimes": "",
        "audited_cases": len(rows),
        "skipped_cases": "",
        "disposition": "descriptive_integrated_metrics",
        "reason": "Grouped integrated audit summary.",
        "price_rmse_mean": _stat(rows, "price_rmse", mean),
        "price_rmse_median": _stat(rows, "price_rmse", median),
        "price_rmse_p90": _p90(rows, "price_rmse"),
        "price_rmse_max": _stat(rows, "price_rmse", max),
        "boundary_mae_mean": _stat(rows, "boundary_mae", mean),
        "strict_delta_rmse_mean": _stat(rows, "strict_delta_rmse", mean),
    }


def _stat(rows: list[dict[str, str]], key: str, fn) -> float | str:
    values = [_float(row.get(key)) for row in rows if row.get(key) not in {"", None}]
    values = [value for value in values if value == value]
    return fn(values) if values else ""


def _p90(rows: list[dict[str, str]], key: str) -> float | str:
    values = sorted(_float(row.get(key)) for row in rows if row.get(key) not in {"", None})
    values = [value for value in values if value == value]
    if not values:
        return ""
    index = min(len(values) - 1, int(0.9 * (len(values) - 1)))
    return values[index]


def _restricted_context(lower_line: str) -> bool:
    markers = ["do not", "does not", "not ", "no ", "cannot", "must not", "is not", "avoid", "without"]
    return any(marker in lower_line for marker in markers)


def _read_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, sort_keys=True))
