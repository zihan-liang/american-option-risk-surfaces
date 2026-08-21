"""Reproducible post-processing for consolidated experiment data.

This module never trains models or mutates accepted source artifacts.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

SHARED_DOMAIN = (0.5, 1.5, 0.01, 1.0)
HD_DOMAIN = (0.2, 1.5, 0.01, 1.0)

DIRK_METRIC_COLUMNS = [
    "method_id",
    "dimension",
    "seed",
    "regime_id",
    "internal_cnpsor_price_rmse",
    "dirk_price_rmse_hd_domain",
    "dirk_price_rmse_shared_domain",
    "reference_type",
    "metric_scope",
    "s_min",
    "s_max",
    "tau_min",
    "tau_max",
    "grid_points",
    "interpolation_method",
    "source_repo",
    "source_branch",
    "source_commit",
    "reference_source",
    "reference_commit",
    "reference_artifact_sha256",
]

TABLE_HEADERS = {
    "source_artifact_inventory.csv": ["repo", "representative_path", "extension", "size_bytes", "git_blob_sha", "content_sha256", "artifact_type", "experiment_family", "method_id", "regime_id", "seed", "dimension", "duplicate_group", "canonical_source"],
    "provenance.csv": ["repo", "branch", "branch_commit", "path", "git_blob_sha", "content_sha256", "duplicate_group"],
    "experiment_runs.csv": ["repo", "branch", "source_commit", "experiment_family", "experiment_id", "method_id", "variant", "dimension", "seed", "status", "diagnostic_only", "candidate_supported", "reference_type", "scoring_protocol"],
    "price_metrics.csv": ["repo", "branch", "source_commit", "experiment_id", "method_id", "dimension", "seed", "metric_name", "metric_value", "reference_type", "metric_scope", "regime_id"],
    "greek_metrics.csv": ["repo", "branch", "source_commit", "experiment_id", "method_id", "dimension", "seed", "metric_name", "metric_value", "greek_definition", "greek_source", "reference_type"],
    "boundary_metrics.csv": ["repo", "branch", "source_commit", "experiment_id", "method_id", "dimension", "seed", "threshold", "reference_count", "predicted_count", "matched_count", "coverage", "found_rate", "conditional_rmse", "full_rmse", "exercise_f1", "boundary_tube_f1", "missing_reason", "boundary_extraction_method"],
    "free_boundary_points.csv": ["repo", "branch", "source_commit", "method_id", "dimension", "seed", "regime_id", "tau_over_T", "boundary_over_K", "boundary_found", "threshold", "missing_reason"],
    "price_slices.csv": ["repo", "branch", "source_commit", "method_id", "dimension", "seed", "regime_id", "S_over_K", "tau_over_T", "V_over_K", "payoff_over_K", "premium_over_K", "reference_type"],
    "structural_metrics.csv": ["repo", "branch", "source_commit", "method_id", "dimension", "seed", "metric_name", "metric_value"],
    "compute_metrics.csv": ["repo", "branch", "source_commit", "method_id", "dimension", "seed", "training_seconds", "evaluation_seconds", "solver_seconds", "oracle_seconds", "screening_seconds", "confirmation_seconds", "wall_clock_seconds", "hardware", "protocol"],
    "classical_solver_metrics.csv": ["repo", "branch", "source_commit", "method_id", "regime_id", "metric_name", "metric_value", "reference_type"],
    "binomial_convergence.csv": ["repo", "branch", "source_commit", "steps", "S_over_K", "tau_over_T", "V_over_K", "comparison_metric", "comparison_value"],
    "ruixi_arm_metrics.csv": ["repo", "branch", "source_commit", "arm", "method_id", "regime_id", "seed", "metric_name", "metric_value", "reference_type"],
    "hd1_metrics.csv": ["repo", "branch", "source_commit", "model_id", "dimension", "seed", "metric_name", "metric_value", "gate_name", "gate_pass"],
    "hd2_candidate_metrics.csv": ["repo", "branch", "source_commit", "candidate_id", "dimension", "seed", "metric_name", "metric_value", "gate_name", "gate_pass", "selection_status"],
    "hd3_metrics.csv": ["repo", "branch", "source_commit", "model_id", "dimension", "seed", "metric_name", "metric_value", "reference_type", "regime_id"],
    "campaign_method_results.csv": ["repo", "branch", "source_commit", "method_id", "method_name", "run_status", "metric_name", "metric_value", "gate_name", "gate_threshold", "gate_pass", "final_decision"],
    "experiment_gates.csv": ["repo", "branch", "source_commit", "method_id", "experiment_id", "gate", "metric", "operator", "threshold", "actual", "passed", "decision"],
    "environment_runs.csv": ["repo", "branch", "source_commit", "experiment_id", "method_id", "seed", "os", "python", "pytorch", "cuda", "gpu", "cpu", "ram", "device"],
    "surface_artifacts.csv": ["repo", "branch", "source_commit", "path", "method_id", "dimension", "seed", "regime_id", "shape", "s_min", "s_max", "tau_min", "tau_max", "fields", "point_count", "content_sha256"],
    "dirk_common_reference_metrics.csv": DIRK_METRIC_COLUMNS,
}


def price_rmse(predicted: Sequence[float], reference: Sequence[float]) -> float:
    """Return the unweighted pointwise RMSE for equal-length finite sequences."""
    if len(predicted) != len(reference):
        raise ValueError("predicted and reference lengths differ")
    if not predicted:
        raise ValueError("RMSE requires at least one point")
    total = 0.0
    for pred, ref in zip(predicted, reference):
        p = float(pred)
        r = float(ref)
        if not (math.isfinite(p) and math.isfinite(r)):
            raise ValueError("RMSE inputs must be finite")
        total += (p - r) ** 2
    return math.sqrt(total / len(predicted))


def _write_header(path: Path, header: Iterable[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(list(header))


def write_empty_output_contract(output_dir: Path | str) -> None:
    """Create schema-valid empty consolidated outputs for contract validation."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for filename, header in TABLE_HEADERS.items():
        _write_header(output / filename, header)

    manifest = {
        "schema_version": 1,
        "headline_cross_model_metric": "dirk_price_rmse_shared_domain",
        "headline_hd_scaling_metric": "dirk_price_rmse_hd_domain",
        "shared_domain": list(SHARED_DOMAIN),
        "hd_domain": list(HD_DOMAIN),
        "generated_tables": sorted(TABLE_HEADERS),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Consolidated experiment CSV data\n\n"
        "Generated by post-processing accepted immutable experiment artifacts. "
        "Empty specialized tables mean no supporting source artifact has yet been parsed; "
        "they must never be interpreted as zero-valued results.\n",
        encoding="utf-8",
    )
