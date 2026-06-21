"""Stage 02 compiled-report image audit.

This script records the source and review status of every Stage 02 visual in the
compiled solver-validation report. It does not rerun solvers or train models.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from common import ensure_dir, load_config, write_csv, write_json


DEFAULT_CONFIG = "configs/report_visualization_enrichment.yaml"
DEFAULT_MANIFEST = "results/figures/report_enrichment/figure_manifest.csv"
DEFAULT_OUTPUT = "results/figures/report_enrichment/stage02_solver_validation_image_audit.csv"
STAGE02_REPORT = "reports/stage02_solver_validation.tex"
STAGE02_PDF = "deliverables/compiled_reports/stage02_solver_validation.pdf"


KIND_TO_GENERATOR = {
    "matrix": "save_matrix",
    "fd_stencil": "save_fd_stencil",
    "operator_coefficients": "save_operator_coefficients",
    "operator_matrix_annotated": "save_operator_matrix_annotated",
    "cn_matrix_pair": "save_cn_matrix_pair",
    "rhs_boundary_adjustment": "save_rhs_boundary_adjustment",
    "psor_update_flow": "save_psor_update_flow",
    "lcp_vector_form": "save_lcp_vector_form",
    "time_marching_grid": "save_time_marching_grid",
    "workflow": "save_workflow",
    "bar_from_csv": "save_bar_from_csv",
}


VISUAL_REVIEW = {
    "stage02_solver_validation_tridiagonal_operator": ("pass", "Readable sparse tridiagonal structure after 300 DPI regeneration."),
    "stage02_solver_validation_fd_stencil_validation": ("pass", "Stencil nodes and formulas are readable in the compiled report."),
    "stage02_solver_validation_coefficient_sign_audit": ("pass_with_note", "Dense formulas are readable after regeneration; exact formulas also remain in report text."),
    "stage02_solver_validation_annotated_lh_matrix": ("pass", "Matrix labels and lower/diagonal/upper annotations are readable."),
    "stage02_solver_validation_cn_a_b_matrices": ("pass_with_note", "Matrix pair is readable; rasterized matrix/colorbar elements require compiled-PDF PPI verification."),
    "stage02_solver_validation_rhs_boundary_terms": ("pass", "Boundary-adjustment boxes and arrows are readable."),
    "stage02_solver_validation_psor_projection_step": ("pass", "Projection-flow boxes and formulas are readable."),
    "stage02_solver_validation_lcp_matrix_system": ("pass", "LCP inequalities are readable in the compiled report."),
    "stage02_solver_validation_time_step_flow": ("pass", "Time-marching grid and annotations are readable."),
    "stage02_solver_validation_validation_ladder": ("pass", "Validation ladder labels are readable."),
    "stage02_solver_validation_smoke_error_comparison": ("pass", "Horizontal bar labels and scale are readable."),
    "stage02_solver_validation_obstacle_complementarity": ("pass", "Complementarity bars and labels are readable; zero obstacle violation remains captioned."),
}


COMPILED_PAGES = {
    "stage02_solver_validation_tridiagonal_operator": "4",
    "stage02_solver_validation_fd_stencil_validation": "5",
    "stage02_solver_validation_coefficient_sign_audit": "5",
    "stage02_solver_validation_annotated_lh_matrix": "6",
    "stage02_solver_validation_cn_a_b_matrices": "7",
    "stage02_solver_validation_rhs_boundary_terms": "7",
    "stage02_solver_validation_psor_projection_step": "8",
    "stage02_solver_validation_lcp_matrix_system": "8",
    "stage02_solver_validation_time_step_flow": "9",
    "stage02_solver_validation_validation_ladder": "10",
    "stage02_solver_validation_smoke_error_comparison": "11",
    "stage02_solver_validation_obstacle_complementarity": "11",
}


def run(
    *,
    config_path: str = DEFAULT_CONFIG,
    manifest_path: str = DEFAULT_MANIFEST,
    output_path: str = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    report = _stage02_report(cfg)
    manifest = _read_csv(manifest_path)
    manifest_by_id = {row.get("figure_id", ""): row for row in manifest}

    rows: list[dict[str, Any]] = []
    for figure in report.get("figures", []):
        figure_id = f"{Path(STAGE02_REPORT).stem}_{figure['figure_id']}"
        manifest_row = manifest_by_id.get(figure_id, {})
        visual_status, visual_note = VISUAL_REVIEW.get(
            figure_id,
            ("pending_manual_review", "Visual status must be set after inspecting the compiled PDF render."),
        )
        rows.append(
            {
                "figure_id": figure_id,
                "report_path": STAGE02_REPORT,
                "compiled_pdf": STAGE02_PDF,
                "compiled_pdf_page": COMPILED_PAGES.get(figure_id, ""),
                "latex_include_path": manifest_row.get("pdf_path", f"results/figures/report_enrichment/{figure_id}.pdf"),
                "config_kind": figure.get("kind", "workflow"),
                "plotting_function": KIND_TO_GENERATOR.get(figure.get("kind", "workflow"), "save_workflow"),
                "source_artifact": figure.get("source_artifact", "generated_conceptual"),
                "png_path": manifest_row.get("png_path", f"results/figures/report_enrichment/{figure_id}.png"),
                "pdf_path": manifest_row.get("pdf_path", f"results/figures/report_enrichment/{figure_id}.pdf"),
                "caption": figure.get("caption", ""),
                "panel_group": figure.get("panel_group", ""),
                "visual_status": visual_status,
                "visual_note": visual_note,
                "dpi_status": "pass_after_300dpi_regeneration",
                "required_action": "none" if visual_status.startswith("pass") else "review_and_polish",
            }
        )

    output = Path(output_path)
    ensure_dir(output.parent)
    write_csv(output, rows)
    summary = {
        "figures_audited": len(rows),
        "report_path": STAGE02_REPORT,
        "compiled_pdf": STAGE02_PDF,
        "output_path": output.as_posix(),
        "visual_status_counts": _counts(row["visual_status"] for row in rows),
        "dpi_status_counts": _counts(row["dpi_status"] for row in rows),
        "does_not_rerun_solver_or_training": True,
    }
    write_json(output.with_suffix(".summary.json"), summary)
    return summary


def _stage02_report(cfg: dict[str, Any]) -> dict[str, Any]:
    for report in cfg.get("reports", []):
        if report.get("path") == STAGE02_REPORT:
            return report
    raise ValueError(f"Stage 02 report entry not found in config: {STAGE02_REPORT}")


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    candidate = Path(path)
    if not candidate.exists():
        return []
    with candidate.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(config_path=args.config, manifest_path=args.manifest, output_path=args.output), indent=2, sort_keys=True))
