"""Stage 15: generate publication-facing figures and a traceability manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from common import ensure_dir, load_config, write_csv, write_json
from plotting.publication import save_bar_figure, save_image_panel, save_schematic, save_two_region_boundary_figure


DEFAULT_CONFIG = "configs/publication_figures.yaml"


def run(config_path: str = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg = load_config(config_path)
    fig_dir = ensure_dir(cfg.get("figure_dir", "results/figures/publication"))
    audit_dir = Path(cfg.get("publication_audit_dir", "results/experiment_17_publication_audit"))
    sources = cfg.get("source_figures", {})

    manifest: list[dict[str, Any]] = []

    _record(manifest, fig_dir, "fig01_workflow_schematic", "workflow_schematic", "Generated from Stage 15 design", lambda p: save_schematic(p))
    _record(manifest, fig_dir, "fig02_free_boundary_regions", "free_boundary_regions", "Generated conceptual formulation", lambda p: save_two_region_boundary_figure(p))
    _record(manifest, fig_dir, "fig03_solver_validation", "solver_validation", sources.get("solver_validation", ""), lambda p: save_image_panel(p, sources.get("solver_validation"), "Solver convergence and domain evidence"))
    _record(manifest, fig_dir, "fig04_risk_surface", "risk_surface", sources.get("risk_surface", ""), lambda p: save_image_panel(p, sources.get("risk_surface"), "Representative audited risk surface"))
    _record(manifest, fig_dir, "fig05_application_heatmap", "application_heatmap", sources.get("application_heatmap", ""), lambda p: save_image_panel(p, sources.get("application_heatmap"), "Application regime heatmap"))
    _record(manifest, fig_dir, "fig06_surrogate_comparison", "surrogate_comparison", str(audit_dir / "baseline_comparison.csv"), lambda p: _surrogate_comparison(p, audit_dir / "baseline_comparison.csv"))
    _record(manifest, fig_dir, "fig07_delta_comparison", "delta_comparison", str(audit_dir / "baseline_comparison.csv"), lambda p: _delta_comparison(p, audit_dir / "baseline_comparison.csv"))
    _record(manifest, fig_dir, "fig08_integrated_workflow_errors", "integrated_workflow_errors", str(audit_dir / "integrated_scope_summary.csv"), lambda p: _integrated_scope(p, audit_dir / "integrated_scope_summary.csv"))
    _record(manifest, fig_dir, "fig09_boundary_failure_modes", "boundary_failure_modes", str(audit_dir / "boundary_failure_analysis.csv"), lambda p: _boundary_failures(p, audit_dir / "boundary_failure_analysis.csv"))

    write_csv(fig_dir / "figure_manifest.csv", manifest)
    summary = {"figures": len(manifest), "figure_dir": str(fig_dir), "manifest": str(fig_dir / "figure_manifest.csv")}
    write_json(fig_dir / "publication_figure_summary.json", summary)
    return summary


def _record(manifest: list[dict[str, Any]], fig_dir: Path, figure_id: str, title: str, source: str, fn) -> None:
    png = fig_dir / f"{figure_id}.png"
    pdf = fig_dir / f"{figure_id}.pdf"
    fn(png)
    fn(pdf)
    manifest.append(
        {
            "figure_id": figure_id,
            "title": title,
            "png_path": str(png),
            "pdf_path": str(pdf),
            "source_artifact": source or "generated_from_repository_evidence",
        }
    )


def _surrogate_comparison(path: Path, csv_path: Path) -> None:
    rows = _read_csv(csv_path)
    labels, values = [], []
    for row in rows:
        if row.get("split") == "validation" and row.get("metric") == "price_rmse":
            labels.append(row["model"])
            values.append(float(row["value"]))
    save_bar_figure(
        path,
        labels[:6],
        values[:6],
        "Validation price RMSE",
        "Price RMSE",
        orientation="horizontal",
    )


def _delta_comparison(path: Path, csv_path: Path) -> None:
    rows = [row for row in _read_csv(csv_path) if row.get("metric") == "delta_rmse"]
    save_bar_figure(
        path,
        [row["model"] for row in rows],
        [float(row["value"]) for row in rows],
        "Delta evidence comparison",
        r"Strict-mask $\Delta$ RMSE",
        orientation="horizontal",
    )


def _integrated_scope(path: Path, csv_path: Path) -> None:
    rows = [row for row in _read_csv(csv_path) if row.get("scope") in {"put", "call", "all"}]
    save_bar_figure(
        path,
        [row["scope"] for row in rows],
        [float(row["price_rmse_mean"]) for row in rows],
        "Integrated workflow price error by scope",
        "Mean price RMSE",
    )


def _boundary_failures(path: Path, csv_path: Path) -> None:
    rows = _read_csv(csv_path)[:8]
    model_labels = {
        "boundary_head_linear": "Boundary linear",
        "pooled_boundary_head_linear_on_calls": "Pooled linear (call audit)",
        "call_boundary_head_mlp": "Call-boundary MLP",
        "call_boundary_head_linear": "Call-boundary linear",
    }
    raw_labels = [f"{row['model']}|{row['regime_key']}" for row in rows]
    display_labels = {
        raw: (
            f"{model_labels.get(row['model'], row['model'])}: "
            f"$T={row['regime_key'].split('_')[1][1:].replace('p', '.')}$"
        )
        for raw, row in zip(raw_labels, rows)
    }
    save_bar_figure(
        path,
        raw_labels,
        [float(row["boundary_mae"]) for row in rows],
        r"Worst boundary-head regimes ($\sigma=0.4$, $r=0.1$, $q=0.03$)",
        "Boundary MAE",
        orientation="horizontal",
        display_labels=display_labels,
    )


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, sort_keys=True))
