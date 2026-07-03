"""Generate figures and plotted data for the PINN experiment report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_SUMMARIES = {
    "Stage 22": REPO_ROOT / "results/experiment_22_pinn_campaign/pinn_campaign_summary.json",
    "Stage 23": REPO_ROOT / "results/experiment_23_pinn_strong_terminal/pinn_campaign_summary.json",
}
STAGE21_METRICS = REPO_ROOT / "results/experiment_21_pinn_inspired_premium/pinn_metrics.csv"
STAGE21_RESIDUALS = REPO_ROOT / "results/experiment_21_pinn_inspired_premium/pinn_residual_metrics.csv"
STAGE21_METHODS = [
    "supervised_positive_premium_mlp",
    "pinn_inspired_premium_mlp",
]
STAGE22_METHODS = [
    "supervised_baseline",
    "pde_weight_1e-6",
    "pde_weight_1e-5",
    "pde_weight_1e-4",
    "pde_weight_1e-3",
    "pde_ramp",
    "residual_sampling",
    "delta_loss",
    "terminal_premium",
    "terminal_plus_ramp",
    "terminal_plus_ramp_plus_residual_sampling",
]
STAGE23_METHODS = [
    "supervised_baseline",
    "terminal_pde_weight_2",
    "terminal_pde_weight_5",
    "terminal_pde_weight_10",
    "terminal_pde_weight_25",
    "terminal_pde_weight_10_residual_sampling",
    "terminal_pde_weight_25_residual_sampling",
]
METRICS = [
    ("Price RMSE", "mean_test_price_rmse"),
    ("Masked residual", "mean_test_residual_mean_abs"),
    ("Boundary RMSE", "mean_test_boundary_rmse"),
    ("Strict Delta RMSE", "mean_test_delta_rmse"),
]
CSV_OUTPUT = REPO_ROOT / "results/figures/pinn_experiment_report/stage22_stage23_relative_changes.csv"
COVERAGE_CSV_OUTPUT = REPO_ROOT / "results/figures/pinn_experiment_report/full_method_coverage.csv"
FIGURE_STEM = REPO_ROOT / "output/figures/pinn_stage22_stage23_relative_changes"


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def aggregate_lookup(summary: dict) -> dict[str, dict]:
    return {row["method"]: row for row in summary["method_aggregates"]}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plotted_rows() -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for stage, path in STAGE_SUMMARIES.items():
        summary = load_summary(path)
        aggregates = aggregate_lookup(summary)
        baseline = aggregates["supervised_baseline"]
        candidate = aggregates[summary["best_method"]]
        for metric_label, metric_key in METRICS:
            baseline_value = float(baseline[metric_key])
            candidate_value = float(candidate[metric_key])
            relative_change = 100.0 * (candidate_value / baseline_value - 1.0)
            rows.append(
                {
                    "stage": stage,
                    "method": candidate["method"],
                    "metric": metric_label,
                    "metric_key": metric_key,
                    "baseline_value": baseline_value,
                    "candidate_value": candidate_value,
                    "relative_change_percent": relative_change,
                }
            )
    return rows


def write_csv(rows: list[dict[str, str | float]]) -> None:
    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage",
        "method",
        "metric",
        "metric_key",
        "baseline_value",
        "candidate_value",
        "relative_change_percent",
    ]
    with CSV_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def coverage_rows() -> list[dict[str, str | float | int]]:
    rows: list[dict[str, str | float | int]] = []
    stage21_metrics = {
        row["model"]: row
        for row in read_csv_rows(STAGE21_METRICS)
        if row["split"] == "test" and row["option_type"] == "all"
    }
    stage21_residuals = {
        row["model"]: row
        for row in read_csv_rows(STAGE21_RESIDUALS)
        if row["split"] == "test"
    }
    for method in STAGE21_METHODS:
        metric_row = stage21_metrics[method]
        residual_row = stage21_residuals[method]
        rows.append(
            {
                "stage": "Stage 21",
                "method": method,
                "candidate_success_count": "not_gated",
                "runs_or_seeds": 1,
                "mean_test_price_rmse": float(metric_row["price_rmse"]),
                "mean_test_residual_mean_abs": float(residual_row["residual_mean_abs"]),
            }
        )

    stage_specs = [
        ("Stage 22", STAGE_SUMMARIES["Stage 22"], STAGE22_METHODS),
        ("Stage 23", STAGE_SUMMARIES["Stage 23"], STAGE23_METHODS),
    ]
    for stage, path, methods in stage_specs:
        summary = load_summary(path)
        aggregates = aggregate_lookup(summary)
        seed_count = len(summary["seeds"])
        for method in methods:
            row = aggregates[method]
            rows.append(
                {
                    "stage": stage,
                    "method": method,
                    "candidate_success_count": int(row["candidate_success_count"]),
                    "runs_or_seeds": seed_count,
                    "mean_test_price_rmse": float(row["mean_test_price_rmse"]),
                    "mean_test_residual_mean_abs": float(row["mean_test_residual_mean_abs"]),
                }
            )
    return rows


def write_coverage_csv(rows: list[dict[str, str | float | int]]) -> None:
    COVERAGE_CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage",
        "method",
        "candidate_success_count",
        "runs_or_seeds",
        "mean_test_price_rmse",
        "mean_test_residual_mean_abs",
    ]
    with COVERAGE_CSV_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_figure(rows: list[dict[str, str | float]]) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.unicode_minus": False,
        }
    )
    stages = list(STAGE_SUMMARIES.keys())
    metrics = [label for label, _ in METRICS]
    values = {
        stage: [
            float(next(row for row in rows if row["stage"] == stage and row["metric"] == metric)["relative_change_percent"])
            for metric in metrics
        ]
        for stage in stages
    }

    x = np.arange(len(metrics))
    width = 0.34
    colors = {"Stage 22": "#7c8a99", "Stage 23": "#167f7a"}

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for offset, stage in zip([-width / 2, width / 2], stages):
        bars = ax.bar(x + offset, values[stage], width, label=stage, color=colors[stage])
        for bar, value in zip(bars, values[stage]):
            va = "bottom" if value >= 0 else "top"
            y = value + (3.0 if value >= 0 else -3.0)
            ax.text(bar.get_x() + bar.get_width() / 2, y, f"{value:+.1f}%", ha="center", va=va, fontsize=8)

    ax.axhline(0.0, color="#333333", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Relative change vs matched supervised baseline (%)")
    ax.set_title("Best PINN-inspired method by stage; lower metric values are better")
    ax.set_ylim(-105, 135)
    ax.legend(frameon=False, loc="upper left")

    FIGURE_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_STEM.with_suffix(".pdf"))
    fig.savefig(FIGURE_STEM.with_suffix(".png"))
    plt.close(fig)


def main() -> None:
    rows = plotted_rows()
    coverage = coverage_rows()
    write_csv(rows)
    write_coverage_csv(coverage)
    save_figure(rows)
    print(f"Wrote {CSV_OUTPUT.relative_to(REPO_ROOT)}")
    print(f"Wrote {COVERAGE_CSV_OUTPUT.relative_to(REPO_ROOT)}")
    print(f"Wrote {FIGURE_STEM.with_suffix('.pdf').relative_to(REPO_ROOT)}")
    print(f"Wrote {FIGURE_STEM.with_suffix('.png').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
