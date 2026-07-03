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
METRICS = [
    ("Price RMSE", "mean_test_price_rmse"),
    ("Masked residual", "mean_test_residual_mean_abs"),
    ("Boundary RMSE", "mean_test_boundary_rmse"),
    ("Strict Delta RMSE", "mean_test_delta_rmse"),
]
CSV_OUTPUT = REPO_ROOT / "results/figures/pinn_experiment_report/stage22_stage23_relative_changes.csv"
FIGURE_STEM = REPO_ROOT / "output/figures/pinn_stage22_stage23_relative_changes"


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def aggregate_lookup(summary: dict) -> dict[str, dict]:
    return {row["method"]: row for row in summary["method_aggregates"]}


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
    write_csv(rows)
    save_figure(rows)
    print(f"Wrote {CSV_OUTPUT.relative_to(REPO_ROOT)}")
    print(f"Wrote {FIGURE_STEM.with_suffix('.pdf').relative_to(REPO_ROOT)}")
    print(f"Wrote {FIGURE_STEM.with_suffix('.png').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
