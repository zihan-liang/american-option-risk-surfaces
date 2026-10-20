from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from common_plot_utils import PLOT_OUT, SUMMARY_CSV, save_figure


def plot_boundary_coverage(df: pd.DataFrame) -> None:
    ordered = df.sort_values(["model", "dimension"]).reset_index(drop=True)
    labels = [f"{row.model}\n{int(row.dimension)}D" for row in ordered.itertuples(index=False)]
    values = 100.0 * ordered["boundary_coverage"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Primary free-boundary coverage (%)")

    upper = max(5.0, float(values.max()) * 1.2 if len(values) else 5.0)
    ax.set_ylim(0.0, upper)

    if len(values) and (values == 0.0).all():
        ax.text(
            0.5,
            0.58,
            "No HD model detected a primary free boundary\nat the frozen tolerance $10^{-6}$",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )

    save_figure(fig, PLOT_OUT / "F_hd_boundary_coverage")


def plot_accuracy_compute(df: pd.DataFrame) -> None:
    ordered = df.sort_values(["training_seconds", "price_rmse_internal"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for row in ordered.itertuples(index=False):
        ax.scatter(row.training_seconds, row.price_rmse_internal, s=55)
        ax.annotate(
            f"{row.model} {int(row.dimension)}D",
            (row.training_seconds, row.price_rmse_internal),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Training time (s)")
    ax.set_ylabel("Price RMSE (internal CN/PSOR reference)")
    save_figure(fig, PLOT_OUT / "F_hd_accuracy_compute_internal")


def main() -> None:
    df = pd.read_csv(SUMMARY_CSV)
    plot_boundary_coverage(df)
    plot_accuracy_compute(df)
    print(f"Boundary/compute figures written to: {PLOT_OUT}")


if __name__ == "__main__":
    main()
