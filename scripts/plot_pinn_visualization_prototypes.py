"""Generate prototype figures for the PINN experiment visualization brief."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, TwoSlopeNorm


REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPO_ROOT / "output/figures"
DATA_DIR = REPO_ROOT / "results/figures/pinn_experiment_report"

STAGE23_SUMMARY = REPO_ROOT / "results/experiment_23_pinn_strong_terminal/pinn_campaign_summary.json"
STAGE23_HISTORY = REPO_ROOT / "results/experiment_23_pinn_strong_terminal/pinn_campaign_training_history.csv"
SURFACE_PATH = REPO_ROOT / "results/experiment_04_application_risk_surfaces/surfaces/put_T1_sig0p2_r0p05_q0.npz"

METRICS = [
    ("Price RMSE", "mean_test_price_rmse"),
    ("Masked residual", "mean_test_residual_mean_abs"),
    ("Boundary RMSE", "mean_test_boundary_rmse"),
    ("Strict Delta RMSE", "mean_test_delta_rmse"),
]

METHOD_LABELS = {
    "supervised_baseline": "baseline",
    "terminal_pde_weight_2": "PDE 2",
    "terminal_pde_weight_5": "PDE 5",
    "terminal_pde_weight_10": "PDE 10",
    "terminal_pde_weight_25": "PDE 25",
    "terminal_pde_weight_10_residual_sampling": "PDE 10 + RS",
    "terminal_pde_weight_25_residual_sampling": "PDE 25 + RS",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": False,
            "axes.unicode_minus": False,
        }
    )


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_all(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.pdf")
    fig.savefig(FIGURE_DIR / f"{stem}.png")
    plt.close(fig)


def save_stage23_metric_tradeoff() -> None:
    summary = load_json(STAGE23_SUMMARY)
    aggregates = {row["method"]: row for row in summary["method_aggregates"]}
    baseline = aggregates["supervised_baseline"]
    methods = [
        "terminal_pde_weight_2",
        "terminal_pde_weight_5",
        "terminal_pde_weight_10",
        "terminal_pde_weight_25",
        "terminal_pde_weight_10_residual_sampling",
        "terminal_pde_weight_25_residual_sampling",
    ]

    values = np.zeros((len(methods), len(METRICS)))
    rows: list[dict[str, str | float | int]] = []
    for row_idx, method in enumerate(methods):
        aggregate = aggregates[method]
        for col_idx, (metric_label, metric_key) in enumerate(METRICS):
            baseline_value = float(baseline[metric_key])
            candidate_value = float(aggregate[metric_key])
            relative_change = 100.0 * (candidate_value / baseline_value - 1.0)
            values[row_idx, col_idx] = relative_change
            rows.append(
                {
                    "method": method,
                    "metric": metric_label,
                    "metric_key": metric_key,
                    "baseline_value": baseline_value,
                    "candidate_value": candidate_value,
                    "relative_change_percent": relative_change,
                    "candidate_success_count": int(aggregate["candidate_success_count"]),
                }
            )
    write_csv(DATA_DIR / "stage23_metric_tradeoff.csv", rows)

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    norm = TwoSlopeNorm(vmin=-90.0, vcenter=0.0, vmax=40.0)
    image = ax.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels([label for label, _ in METRICS])
    ylabels = [f"{METHOD_LABELS.get(method, method)} ({aggregates[method]['candidate_success_count']}/3)" for method in methods]
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(ylabels)
    ax.set_title("Stage 23 terminal-scaled methods versus matched supervised baseline")
    ax.set_xlabel("Lower metric values are better; negative percentages are improvements")
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                f"{values[row_idx, col_idx]:+.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="#111111",
            )
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Relative change (%)")
    save_all(fig, "pinn_stage23_metric_tradeoff")


def read_history_rows() -> list[dict[str, str]]:
    with STAGE23_HISTORY.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate_history(rows: list[dict[str, str]], method: str, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] != method:
            continue
        grouped[int(row["epoch"])].append(float(row[key]))
    epochs = np.array(sorted(grouped), dtype=float)
    means = np.array([np.mean(grouped[int(epoch)]) for epoch in epochs])
    mins = np.array([np.min(grouped[int(epoch)]) for epoch in epochs])
    maxs = np.array([np.max(grouped[int(epoch)]) for epoch in epochs])
    return epochs, means, mins, maxs


def save_stage23_training_curves() -> None:
    rows = read_history_rows()
    csv_rows: list[dict[str, str | float | int]] = []
    for method in ["supervised_baseline", "terminal_pde_weight_5"]:
        for key in ["validation_premium_rmse", "train_supervised_loss", "train_pde_loss"]:
            epochs, means, mins, maxs = aggregate_history(rows, method, key)
            for epoch, mean, low, high in zip(epochs, means, mins, maxs):
                csv_rows.append(
                    {
                        "method": method,
                        "series": key,
                        "epoch": int(epoch),
                        "mean": float(mean),
                        "min": float(low),
                        "max": float(high),
                    }
                )
    write_csv(DATA_DIR / "stage23_training_curves.csv", csv_rows)

    colors = {
        "supervised_baseline": "#7c8a99",
        "terminal_pde_weight_5": "#167f7a",
        "terminal_pde_weight_5_pde": "#b65f2a",
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))

    for method in ["supervised_baseline", "terminal_pde_weight_5"]:
        epochs, means, mins, maxs = aggregate_history(rows, method, "validation_premium_rmse")
        label = "supervised baseline" if method == "supervised_baseline" else "terminal PDE 5"
        axes[0].plot(epochs, means, label=label, color=colors[method], lw=2)
        axes[0].fill_between(epochs, mins, maxs, color=colors[method], alpha=0.16, linewidth=0)
    axes[0].set_title("Validation premium RMSE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("RMSE")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    for method, key, label, color_key in [
        ("supervised_baseline", "train_supervised_loss", "baseline supervised loss", "supervised_baseline"),
        ("terminal_pde_weight_5", "train_supervised_loss", "terminal PDE 5 supervised loss", "terminal_pde_weight_5"),
        ("terminal_pde_weight_5", "train_pde_loss", "terminal PDE 5 masked PDE loss", "terminal_pde_weight_5_pde"),
    ]:
        epochs, means, _, _ = aggregate_history(rows, method, key)
        means = np.maximum(means, 1.0e-12)
        axes[1].plot(epochs, means, label=label, color=colors[color_key], lw=2)
    axes[1].set_title("Training loss components")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7)

    fig.suptitle("Stage 23 capped put training traces across seeds 21-23", y=1.02, fontsize=11)
    save_all(fig, "pinn_stage23_training_curves")


def save_premium_mask_panel() -> None:
    surface = np.load(SURFACE_PATH)
    spots = surface["spots"]
    taus = surface["taus"]
    values = surface["values"]
    payoff = surface["payoff"]
    boundary = surface["boundary"]
    premium = np.maximum(values - payoff[None, :], 0.0)
    pde_mask = (
        (premium > 1.0e-5)
        & (~surface["mask_near_boundary"])
        & (~surface["mask_near_kink"])
        & (taus[:, None] > 0.08)
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.35), sharey=True)
    extent = [float(spots.min()), float(spots.max()), float(taus.min()), float(taus.max())]
    premium_image = axes[0].imshow(premium, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    axes[0].plot(boundary[1:], taus[1:], color="white", lw=1.8, label="PSOR boundary")
    axes[0].axvline(1.0, color="#222222", ls="--", lw=0.9, alpha=0.85)
    axes[0].set_title("PSOR continuation premium")
    axes[0].set_xlabel(r"$S/K$")
    axes[0].set_ylabel(r"$\tau/T$")
    axes[0].legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.82,
        fontsize=8,
        loc="upper right",
    )
    cbar0 = fig.colorbar(premium_image, ax=axes[0], fraction=0.046, pad=0.03)
    cbar0.set_label(r"$C/K$")

    mask_cmap = ListedColormap(["#f2f4f7", "#167f7a"])
    mask_image = axes[1].imshow(pde_mask.astype(float), origin="lower", extent=extent, aspect="auto", cmap=mask_cmap, vmin=0, vmax=1)
    axes[1].plot(boundary[1:], taus[1:], color="#222222", lw=1.4)
    axes[1].axvline(1.0, color="#222222", ls="--", lw=0.9, alpha=0.85)
    axes[1].set_title("Masked PDE residual support")
    axes[1].set_xlabel(r"$S/K$")
    cbar1 = fig.colorbar(mask_image, ax=axes[1], fraction=0.046, pad=0.03, ticks=[0, 1])
    cbar1.ax.set_yticklabels(["excluded", "used"])

    fig.suptitle("Representative American put surface: T=1, sigma=0.2, r=0.05, q=0", y=1.02, fontsize=11)
    save_all(fig, "pinn_psor_premium_mask_panel")


def main() -> None:
    apply_style()
    save_stage23_metric_tradeoff()
    save_stage23_training_curves()
    save_premium_mask_panel()
    for name in [
        "pinn_stage23_metric_tradeoff",
        "pinn_stage23_training_curves",
        "pinn_psor_premium_mask_panel",
    ]:
        print(f"Wrote {Path('output/figures') / (name + '.pdf')}")
        print(f"Wrote {Path('output/figures') / (name + '.png')}")
    print("Wrote results/figures/pinn_experiment_report/stage23_metric_tradeoff.csv")
    print("Wrote results/figures/pinn_experiment_report/stage23_training_curves.csv")


if __name__ == "__main__":
    main()
