"""Generate prototype figures for the PINN experiment visualization brief."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch


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


def add_box(
    ax: plt.Axes,
    *,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = "#2d3748",
    fontsize: float = 8.7,
    weight: str = "normal",
    linestyle: str = "-",
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.012",
        linewidth=1.05,
        linestyle=linestyle,
        facecolor=facecolor,
        edgecolor=edgecolor,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        color="#172033",
        transform=ax.transAxes,
        linespacing=1.22,
    )


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    label: str | None = None,
    label_offset: tuple[float, float] = (0.0, 0.0),
    color: str = "#405469",
    connectionstyle: str = "arc3",
    zorder: float = 3.0,
) -> None:
    annotation = ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={
            "arrowstyle": "->",
            "lw": 1.35,
            "color": color,
            "shrinkA": 2,
            "shrinkB": 2,
            "connectionstyle": connectionstyle,
        },
    )
    annotation.set_zorder(zorder)
    if label:
        x = (start[0] + end[0]) / 2 + label_offset[0]
        y = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=7.2,
            color=color,
            transform=ax.transAxes,
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.78},
        )


def add_group(
    ax: plt.Axes,
    *,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    edgecolor: str = "#8a8f98",
) -> None:
    group = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.008",
        linewidth=1.25,
        linestyle=(0, (5, 4)),
        facecolor="none",
        edgecolor=edgecolor,
        transform=ax.transAxes,
    )
    ax.add_patch(group)
    ax.text(
        xy[0],
        xy[1] + height + 0.018,
        title,
        ha="left",
        va="bottom",
        fontsize=9.2,
        color="#172033",
        transform=ax.transAxes,
    )


def draw_compact_network(
    ax: plt.Axes,
    *,
    input_nodes: list[tuple[float, float, str]],
    hidden_layers: list[list[tuple[float, float]]],
    output_node: tuple[float, float, str],
) -> None:
    """Draw a small fully connected MLP glyph in axes coordinates."""

    def connect(source: list[tuple[float, float]], target: list[tuple[float, float]]) -> None:
        for x0, y0 in source:
            for x1, y1 in target:
                ax.plot(
                    [x0, x1],
                    [y0, y1],
                    color="#8b97a8",
                    linewidth=0.42,
                    alpha=0.62,
                    transform=ax.transAxes,
                    zorder=1,
                )

    input_points = [(x, y) for x, y, _ in input_nodes]
    output_points = [(output_node[0], output_node[1])]
    connect(input_points, hidden_layers[0])
    for left, right in zip(hidden_layers[:-1], hidden_layers[1:]):
        connect(left, right)
    connect(hidden_layers[-1], output_points)

    def add_node(
        x: float,
        y: float,
        *,
        label: str | None = None,
        size: float = 92,
        facecolor: str = "#e8f5f3",
        edgecolor: str = "#2c7a7b",
        fontsize: float = 5.7,
    ) -> None:
        ax.scatter(
            [x],
            [y],
            s=size,
            marker="o",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.85,
            transform=ax.transAxes,
            zorder=3,
        )
        if label:
            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=fontsize,
                weight="bold",
                color="#172033",
                transform=ax.transAxes,
                zorder=4,
            )

    for x, y, label in input_nodes:
        add_node(x, y, label=label, size=112, facecolor="#edf2f7", edgecolor="#344154")
    for layer in hidden_layers:
        for x, y in layer:
            add_node(x, y, size=78, facecolor="#cfe8f3", edgecolor="#2c5f86")
    add_node(
        output_node[0],
        output_node[1],
        label=output_node[2],
        size=160,
        facecolor="#f7c948",
        edgecolor="#8a6411",
        fontsize=5.2,
    )

    ax.text(
        0.168,
        0.832,
        "fully connected neural net",
        ha="center",
        va="bottom",
        fontsize=5.4,
        color="#52616f",
        transform=ax.transAxes,
    )


def save_pinn_project_mechanism_illustration() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.set_axis_off()
    ax.text(
        0.5,
        0.975,
        "Project-specific PINN-inspired mechanism",
        ha="center",
        va="center",
        fontsize=11.5,
        weight="bold",
        transform=ax.transAxes,
    )

    colors = {
        "input": "#edf2f7",
        "psor": "#eef3ff",
        "model": "#e8f5f3",
        "price": "#f1f8e9",
        "residual": "#fff4e6",
        "objective": "#f7eef8",
        "gate": "#f5f7fa",
        "claim": "#fbfbfd",
        "loss": "#d8b4d8",
        "autodiff": "#e9f7f5",
        "premium": "#f7c948",
        "network": "#3f8fc5",
    }

    add_group(ax, xy=(0.035, 0.30), width=0.33, height=0.56, title="Premium Surrogate Network")
    add_group(ax, xy=(0.385, 0.30), width=0.11, height=0.56, title="AutoDiff", edgecolor="#78aaa4")
    add_group(ax, xy=(0.525, 0.30), width=0.44, height=0.56, title="PINN-inspired Regularization")

    input_nodes = [
        (0.065, 0.805, "$x$"),
        (0.065, 0.775, "$\\theta$"),
        (0.065, 0.745, "$r$"),
        (0.065, 0.715, "$q$"),
        (0.065, 0.685, "$\\sigma$"),
        (0.065, 0.655, "$T$"),
    ]
    hidden_y = [0.795, 0.755, 0.715, 0.675]
    draw_compact_network(
        ax,
        input_nodes=input_nodes,
        hidden_layers=[
            [(0.145, y) for y in hidden_y],
            [(0.225, y) for y in hidden_y],
        ],
        output_node=(0.305, 0.735, "$N_\\vartheta$"),
    )
    add_box(
        ax,
        xy=(0.185, 0.525),
        width=0.155,
        height=0.088,
        text="softplus + $\\theta$ scaling\n$\\widehat C=\\theta\\,\\mathrm{softplus}(N_\\vartheta(z))$\n$\\widehat C\\geq0$",
        facecolor=colors["premium"],
        edgecolor="#8a6411",
        fontsize=4.65,
        weight="bold",
    )
    add_box(
        ax,
        xy=(0.065, 0.43),
        width=0.205,
        height=0.085,
        text="Reconstruct price\n$\\widehat u=\\Phi/K+\\widehat C$",
        facecolor=colors["price"],
        fontsize=6.8,
        weight="bold",
    )

    for y, label in [
        (0.72, "$\\widehat u_\\theta$"),
        (0.62, "$\\widehat u_x$"),
        (0.52, "$\\widehat u_{xx}$"),
        (0.42, "$\\Delta$"),
    ]:
        add_box(
            ax,
            xy=(0.405, y),
            width=0.055,
            height=0.055,
            text=label,
            facecolor=colors["autodiff"],
            edgecolor="#3d8a83",
            fontsize=7.0,
            weight="bold",
        )

    add_box(
        ax,
        xy=(0.555, 0.71),
        width=0.37,
        height=0.08,
        text="$L_{\\mathrm{sup}}$: match PSOR price / premium labels",
        facecolor=colors["loss"],
        edgecolor="#815681",
        fontsize=6.9,
        weight="bold",
    )
    add_box(
        ax,
        xy=(0.555, 0.58),
        width=0.37,
        height=0.09,
        text="$L_{\\mathrm{PDE}}$: masked Black-Scholes residual\nPSOR continuation mask;\naway from boundary/kink/maturity",
        facecolor=colors["loss"],
        edgecolor="#815681",
        fontsize=6.1,
        weight="bold",
    )
    add_box(
        ax,
        xy=(0.555, 0.47),
        width=0.37,
        height=0.07,
        text="$L_\\Delta$ optional: Delta loss / Sobolev-style method",
        facecolor=colors["loss"],
        edgecolor="#815681",
        fontsize=6.6,
        weight="bold",
    )
    add_box(
        ax,
        xy=(0.555, 0.36),
        width=0.37,
        height=0.075,
        text="Structure: $\\theta$ terminal scaling + softplus premium\nobstacle consistency by reconstruction",
        facecolor="#eaddec",
        edgecolor="#815681",
        fontsize=6.3,
        weight="bold",
    )
    add_box(
        ax,
        xy=(0.065, 0.32),
        width=0.225,
        height=0.065,
        text="CN/PSOR remains benchmark\nand label generator",
        facecolor=colors["psor"],
        edgecolor="#516287",
        fontsize=6.2,
        weight="bold",
    )
    add_box(
        ax,
        xy=(0.305, 0.06),
        width=0.38,
        height=0.13,
        text="$L(\\vartheta)=L_{\\mathrm{sup}}+\\lambda_{\\mathrm{PDE}}L_{\\mathrm{PDE}}+\\lambda_\\Delta L_\\Delta$\noptimizer updates network parameters $\\vartheta$",
        facecolor="#98c8a4",
        edgecolor="#477a52",
        fontsize=7.2,
        weight="bold",
    )
    add_box(
        ax,
        xy=(0.715, 0.055),
        width=0.245,
        height=0.14,
        text="Evaluation gates vs matched\nsupervised neural baseline\nprice RMSE | residual\nboundary RMSE | strict $\\Delta$ RMSE",
        facecolor=colors["gate"],
        edgecolor="#718096",
        fontsize=6.2,
        weight="bold",
    )

    add_arrow(ax, (0.305, 0.695), (0.265, 0.613))
    add_arrow(ax, (0.245, 0.525), (0.228, 0.515))
    add_arrow(ax, (0.270, 0.47), (0.405, 0.55), color="#516287")
    add_arrow(ax, (0.46, 0.745), (0.555, 0.75), color="#516287")
    add_arrow(ax, (0.46, 0.645), (0.555, 0.625), color="#516287")
    add_arrow(ax, (0.46, 0.445), (0.555, 0.505), color="#516287")
    add_arrow(ax, (0.73, 0.36), (0.55, 0.19), color="#477a52")
    add_arrow(
        ax,
        (0.495, 0.19),
        (0.145, 0.755),
        color="#d95f2a",
        connectionstyle="arc3,rad=-0.36",
        zorder=0.2,
    )
    add_arrow(ax, (0.685, 0.125), (0.715, 0.125), color="#718096")

    ax.text(
        0.50,
        0.225,
        "feedback objective",
        ha="center",
        va="bottom",
        fontsize=6.7,
        color="#477a52",
        transform=ax.transAxes,
    )
    save_all(fig, "pinn_project_mechanism_illustration")


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
    save_pinn_project_mechanism_illustration()
    save_stage23_metric_tradeoff()
    save_stage23_training_curves()
    save_premium_mask_panel()
    for name in [
        "pinn_project_mechanism_illustration",
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
