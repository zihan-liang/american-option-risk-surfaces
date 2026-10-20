"""Draw a teaching figure comparing activation smoothness for PINN autodiff."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPO_ROOT / "reports/figures"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
        }
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def save_activation_smoothness_comparison() -> None:
    x = np.linspace(-4.0, 4.0, 1200)
    positive = x >= 0.0
    negative = x < 0.0
    logistic = sigmoid(x)

    rows = [
        (
            "ReLU",
            np.maximum(x, 0.0),
            np.where(positive, 1.0, 0.0),
            np.zeros_like(x),
            "#c44e52",
        ),
        (
            "tanh",
            np.tanh(x),
            1.0 - np.tanh(x) ** 2,
            -2.0 * np.tanh(x) * (1.0 - np.tanh(x) ** 2),
            "#4c72b0",
        ),
        (
            "softplus",
            np.logaddexp(0.0, x),
            logistic,
            logistic * (1.0 - logistic),
            "#55a868",
        ),
        (
            "sine",
            np.sin(x),
            np.cos(x),
            -np.sin(x),
            "#8172b2",
        ),
    ]

    fig, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(8.6, 7.0),
        sharex=True,
        constrained_layout=True,
    )
    column_titles = [r"$f(x)$", r"$f'(x)$", r"$f''(x)$"]
    ylims = [(-1.25, 4.25), (-1.2, 1.25), (-1.25, 1.25)]

    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, pad=7)

    for row_idx, (name, f_value, first, second, color) in enumerate(rows):
        axes[row_idx, 0].set_ylabel(name, rotation=0, ha="right", va="center", labelpad=34, weight="bold")

        for col_idx, values in enumerate([f_value, first, second]):
            ax = axes[row_idx, col_idx]
            ax.axhline(0.0, color="#263238", linewidth=0.7, alpha=0.55)
            ax.axvline(0.0, color="#263238", linewidth=0.65, alpha=0.18)
            ax.set_xlim(-4.0, 4.0)
            ax.set_ylim(*ylims[col_idx])
            ax.plot(x, values, color=color, linewidth=1.8)
            if row_idx == 0:
                ax.axvline(0.0, color="#c44e52", linewidth=1.35, alpha=0.85)

        if row_idx == 0:
            relu_deriv_ax = axes[0, 1]
            relu_deriv_ax.lines[-1].remove()
            relu_deriv_ax.plot(x[negative], np.zeros_like(x[negative]), color=color, linewidth=1.8)
            relu_deriv_ax.plot(x[positive], np.ones_like(x[positive]), color=color, linewidth=1.8)
            relu_deriv_ax.scatter([0.0], [0.0], s=34, facecolors="white", edgecolors=color, linewidths=1.2, zorder=5)
            relu_deriv_ax.scatter([0.0], [1.0], s=34, facecolors=color, edgecolors=color, linewidths=1.2, zorder=5)

            relu_second_ax = axes[0, 2]
            relu_second_ax.lines[-1].remove()
            eps = 0.035
            left = x < -eps
            right = x > eps
            relu_second_ax.plot(x[left], np.zeros_like(x[left]), color=color, linewidth=1.8)
            relu_second_ax.plot(x[right], np.zeros_like(x[right]), color=color, linewidth=1.8)
            relu_second_ax.scatter([0.0], [0.0], marker="x", s=58, color=color, linewidths=1.7, zorder=5)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$x$")

    for ax in axes[:, 0]:
        ax.set_yticks([-1, 0, 1, 2, 3, 4])
    for ax in axes[:, 1]:
        ax.set_yticks([-1, 0, 1])
    for ax in axes[:, 2]:
        ax.set_yticks([-1, 0, 1])

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "pinn_activation_smoothness_comparison.pdf")
    fig.savefig(FIGURE_DIR / "pinn_activation_smoothness_comparison.png")
    plt.close(fig)


def main() -> None:
    apply_style()
    save_activation_smoothness_comparison()


if __name__ == "__main__":
    main()
