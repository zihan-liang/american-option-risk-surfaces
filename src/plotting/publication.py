"""Publication-facing plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from plotting.report_visuals import format_display_label


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def save_schematic(path: str | Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(8.2, 2.6))
    ax.axis("off")
    labels = [
        "CN/PSOR\nbenchmark",
        "Diagnostics\nand masks",
        "Clean PSOR\ndataset",
        "Separate heads\nprice, boundary, $\\Delta$",
        "PSOR audit\nfor screening",
    ]
    xs = np.linspace(0.08, 0.92, len(labels))
    for x, label in zip(xs, labels):
        ax.text(x, 0.55, label, ha="center", va="center", bbox={"boxstyle": "round,pad=0.35", "fc": "#edf2f7", "ec": "#2d3748"})
    for left, right in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(right - 0.075, 0.55), xytext=(left + 0.075, 0.55), arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.text(0.5, 0.12, "Neural components are scoped diagnostics, not replacements for the free-boundary solver.", ha="center")
    fig.savefig(path)
    plt.close(fig)


def save_two_region_boundary_figure(path: str | Path) -> None:
    _style()
    tau = np.linspace(0.02, 2.0, 160)
    put_boundary = 0.7 + 0.22 * (1 - np.exp(-1.8 * tau))
    call_boundary = 1.45 + 0.34 * np.exp(-1.2 * tau)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), sharey=True)
    for ax, boundary, title, exercise_side in [
        (axes[0], put_boundary, "American put", "below"),
        (axes[1], call_boundary, "Dividend-paying call", "above"),
    ]:
        ax.plot(tau, boundary, color="#1f77b4", lw=2, label="free boundary")
        if exercise_side == "below":
            ax.fill_between(tau, 0.35, boundary, color="#fed7d7", alpha=0.8, label="exercise")
            ax.fill_between(tau, boundary, 2.05, color="#c6f6d5", alpha=0.75, label="continuation")
        else:
            ax.fill_between(tau, 0.35, boundary, color="#c6f6d5", alpha=0.75, label="continuation")
            ax.fill_between(tau, boundary, 2.05, color="#fed7d7", alpha=0.8, label="exercise")
        ax.set_title(title)
        ax.set_xlabel(r"time to maturity $\tau$")
        ax.set_xlim(tau.min(), tau.max())
        ax.set_ylim(0.35, 2.05)
    axes[0].set_ylabel(r"$S/K$")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def save_image_panel(path: str | Path, source: str | Path | None, title: str) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.set_title(title)
    if source and Path(source).exists():
        image = mpimg.imread(source)
        ax.imshow(image)
        ax.axis("off")
    else:
        ax.text(0.5, 0.5, "Source figure unavailable", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(path)
    plt.close(fig)


def save_bar_figure(
    path: str | Path,
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    *,
    rotate: bool = False,
    orientation: str = "vertical",
    display_labels: dict[str, str] | None = None,
) -> None:
    _style()
    formatted = [format_display_label(label, display_labels) for label in labels]
    horizontal = orientation.lower() == "horizontal"
    if horizontal:
        fig, ax = plt.subplots(figsize=(7.2, max(3.6, 0.5 * max(len(formatted), 1))))
    else:
        fig_width = max(5.8, min(10.0, 1.0 + 0.8 * max(len(formatted), 1)))
        fig, ax = plt.subplots(figsize=(fig_width, 3.6))
    if labels and values:
        if horizontal:
            positions = np.arange(len(values))
            ax.barh(positions, values, color="#2b6cb0")
            ax.set_yticks(positions)
            ax.set_yticklabels(formatted)
            ax.invert_yaxis()
            ax.set_xlabel(ylabel)
            ax.grid(axis="x", alpha=0.25)
        else:
            ax.bar(range(len(values)), values, color="#2b6cb0")
            ax.set_xticks(range(len(formatted)))
            ax.set_xticklabels(
                formatted,
                rotation=30 if rotate else 0,
                ha="right" if rotate else "center",
            )
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)
