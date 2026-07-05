"""Draw clean payoff diagrams matching the uploaded option-payoff sketches.

The uploaded image contains two payoff shapes:
1. A long call payoff: max(S - E, 0).
2. A put-shaped payoff drawn above the axis: max(E - S, 0).

Note: the handwriting says "put-sell", but the drawn curve is the standard
long put payoff shape. A short/sold put would be -max(E - S, 0), below the axis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "figures"

STRIKE = 1.0
MAX_S = 2.1

NAVY = "#1f2937"
BLUE = "#1f4e79"
RED = "#9f4a4a"
GRID = "#e5e7eb"


def axis_arrow(ax, start, end, label, label_offset=(0.0, 0.0)) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=18,
            linewidth=2.4,
            color=NAVY,
            shrinkA=0,
            shrinkB=0,
        )
    )
    ax.text(
        end[0] + label_offset[0],
        end[1] + label_offset[1],
        label,
        fontsize=22,
        color=NAVY,
        ha="center",
        va="center",
    )


def base_axes(ax, *, y_max: float = 1.25) -> None:
    ax.set_xlim(-0.08, MAX_S)
    ax.set_ylim(-0.16, y_max)
    ax.axis("off")
    axis_arrow(ax, (0, 0), (MAX_S - 0.08, 0), r"$S$", label_offset=(0.06, -0.04))
    axis_arrow(ax, (0, 0), (0, y_max - 0.10), r"$\Phi$", label_offset=(-0.04, 0.08))
    ax.axvline(STRIKE, ymin=0.08, ymax=0.17, color=NAVY, lw=1.8)
    ax.text(STRIKE, -0.08, r"$E$", fontsize=18, color=NAVY, ha="center", va="top")


def save_figure(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{name}.png"
    pdf = OUT_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)


def draw_long_call() -> None:
    """Long call payoff: max(S - E, 0)."""
    s = np.linspace(0, MAX_S, 500)
    payoff = np.maximum(s - STRIKE, 0)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    base_axes(ax)
    ax.plot(s, payoff, color=BLUE, lw=3.0, solid_capstyle="round")
    ax.scatter([STRIKE], [0], s=42, color=BLUE, zorder=4)

    ax.text(0.10, 1.12, "Call payoff", fontsize=22, color=NAVY, ha="left")
    ax.text(0.16, 0.38, r"$\Phi(S)=\max(S-E,0)$", fontsize=17, color=BLUE, ha="left")
    ax.text(1.08, -0.11, "strike price", fontsize=13, color=NAVY, ha="left", va="top")
    ax.plot([0, MAX_S - 0.08], [0, 0], color=NAVY, lw=2.0)
    ax.grid(False)

    save_figure(fig, "call_buy_payoff")


def draw_put_as_drawn() -> None:
    """Put-shaped payoff drawn in the screenshot: max(E - S, 0)."""
    s = np.linspace(0, MAX_S, 500)
    payoff = np.maximum(STRIKE - s, 0)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    base_axes(ax)
    ax.plot(s, payoff, color=RED, lw=3.0, solid_capstyle="round")
    ax.scatter([STRIKE], [0], s=42, color=RED, zorder=4)

    ax.text(0.10, 1.12, "Put payoff", fontsize=22, color=NAVY, ha="left")
    ax.text(1.12, 0.48, r"$\Phi(S)=\max(E-S,0)$", fontsize=17, color=RED, ha="left")
    ax.text(1.08, -0.11, "strike price", fontsize=13, color=NAVY, ha="left", va="top")
    ax.plot([0, MAX_S - 0.08], [0, 0], color=NAVY, lw=2.0)
    ax.grid(False)

    save_figure(fig, "put_payoff_as_drawn")


def draw_combined() -> None:
    """Combined two-panel version for easy side-by-side use."""
    s = np.linspace(0, MAX_S, 500)
    call_payoff = np.maximum(s - STRIKE, 0)
    put_payoff = np.maximum(STRIKE - s, 0)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.4))

    for ax, title, payoff, color, formula in [
        (axes[0], "Call payoff", call_payoff, BLUE, r"$\Phi(S)=\max(S-E,0)$"),
        (axes[1], "Put payoff", put_payoff, RED, r"$\Phi(S)=\max(E-S,0)$"),
    ]:
        base_axes(ax)
        ax.plot(s, payoff, color=color, lw=3.0, solid_capstyle="round")
        ax.scatter([STRIKE], [0], s=42, color=color, zorder=4)
        ax.text(0.10, 1.12, title, fontsize=21, color=NAVY, ha="left")
        formula_x = 1.10 if title.startswith("Put") else 0.18
        formula_y = 0.48 if title.startswith("Put") else 0.38
        ax.text(formula_x, formula_y, formula, fontsize=15, color=color, ha="left")

    fig.subplots_adjust(wspace=0.28)
    save_figure(fig, "basic_option_payoffs_two_panel")


def main() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.edgecolor": NAVY,
            "grid.color": GRID,
        }
    )
    draw_long_call()
    draw_put_as_drawn()
    draw_combined()


if __name__ == "__main__":
    main()
