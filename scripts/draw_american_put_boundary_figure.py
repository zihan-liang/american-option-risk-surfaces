"""Draw a high-quality conceptual American-put free-boundary figure.

The uploaded sketch is about the optimal exercise boundary for an American put:

- For S <= S_f(t), immediate exercise is optimal and P(S,t) = E - S.
- For S > S_f(t), holding is optimal and P(S,t) > E - S.
- At the free boundary, value matching holds: P(S_f(t), t) = E - S_f(t).
- Smooth pasting holds: partial P / partial S at S_f(t) equals -1.
- The American put value lies above the European put because early exercise has value.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "figures"
PNG_PATH = OUT_DIR / "american_put_free_boundary_high_quality.png"
PDF_PATH = OUT_DIR / "american_put_free_boundary_high_quality.pdf"


BOUNDARY = "#6b7280"
ACCENT = "#b45309"
BLUE = "#1f4e79"
LIGHT_BLUE = "#5b8fc9"
PAYOFF_RED = "#9f4a4a"
BLACK = "#1f2937"
REGION_LEFT = "#f8fafc"
REGION_RIGHT = "#f1f5f9"


def add_axis_arrow(ax, start, end, label, label_offset=(0, 0), fontsize=30):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=22,
            linewidth=3.2,
            color=BLACK,
            shrinkA=0,
            shrinkB=0,
        )
    )
    ax.text(
        end[0] + label_offset[0],
        end[1] + label_offset[1],
        label,
        fontsize=fontsize,
        color=BLACK,
        ha="center",
        va="center",
    )


def add_arrow(ax, start, end, color=ACCENT, linewidth=3.0, mutation_scale=18):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(figsize=(15.5, 8.6), constrained_layout=False)
    ax.set_xlim(-0.10, 1.95)
    ax.set_ylim(-0.26, 1.28)
    ax.axis("off")

    # Conceptual parameters.
    E = 1.0
    sf = 0.50
    pf = E - sf
    s = np.linspace(0.0, 1.75, 700)
    payoff = np.maximum(E - s, 0.0)

    # A smooth continuation value satisfying value matching and smooth pasting.
    x = np.maximum(s - sf, 0.0)
    american_continuation = pf * np.exp(-x / pf)
    american = np.where(s <= sf, payoff, american_continuation)

    # A stylized European put, below the American put and without early exercise.
    european = 0.95 * np.exp(-2.15 * s) + 0.015
    european = np.minimum(european, american - 0.018)
    european = np.maximum(european, 0.018)

    # Axes.
    add_axis_arrow(ax, (0.0, 0.0), (1.70, 0.0), r"$S$", label_offset=(0.07, -0.06), fontsize=38)
    add_axis_arrow(ax, (0.0, 0.0), (0.0, 1.12), r"$P$", label_offset=(-0.07, 0.03), fontsize=38)

    # Region shading.
    ax.add_patch(Rectangle((0, 0), sf, 1.08, facecolor=REGION_LEFT, alpha=0.95, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((sf, 0), 1.70 - sf, 1.08, facecolor=REGION_RIGHT, alpha=0.95, edgecolor="none", zorder=0))

    # Curves.
    ax.plot(s, payoff, color=PAYOFF_RED, lw=3.2, label=r"payoff $=\max(E-S,0)$", zorder=2)
    ax.plot(s, american, color=BLUE, lw=3.5, label="American put value", zorder=3)
    ax.plot(s, european, color=LIGHT_BLUE, lw=3.0, label="European put value", zorder=2)

    # Free boundary and matching point.
    ax.plot([sf, sf], [0, pf + 0.28], color=BOUNDARY, lw=2.4, ls="--")
    ax.scatter([sf], [pf], s=95, color=ACCENT, edgecolor="white", linewidth=1.4, zorder=6)
    ax.annotate(
        r"$S_f(t)$",
        xy=(sf, pf),
        xytext=(sf + 0.13, pf + 0.25),
        arrowprops={"arrowstyle": "->", "lw": 2.2, "color": ACCENT},
        color=ACCENT,
        fontsize=25,
        ha="left",
    )

    # Exercise/hold interpretation.
    ax.text(0.05, 1.05, r"Exercise region: $S\leq S_f(t)$", color=BLACK, fontsize=21, ha="left")
    ax.text(0.09, 0.92, r"$P=E-S$", color=BLACK, fontsize=21, ha="left")
    ax.text(0.78, 1.05, r"Continuation / hold region: $S>S_f(t)$", color=BLACK, fontsize=21, ha="left")
    ax.text(0.83, 0.92, r"$P>E-S$", color=BLACK, fontsize=21, ha="left")

    # Curve labels.
    ax.annotate(
        "American put",
        xy=(1.25, np.interp(1.25, s, american)),
        xytext=(1.42, 0.27),
        arrowprops={"arrowstyle": "->", "lw": 1.8, "color": BLUE},
        color=BLUE,
        fontsize=20,
        ha="left",
    )
    ax.annotate(
        "European put",
        xy=(1.18, np.interp(1.18, s, european)),
        xytext=(1.42, 0.14),
        arrowprops={"arrowstyle": "->", "lw": 1.8, "color": LIGHT_BLUE},
        color=LIGHT_BLUE,
        fontsize=20,
        ha="left",
    )
    ax.text(1.03, 0.075, "payoff", color=PAYOFF_RED, fontsize=19, ha="left")
    ax.text(E, -0.06, r"$E$", color=PAYOFF_RED, fontsize=25, ha="center", va="top")

    # Tangent line at the boundary to show smooth pasting slope -1.
    tangent_s = np.linspace(sf - 0.22, sf + 0.26, 2)
    tangent = pf - (tangent_s - sf)
    ax.plot(tangent_s, tangent, color=ACCENT, lw=2.2, alpha=0.76)


    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(PNG_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
