"""Generate four high-quality figures explaining call and put payoffs.

Four panels (each saved as a separate PNG and PDF):

1. ``call_payoff_2d``  : Call payoff as a function of stock price S.
2. ``call_positive_region`` : Region in the (t, S) plane where call payoff is positive.
3. ``put_payoff_2d``   : Put payoff as a function of stock price S.
4. ``put_positive_region``  : Region in the (t, S) plane where put payoff is positive.

A combined 2x2 panel figure is also produced for convenience.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Professional, restrained palette.
BLACK = "#1f2933"
GREY = "#6b7280"
BLUE = "#1f4e79"
RED = "#b3322c"
LIGHT_BLUE = "#9bbbd9"
LIGHT_RED = "#d99b97"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": BLACK,
    "axes.labelcolor": BLACK,
    "xtick.color": BLACK,
    "ytick.color": BLACK,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

K = 100.0
S_MAX = 200.0


# ---------------------------------------------------------------------------
# 2D payoff helpers
# ---------------------------------------------------------------------------

def _draw_payoff_2d(ax, kind: str) -> None:
    s = np.linspace(0.0, S_MAX, 400)
    if kind == "call":
        phi = np.maximum(s - K, 0.0)
        colour = BLUE
        formula = r"$\Phi_{\mathrm{Call}}(S)=\max(S-K,0)$"
        title = "Call payoff"
    else:
        phi = np.maximum(K - s, 0.0)
        colour = RED
        formula = r"$\Phi_{\mathrm{Put}}(S)=\max(K-S,0)$"
        title = "Put payoff"

    ax.plot(s, phi, color=colour, lw=3.2, zorder=3)

    # Axes with arrowheads via annotate.
    y_top = max(phi.max(), K) * 1.15
    ax.set_xlim(-0.05 * S_MAX, S_MAX * 1.08)
    ax.set_ylim(-0.05 * y_top, y_top)
    ax.annotate("", xy=(S_MAX * 1.05, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.2))
    ax.annotate("", xy=(0, y_top * 0.98), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.2))

    # Strike marker.
    ax.plot([K, K], [0, 0], marker="|", color=BLACK, ms=10, mew=2)
    ax.text(K, -0.08 * y_top, "$K$", ha="center", va="top",
            color=BLACK, fontsize=16)

    # Axis labels.
    ax.text(S_MAX * 1.06, -0.04 * y_top, "$S$", color=BLACK, fontsize=18,
            ha="left", va="top")
    ax.text(-0.02 * S_MAX, y_top * 1.02, r"$\Phi$", color=BLACK, fontsize=18,
            ha="right", va="bottom")

    # Title and formula.
    ax.text(0.04 * S_MAX, y_top * 0.92, title,
            color=BLACK, fontsize=20, ha="left", va="top")
    ax.text(0.04 * S_MAX, y_top * 0.78, formula,
            color=colour, fontsize=15, ha="left", va="top")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("left", "bottom"):
        ax.spines[spine].set_visible(False)


# ---------------------------------------------------------------------------
# Payoff-positive region helpers
# ---------------------------------------------------------------------------

def _draw_positive_region(ax, kind: str) -> None:
    """Draw the region in the (t, S) plane where the payoff is positive."""
    if kind == "call":
        colour = BLUE
        fill = LIGHT_BLUE
        title = "Call payoff-positive region"
        condition = r"$S>K \Rightarrow \Phi_{\mathrm{Call}}(S)>0$"
        y0, y1 = K, S_MAX
        label_y = K + 0.55 * (S_MAX - K)
    else:
        colour = RED
        fill = LIGHT_RED
        title = "Put payoff-positive region"
        condition = r"$S<K \Rightarrow \Phi_{\mathrm{Put}}(S)>0$"
        y0, y1 = 0.0, K
        label_y = 0.55 * K

    ax.set_xlim(-0.08, 1.12)
    ax.set_ylim(-0.05 * S_MAX, 1.08 * S_MAX)
    ax.axis("off")

    # Axes: horizontal time, vertical spot price.
    ax.annotate("", xy=(1.05, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.4))
    ax.annotate("", xy=(0, 1.02 * S_MAX), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.4))
    ax.text(1.08, -0.04 * S_MAX, "$t$", color=BLACK, fontsize=18, ha="left", va="top")
    ax.text(-0.04, 1.04 * S_MAX, "$S$", color=BLACK, fontsize=18, ha="right", va="bottom")

    # Strike line and positive-payoff region.
    ax.axhline(K, xmin=0.07, xmax=0.94, color=GREY, lw=1.6, ls="--")
    ax.fill_between([0, 1], y0, y1, color=fill, alpha=0.55, edgecolor=colour, linewidth=2.0)

    # Boundary and labels.
    ax.text(-0.025, K, "$K$", color=BLACK, fontsize=16, ha="right", va="center")
    ax.text(0.06, 0.98 * S_MAX, title, color=BLACK, fontsize=18, ha="left", va="top")
    ax.text(0.12, label_y, condition, color=colour, fontsize=15, ha="left", va="center")
    ax.text(0.12, label_y - 0.12 * S_MAX, "payoff can be created here",
            color=colour, fontsize=13, ha="left", va="center")


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "pdf"):
        out = OUTPUT_DIR / f"{name}.{ext}"
        fig.savefig(out)
        print(out)


def main() -> None:
    # 1. Call payoff 2D.
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    _draw_payoff_2d(ax, "call")
    _save(fig, "call_payoff_2d")
    plt.close(fig)

    # 2. Call payoff-positive region in the (t, S) plane.
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    _draw_positive_region(ax, "call")
    _save(fig, "call_positive_region")
    plt.close(fig)

    # 3. Put payoff 2D.
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    _draw_payoff_2d(ax, "put")
    _save(fig, "put_payoff_2d")
    plt.close(fig)

    # 4. Put payoff-positive region in the (t, S) plane.
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    _draw_positive_region(ax, "put")
    _save(fig, "put_positive_region")
    plt.close(fig)

    # Combined 2x2 panel.
    fig = plt.figure(figsize=(13.0, 9.0))
    ax_a = fig.add_subplot(2, 2, 1)
    _draw_payoff_2d(ax_a, "call")
    ax_b = fig.add_subplot(2, 2, 2)
    _draw_positive_region(ax_b, "call")
    ax_c = fig.add_subplot(2, 2, 3)
    _draw_payoff_2d(ax_c, "put")
    ax_d = fig.add_subplot(2, 2, 4)
    _draw_positive_region(ax_d, "put")
    fig.tight_layout()
    _save(fig, "call_put_payoffs_2x2_regions")
    plt.close(fig)


if __name__ == "__main__":
    main()
