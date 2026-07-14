"""Draw a concise worked computational graph for reverse-mode autodiff."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch


REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_STEM = REPO_ROOT / "output/figures/autodiff_worked_example"

NODE_LABELS = {
    "x": r"$x=2$",
    "a": r"$a=x^2=4$",
    "b": r"$b=3x=6$",
    "c": r"$c=a+b=10$",
    "f": r"$f=\sin(c)=\sin(10)$",
}

LOCAL_DERIVATIVES = {
    ("x", "a"): r"$4$",
    ("x", "b"): r"$3$",
    ("a", "c"): r"$1$",
    ("b", "c"): r"$1$",
    ("c", "f"): r"$\cos(10)$",
}

FINAL_GRADIENT = r"$\dfrac{df}{dx}=7\cos(10)$"

INK = "#172033"
FORWARD = "#356c9b"
BACKWARD = "#d07a38"
NODE_FILL = "#f8fafc"
NODE_EDGE = "#536273"


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    curve: float,
    shrink_start: float,
    shrink_end: float,
    zorder: int = 1,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.65,
            color=color,
            connectionstyle=f"arc3,rad={curve}",
            shrinkA=shrink_start,
            shrinkB=shrink_end,
            zorder=zorder,
        )
    )


def add_node(
    ax: plt.Axes,
    center: tuple[float, float],
    label: str,
    *,
    radius: float,
    fontsize: float,
) -> None:
    ax.add_patch(
        Circle(
            center,
            radius=radius,
            facecolor=NODE_FILL,
            edgecolor=NODE_EDGE,
            linewidth=1.35,
            zorder=3,
        )
    )
    ax.text(
        *center,
        label,
        ha="center",
        va="center",
        color=INK,
        fontsize=fontsize,
        zorder=4,
    )


def make_figure() -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "savefig.dpi": 300,
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    ax.set_xlim(-0.15, 10.15)
    ax.set_ylim(-0.45, 4.05)
    ax.axis("off")

    positions = {
        "x": (0.85, 2.0),
        "a": (3.25, 3.0),
        "b": (3.25, 1.0),
        "c": (6.10, 2.0),
        "f": (8.95, 2.0),
    }
    node_radius_points = {"x": 37, "a": 47, "b": 47, "c": 49, "f": 64}

    ax.text(
        5.0,
        3.83,
        r"$f(x)=\sin(x^2+3x),\quad x=2$",
        ha="center",
        va="center",
        fontsize=14,
        color=INK,
        weight="semibold",
    )

    forward_edges = [("x", "a"), ("x", "b"), ("a", "c"), ("b", "c"), ("c", "f")]
    for source, target in forward_edges:
        add_arrow(
            ax,
            positions[source],
            positions[target],
            color=FORWARD,
            curve=0.0,
            shrink_start=node_radius_points[source],
            shrink_end=node_radius_points[target],
        )

    # Reverse arrows occupy parallel arcs, leaving the graph structure primary.
    reverse_edges = [("a", "x", 0.18), ("b", "x", -0.18), ("c", "a", 0.18), ("c", "b", -0.18), ("f", "c", 0.16)]
    for source, target, curve in reverse_edges:
        add_arrow(
            ax,
            positions[source],
            positions[target],
            color=BACKWARD,
            curve=curve,
            shrink_start=node_radius_points[source],
            shrink_end=node_radius_points[target],
            zorder=2,
        )

    add_node(ax, positions["x"], NODE_LABELS["x"], radius=0.61, fontsize=15)
    add_node(ax, positions["a"], NODE_LABELS["a"], radius=0.77, fontsize=12.5)
    add_node(ax, positions["b"], NODE_LABELS["b"], radius=0.77, fontsize=12.5)
    add_node(ax, positions["c"], NODE_LABELS["c"], radius=0.80, fontsize=12.5)
    add_node(ax, positions["f"], NODE_LABELS["f"], radius=1.04, fontsize=11.2)

    derivative_labels = [
        (2.00, 3.42, LOCAL_DERIVATIVES[("x", "a")]),
        (2.00, 0.57, LOCAL_DERIVATIVES[("x", "b")]),
        (4.72, 3.29, LOCAL_DERIVATIVES[("a", "c")]),
        (4.72, 0.72, LOCAL_DERIVATIVES[("b", "c")]),
        (7.53, 2.48, LOCAL_DERIVATIVES[("c", "f")]),
    ]
    for x, y, label in derivative_labels:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=11,
            color=BACKWARD,
            bbox={"boxstyle": "round,pad=0.10", "fc": "white", "ec": "none"},
            zorder=5,
        )

    ax.text(
        5.0,
        -0.18,
        FINAL_GRADIENT,
        ha="center",
        va="center",
        fontsize=15,
        color=INK,
        weight="semibold",
    )
    fig.tight_layout(pad=0.35)
    return fig


def main() -> None:
    FIGURE_STEM.parent.mkdir(parents=True, exist_ok=True)
    figure = make_figure()
    figure.savefig(FIGURE_STEM.with_suffix(".pdf"), facecolor="white", bbox_inches="tight", pad_inches=0.08)
    figure.savefig(FIGURE_STEM.with_suffix(".png"), facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


if __name__ == "__main__":
    main()
