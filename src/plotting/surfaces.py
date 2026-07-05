"""Matplotlib helpers for solver and risk-surface figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_price_surface(
    spots: np.ndarray,
    taus: np.ndarray,
    values: np.ndarray,
    path: str | Path,
    title: str,
    *,
    xlabel: str = r"$S/K$",
    ylabel: str = r"Time to maturity, $\tau$",
    colorbar_label: str = r"$U/K$",
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    mesh = ax.pcolormesh(spots, taus, values, shading="auto")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_curve(
    x: np.ndarray,
    y: np.ndarray,
    path: str | Path,
    title: str,
    ylabel: str,
    *,
    xlabel: str = r"Time to maturity, $\tau$",
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
