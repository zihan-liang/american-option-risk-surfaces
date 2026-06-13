"""Risk-map plotting helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_heatmap(x_labels: list[float], y_labels: list[float], values: np.ndarray, path: str | Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    mesh = ax.imshow(values, origin="lower", aspect="auto")
    ax.set_xticks(range(len(x_labels)), [str(x) for x in x_labels])
    ax.set_yticks(range(len(y_labels)), [str(y) for y in y_labels])
    ax.set_xlabel("sigma")
    ax.set_ylabel("q")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
