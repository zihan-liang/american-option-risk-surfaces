"""CSV-only publication plots for the HD3 visualization package."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from hd_pipeline.hd3_visualization_core import (
    CANONICAL_DIMENSIONS,
    MODEL_IDS,
    write_rows_csv,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_figure_metrics(
    *, dimension_summary_path: Path, structural_path: Path, output_path: Path
) -> Path:
    scaling = _read_csv(dimension_summary_path)
    structural = _read_csv(structural_path)
    structural_by_key = {
        (int(row["dimension"]), row["model_id"], row["run_class"]): row
        for row in structural
    }
    output: list[dict[str, Any]] = []
    required_scaling = (
        "normalized_price_rmse", "delta_rmse", "trusted_pde_p95", "training_seconds"
    )
    required_structural = ("permutation_consistency_rmse", "transverse_invariance_rmse")
    for row in scaling:
        key = (int(row["dimension"]), row["model_id"], row["run_class"])
        if key not in structural_by_key:
            raise RuntimeError(f"missing_structural_metrics:{key}")
        srow = structural_by_key[key]
        merged: dict[str, Any] = {
            "dimension": key[0], "model_id": key[1], "run_class": key[2]
        }
        for field in required_scaling:
            if field not in row:
                raise RuntimeError(f"missing_scaling_metric:{field}:{key}")
            merged[field] = float(row[field])
        for field in required_structural:
            if field not in srow:
                raise RuntimeError(f"missing_structural_metric:{field}:{key}")
            merged[field] = float(srow[field])
        output.append(merged)
    output.sort(key=lambda r: (int(r["dimension"]), str(r["model_id"])))
    return write_rows_csv(output_path, output)


def _pyplot():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt


def _save_figure(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    _pyplot().close(fig)
    return path


def _series_by_model(rows: list[dict[str, str]], field: str):
    result = {}
    for model_id in MODEL_IDS:
        subset = sorted(
            (r for r in rows if r["model_id"] == model_id),
            key=lambda r: int(r["dimension"]),
        )
        result[model_id] = (
            np.asarray([int(r["dimension"]) for r in subset], dtype=int),
            np.asarray([float(r[field]) for r in subset], dtype=float),
        )
    return result


def _plot_model_scaling(ax, rows: list[dict[str, str]], field: str, ylabel: str):
    for model_id, (dimensions, values) in _series_by_model(rows, field).items():
        canonical = dimensions != 100
        line, = ax.plot(dimensions[canonical], values[canonical], marker="o", label=model_id.replace("_", "-"))
        if np.any(~canonical):
            stress_d = dimensions[~canonical]
            stress_v = values[~canonical]
            ax.plot(
                np.concatenate(([dimensions[canonical][-1]], stress_d)),
                np.concatenate(([values[canonical][-1]], stress_v)),
                linestyle="--", marker=None, color=line.get_color(), alpha=0.7,
            )
            ax.scatter(stress_d, stress_v, marker="X", s=65, color=line.get_color())
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel(ylabel)
    ax.set_xticks([10, 20, 50, 100])
    ax.grid(alpha=0.2)


def plot_scaling_summary(*, data_dir: Path, figure_dir: Path) -> Path:
    rows = _read_csv(Path(data_dir) / "figure_metrics.csv")
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    specs = [
        ("normalized_price_rmse", "Normalized price RMSE"),
        ("delta_rmse", "Delta RMSE"),
        ("trusted_pde_p95", "Trusted PDE residual P95"),
        ("training_seconds", "Training time (s)"),
    ]
    for ax, (field, ylabel) in zip(axes.ravel(), specs):
        _plot_model_scaling(ax, rows, field, ylabel)
        ax.annotate("100D stress", xy=(100, ax.get_ylim()[0]), xytext=(-4, 8), textcoords="offset points", ha="right", va="bottom", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("HD3 dimensional scaling: canonical 10D–50D with 100D stress")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save_figure(fig, Path(figure_dir) / "hd3_scaling_summary.png")


def plot_structural_diagnostics(*, data_dir: Path, figure_dir: Path) -> Path:
    rows = _read_csv(Path(data_dir) / "figure_metrics.csv")
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    _plot_model_scaling(axes[0], rows, "permutation_consistency_rmse", "Permutation consistency RMSE")
    axes[0].set_yscale("log")
    _plot_model_scaling(axes[1], rows, "transverse_invariance_rmse", "Transverse invariance RMSE")
    for ax in axes:
        ax.annotate("100D stress", xy=(100, ax.get_ylim()[0]), xytext=(-4, 8), textcoords="offset points", ha="right", va="bottom", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("HD3 structural diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return _save_figure(fig, Path(figure_dir) / "hd3_structural_diagnostics.png")


def _truthy(value: str | bool) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def plot_price_slices(*, data_dir: Path, figure_dir: Path) -> Path:
    rows = _read_csv(Path(data_dir) / "diagonal_slices.csv")
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3), sharey=True)
    all_pass = True
    for ax, d in zip(axes, CANONICAL_DIMENSIONS):
        subset = sorted(
            (r for r in rows if int(r["dimension"]) == d and math.isclose(float(r["tau"]), 1.0)),
            key=lambda r: float(r["S"]),
        )
        if not subset:
            raise RuntimeError(f"missing_price_slice:d{d}:tau1")
        s = np.asarray([float(r["S"]) for r in subset])
        all_pass = all_pass and all(_truthy(r["reference_agreement_passed"]) for r in subset)
        ax.plot(s, [float(r["cnpsor"]) for r in subset], linewidth=2.0, label="CN/PSOR Oracle")
        ax.plot(s, [float(r["crr"]) for r in subset], linestyle="--", label="CRR Binomial Tree")
        ax.plot(s, [float(r["fc_hvi"]) for r in subset], linestyle="-.", label="FC-HVI")
        ax.plot(s, [float(r["deepsets_hvi"]) for r in subset], linestyle=":", linewidth=2.0, label="DeepSets-HVI")
        ax.axvline(1.0, linestyle=":", linewidth=1.0, alpha=0.7)
        ax.set_title(f"{d}D canonical")
        ax.set_xlabel(r"Common asset price $S$ ($S_1=\cdots=S_d=S$)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Normalized American put value")
    handles, labels = axes[0].get_legend_handles_labels()
    if not all_pass:
        labels = ["CRR cross-check" if label == "CRR Binomial Tree" else label for label in labels]
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(r"HD3 symmetric-diagonal price slices at $\tau=1$")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return _save_figure(fig, Path(figure_dir) / "hd3_price_slices.png")


def _surface_matrix(rows: list[dict[str, str]], field: str):
    spots = np.asarray(sorted({float(r["S"]) for r in rows}), dtype=float)
    taus = np.asarray(sorted({float(r["tau"]) for r in rows}), dtype=float)
    lookup = {(float(r["tau"]), float(r["S"])): float(r[field]) for r in rows}
    matrix = np.asarray([[lookup[(t, s)] for s in spots] for t in taus], dtype=float)
    return spots, taus, matrix


def plot_error_heatmaps(*, data_dir: Path, figure_dir: Path) -> Path:
    surfaces = {d: _read_csv(Path(data_dir) / f"error_surface_{d}d.csv") for d in CANONICAL_DIMENSIONS}
    maxima = []
    for rows in surfaces.values():
        maxima.extend(float(r["fc_abs_error"]) for r in rows)
        maxima.extend(float(r["deepsets_abs_error"]) for r in rows)
    vmax = max(maxima) if maxima else 1.0
    if vmax <= 0.0:
        vmax = 1.0
    plt = _pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.0), sharex=True, sharey=True)
    image = None
    for col, d in enumerate(CANONICAL_DIMENSIONS):
        for row_idx, (field, model_label) in enumerate((("fc_abs_error", "FC-HVI"), ("deepsets_abs_error", "DeepSets-HVI"))):
            spots, taus, matrix = _surface_matrix(surfaces[d], field)
            image = axes[row_idx, col].pcolormesh(spots, taus, matrix, shading="auto", vmin=0.0, vmax=vmax)
            axes[row_idx, col].set_title(f"{model_label}, {d}D")
            axes[row_idx, col].set_xlabel("Common asset price $S$")
            if col == 0:
                axes[row_idx, col].set_ylabel(r"Time to maturity $\tau$")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label="Absolute error vs CN/PSOR", shrink=0.90)
    fig.suptitle("HD3 symmetric-diagonal price-error landscapes")
    fig.subplots_adjust(top=0.90, right=0.90, wspace=0.12, hspace=0.25)
    return _save_figure(fig, Path(figure_dir) / "hd3_error_heatmaps.png")


def plot_exercise_boundaries(*, data_dir: Path, figure_dir: Path) -> Path:
    rows = _read_csv(Path(data_dir) / "exercise_boundaries.csv")
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharey=True)
    for ax, d in zip(axes, CANONICAL_DIMENSIONS):
        subset = sorted((r for r in rows if int(r["dimension"]) == d), key=lambda r: float(r["tau"]))
        tau = np.asarray([float(r["tau"]) for r in subset])
        ax.plot(tau, [float(r["cnpsor_boundary"]) for r in subset], linewidth=2.0, label="CN/PSOR Oracle")
        ax.plot(tau, [float(r["fc_hvi_boundary"]) for r in subset], linestyle="--", label="FC-HVI")
        ax.plot(tau, [float(r["deepsets_hvi_boundary"]) for r in subset], linestyle=":", linewidth=2.0, label="DeepSets-HVI")
        ax.set_title(f"{d}D canonical")
        ax.set_xlabel(r"Time to maturity $\tau$")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(r"Exercise boundary $S^*(\tau)$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("HD3 exercise-boundary diagnostic")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return _save_figure(fig, Path(figure_dir) / "hd3_exercise_boundaries.png")


def plot_100d_stress_slice(*, data_dir: Path, figure_dir: Path) -> Path:
    rows = _read_csv(Path(data_dir) / "diagonal_slices.csv")
    subset = sorted(
        (r for r in rows if int(r["dimension"]) == 100 and math.isclose(float(r["tau"]), 1.0)),
        key=lambda r: float(r["S"]),
    )
    if not subset:
        raise RuntimeError("missing_100d_stress_slice")
    passed = all(_truthy(r["reference_agreement_passed"]) for r in subset)
    s = np.asarray([float(r["S"]) for r in subset])
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(s, [float(r["cnpsor"]) for r in subset], linewidth=2.0, label="CN/PSOR Oracle")
    ax.plot(s, [float(r["crr"]) for r in subset], linestyle="--", label="CRR Binomial Tree" if passed else "CRR cross-check")
    ax.plot(s, [float(r["fc_hvi"]) for r in subset], linestyle="-.", label="FC-HVI")
    ax.plot(s, [float(r["deepsets_hvi"]) for r in subset], linestyle=":", linewidth=2.0, label="DeepSets-HVI")
    ax.axvline(1.0, linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(r"Common asset price $S$ ($S_1=\cdots=S_{100}=S$)")
    ax.set_ylabel("Normalized American put value")
    ax.set_title(r"100D stress: symmetric-diagonal price slice at $\tau=1$")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    return _save_figure(fig, Path(figure_dir) / "hd3_100d_stress_slice.png")


def generate_all_figures(*, data_dir: Path, figure_dir: Path) -> list[Path]:
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_scaling_summary(data_dir=data_dir, figure_dir=figure_dir),
        plot_price_slices(data_dir=data_dir, figure_dir=figure_dir),
        plot_error_heatmaps(data_dir=data_dir, figure_dir=figure_dir),
        plot_structural_diagnostics(data_dir=data_dir, figure_dir=figure_dir),
        plot_exercise_boundaries(data_dir=data_dir, figure_dir=figure_dir),
        plot_100d_stress_slice(data_dir=data_dir, figure_dir=figure_dir),
    ]
