from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = REPO_ROOT / "results" / "hd3_matched_american_put_boundary_complete"
PLOT_OUT = RESULT_ROOT / "plots_stage1"
SUMMARY_CSV = PLOT_OUT / "hd_stage1_summary.csv"
BINOMIAL_CSV = PLOT_OUT / "binomial_convergence_stage1.csv"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}")


def save_figure(fig, out_base: Path, *, dpi: int = 300) -> None:
    ensure_dir(out_base.parent)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def read_run_metrics() -> pd.DataFrame:
    path = RESULT_ROOT / "run_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing final run metrics: {path}")

    df = pd.read_csv(path)
    require_columns(
        df,
        [
            "dimension",
            "model_id",
            "normalized_price_rmse",
            "reduced_delta_rmse",
            "reduced_gamma_rmse",
            "trusted_pde_p95",
            "permutation_consistency_rmse",
            "transverse_invariance_rmse",
            "training_seconds",
            "parameter_count",
            "peak_cuda_gib",
            "boundary_coverage",
            "reference_boundary_count",
            "predicted_boundary_count",
            "matched_boundary_count",
            "integrity",
        ],
        "run_metrics.csv",
    )

    # Stage 1 must use only the six canonical final rows.
    if "run_class" in df.columns:
        df = df[df["run_class"] == "canonical"].copy()
    if "split" in df.columns:
        df = df[df["split"] == "test"].copy()

    expected_dimensions = {10, 20, 50}
    observed_dimensions = set(int(v) for v in df["dimension"].unique())
    if observed_dimensions != expected_dimensions:
        raise ValueError(
            f"Expected canonical dimensions {sorted(expected_dimensions)}, "
            f"found {sorted(observed_dimensions)}"
        )

    display_names = {
        "FC_HVI": "FC-HVI",
        "DEEPSETS_HVI": "DeepSets-HVI",
    }
    unknown = sorted(set(df["model_id"]) - set(display_names))
    if unknown:
        raise ValueError(f"Unexpected model_id values: {unknown}")

    df["model"] = df["model_id"].map(display_names)

    if len(df) != 6:
        raise ValueError(f"Expected exactly six canonical test rows, found {len(df)}")
    if not df["integrity"].astype(bool).all():
        raise ValueError("At least one canonical result row has integrity=False")

    return df.sort_values(["model", "dimension"]).reset_index(drop=True)


def annotate_line_end(ax, x, y, text: str) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=(7, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
    )


def plot_metric_vs_dimension(
    df: pd.DataFrame,
    *,
    y_col: str,
    y_label: str,
    out_name: str,
    log_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    for model in ("FC-HVI", "DeepSets-HVI"):
        group = df[df["model"] == model].sort_values("dimension")
        ax.plot(group["dimension"], group[y_col], marker="o")
        annotate_line_end(
            ax,
            group["dimension"].iloc[-1],
            group[y_col].iloc[-1],
            model,
        )

    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel(y_label)
    ax.set_xticks([10, 20, 50])
    if log_y:
        ax.set_yscale("log")
    ax.margins(x=0.18)
    save_figure(fig, PLOT_OUT / out_name)
