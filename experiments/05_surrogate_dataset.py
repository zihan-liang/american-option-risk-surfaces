"""Stage 6: construct a regime-split surrogate dataset from clean PSOR labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from common import ensure_dir, load_config, write_csv, write_json
from data.dataset_builder import build_surface_samples
from data.splits import three_way_regime_split
from experiment_utils import solver_config_from_dict


FEATURE_NAMES = ["log_moneyness", "tau_over_T", "r", "q", "sigma", "T", "is_call"]
LABEL_NAMES = ["price_over_K", "premium_over_K", "payoff_over_K", "delta", "K_gamma"]
METADATA_NAMES = [
    "boundary",
    "exercise_indicator",
    "strict_greek_mask",
    "near_kink",
    "near_boundary",
    "near_zero_spot",
    "regime_key",
    "split",
    "diagnostic_status",
    "source_stage",
    "solver_config_hash",
]


def run(
    config_path: str,
    *,
    output_root: str | Path = "results/experiment_05_surrogate_dataset",
    dataset_path: str | Path = "data/processed/surrogate_dataset.npz",
    figure_root: str | Path = "results/figures/stage06_surrogate_dataset",
    overrides: dict | None = None,
) -> dict:
    raw = load_config(config_path)
    if overrides:
        raw = {**raw, **overrides}
    out_dir = ensure_dir(output_root)
    ensure_dir(Path(dataset_path).parent)
    fig_dir = ensure_dir(figure_root)

    source_root = Path(raw["source_experiment_root"])
    risk_rows = _read_csv(source_root / "risk_metrics.csv")
    flag_rows = _read_csv(source_root / "regime_flags.csv")
    boundary_summary_path = source_root / "boundary_summary.csv"
    if not boundary_summary_path.exists():
        raise FileNotFoundError(boundary_summary_path)

    clean_regimes, excluded_regimes = select_dataset_regimes(risk_rows, flag_rows)
    split_map = three_way_regime_split(
        clean_regimes,
        train_fraction=float(raw.get("train_fraction", 0.70)),
        validation_fraction=float(raw.get("validation_fraction", 0.15)),
        seed=int(raw.get("seed", 7)),
    )

    split_rows: list[dict[str, Any]] = []
    arrays: list[dict[str, np.ndarray]] = []
    for split_name, regimes in split_map.items():
        for row in regimes:
            cfg = _config_from_regime(raw, row)
            sample = build_surface_samples(
                cfg,
                regime_key_value=row["case_id"],
                split_name=split_name,
                diagnostic_status="clean",
                source_stage="stage05",
            )
            arrays.append(sample)
            split_rows.append(
                {
                    "split": split_name,
                    "case_id": row["case_id"],
                    "regime_class": row["regime_class"],
                    "option_type": row["option_type"],
                    "T": row["T"],
                    "sigma": row["sigma"],
                    "r": row["r"],
                    "q": row["q"],
                    "samples": len(sample["features"]),
                }
            )

    if not arrays:
        raise ValueError("no clean regimes selected for dataset construction")

    dataset = _concatenate_samples(arrays)
    np.savez_compressed(dataset_path, **dataset)

    write_csv(out_dir / "split_table.csv", split_rows)
    write_csv(out_dir / "excluded_regimes.csv", excluded_regimes)
    schema = {
        "feature_names": FEATURE_NAMES,
        "label_names": LABEL_NAMES,
        "metadata_names": METADATA_NAMES,
        "primary_split": "regime",
    }
    write_json(out_dir / "dataset_schema.json", schema)
    write_csv(out_dir / "label_summary.csv", _label_summary(dataset["labels"]))
    write_csv(out_dir / "mask_summary.csv", _mask_summary(dataset))
    manifest = _manifest(raw, source_root, dataset_path, dataset, split_rows, clean_regimes, excluded_regimes)
    write_json(out_dir / "dataset_manifest.json", manifest)
    write_json(out_dir / "dataset_audit.json", _audit(clean_regimes, excluded_regimes, split_rows))
    _save_label_figures(dataset["labels"], fig_dir)
    return manifest


def select_dataset_regimes(
    risk_rows: list[dict[str, Any]],
    flag_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    flags_by_case = {row["case_id"]: row for row in flag_rows}
    clean: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in risk_rows:
        flags = flags_by_case.get(row["case_id"], {})
        reasons: list[str] = []
        if row.get("regime_class") == "no_dividend_call_control":
            reasons.append("control_regime")
        if row.get("regime_class") not in {"put", "dividend_call", "no_dividend_call_control"}:
            reasons.append("unsupported_regime")
        if _as_bool(flags.get("diagnostic_failure", False)):
            reasons.append("diagnostic_failure")
        if _as_bool(flags.get("delicate_regime", False)):
            reasons.append("delicate_regime")
        if reasons:
            excluded.append({**row, "exclusion_reason": ";".join(reasons)})
        else:
            clean.append(row)
    return clean, excluded


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _config_from_regime(raw: dict, row: dict[str, Any]):
    return solver_config_from_dict(
        {
            "option_type": row["option_type"],
            "k": raw["k"],
            "t": float(row["T"]),
            "r": float(row["r"]),
            "q": float(row["q"]),
            "sigma": float(row["sigma"]),
            "s_max": raw["s_max"],
            "m": raw["m"],
            "n": raw["n"],
            "omega": raw["omega"],
            "tol": raw["tol"],
            "max_iter": raw["max_iter"],
            "american": True,
        }
    )


def _concatenate_samples(arrays: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = [
        "features",
        "labels",
        "boundary",
        "exercise_indicator",
        "strict_greek_mask",
        "near_kink",
        "near_boundary",
        "near_zero_spot",
        "regime_key",
        "split",
        "diagnostic_status",
        "source_stage",
        "solver_config_hash",
    ]
    return {key: np.concatenate([sample[key] for sample in arrays], axis=0) for key in keys}


def _label_summary(labels: np.ndarray) -> list[dict[str, float | str]]:
    rows = []
    for idx, name in enumerate(LABEL_NAMES):
        values = labels[:, idx]
        rows.append(
            {
                "label": name,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        )
    return rows


def _mask_summary(dataset: dict[str, np.ndarray]) -> list[dict[str, float | int | str]]:
    total = int(len(dataset["features"]))
    rows = []
    for key in ["exercise_indicator", "strict_greek_mask", "near_kink", "near_boundary", "near_zero_spot"]:
        active = int(np.count_nonzero(dataset[key]))
        rows.append({"mask": key, "samples": active, "fraction": float(active / total)})
    return rows


def _manifest(
    raw: dict,
    source_root: Path,
    dataset_path: str | Path,
    dataset: dict[str, np.ndarray],
    split_rows: list[dict[str, Any]],
    clean_regimes: list[dict[str, Any]],
    excluded_regimes: list[dict[str, Any]],
) -> dict[str, Any]:
    split_counts = {
        split: sum(1 for row in split_rows if row["split"] == split)
        for split in ["train", "validation", "test"]
    }
    sample_split_values = dataset["split"].astype(str)
    sample_counts = {
        split: int(np.count_nonzero(sample_split_values == split))
        for split in ["train", "validation", "test"]
    }
    return {
        "source_stage": "stage05_application_risk_surfaces",
        "source_experiment_root": str(source_root),
        "dataset_path": str(dataset_path),
        "samples": int(len(dataset["features"])),
        "features": int(dataset["features"].shape[1]),
        "labels": int(dataset["labels"].shape[1]),
        "clean_regimes": len(clean_regimes),
        "excluded_regimes": len(excluded_regimes),
        "split_regime_counts": split_counts,
        "split_sample_counts": sample_counts,
        "feature_names": FEATURE_NAMES,
        "label_names": LABEL_NAMES,
        "solver_grid": {"m": raw["m"], "n": raw["n"], "s_max": raw["s_max"]},
        "seed": raw.get("seed", 7),
        "source_files": [
            str(source_root / "risk_metrics.csv"),
            str(source_root / "regime_flags.csv"),
            str(source_root / "boundary_summary.csv"),
        ],
    }


def _audit(clean_regimes: list[dict[str, Any]], excluded_regimes: list[dict[str, Any]], split_rows: list[dict[str, Any]]) -> dict:
    split_by_case = {row["case_id"]: row["split"] for row in split_rows}
    return {
        "clean_regime_keys": [row["case_id"] for row in clean_regimes],
        "excluded_regime_keys": [row["case_id"] for row in excluded_regimes],
        "excluded_reasons": {row["case_id"]: row["exclusion_reason"] for row in excluded_regimes},
        "split_by_regime": split_by_case,
        "leakage_check": len(split_by_case) == len(split_rows),
    }


def _save_label_figures(labels: np.ndarray, figure_root: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(figure_root)
    for idx, name in enumerate(LABEL_NAMES):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(labels[:, idx], bins=60)
        ax.set_title(name)
        ax.set_xlabel(name)
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(figure_root / f"{name}_hist.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/surrogate_dataset.yaml")
    args = parser.parse_args()
    print(run(args.config))
