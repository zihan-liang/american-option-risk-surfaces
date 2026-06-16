"""Stage 11: integrated modular workflow evaluation with PSOR audit."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from common import ensure_dir, load_config, write_csv, write_json
from data.dataset_builder import build_surface_samples
from experiment_utils import solver_config_from_dict
from workflow import IntegratedWorkflow, integrated_metrics, safe_speedup


def run(
    config_path: str,
    *,
    output_root: str | Path = "results/experiment_13_integrated_application_workflow",
    figure_root: str | Path = "results/figures/stage11_integrated_workflow",
    overrides: dict | None = None,
) -> dict:
    cfg = load_config(config_path)
    if overrides:
        cfg = {**cfg, **overrides}
    out_dir = ensure_dir(output_root)
    fig_dir = ensure_dir(figure_root)
    fresh_dir = ensure_dir(out_dir / "fresh_cases")
    write_json(out_dir / "integrated_config_manifest.json", cfg)

    workflow = IntegratedWorkflow.load(
        cfg["price_model"],
        cfg["boundary_model"],
        cfg["delta_model"],
        device=cfg.get("device", "auto"),
    )
    case_paths = _dry_run_case_paths(cfg)
    psor_timings: dict[str, float] = {}
    if not case_paths:
        case_paths, psor_timings = _generate_fresh_cases(cfg, fresh_dir)

    case_rows: list[dict] = []
    by_regime_rows: list[dict] = []
    violation_rows: list[dict] = []
    timing_rows: list[dict] = []
    for case_path in case_paths:
        case = _load_case(case_path)
        features = case["features"].astype(np.float32)
        labels = case["labels"].astype(np.float32)
        payoff = labels[:, 2]
        case_id = _case_id(case, case_path)
        option_type = _option_type(features)
        psor_seconds = float(case.get("psor_seconds", np.array([psor_timings.get(case_id, float("nan"))]))[0])
        start = time.perf_counter()
        prediction = workflow.predict(features, payoff, batch_size=int(cfg.get("batch_size", 8192)))
        surrogate_seconds = time.perf_counter() - start
        metrics = integrated_metrics(
            prediction=prediction,
            features=features,
            labels=labels,
            boundary=case["boundary"].astype(float),
            strict_mask=case["strict_greek_mask"].astype(bool),
        )
        common = {
            "case_id": case_id,
            "option_type": option_type,
            "samples": int(len(features)),
        }
        row = {**common, **metrics}
        case_rows.append(row)
        by_regime_rows.append(row)
        violation_rows.append(
            {
                **common,
                "max_price_obstacle_violation": metrics["max_price_obstacle_violation"],
                "max_delta_bound_violation": metrics["max_delta_bound_violation"],
                "boundary_samples": metrics["boundary_samples"],
                "strict_delta_samples": metrics["strict_delta_samples"],
            }
        )
        timing_rows.append(
            {
                **common,
                "psor_seconds": psor_seconds,
                "surrogate_seconds": surrogate_seconds,
                "speedup": safe_speedup(psor_seconds, surrogate_seconds),
            }
        )

    write_csv(out_dir / "integrated_case_metrics.csv", case_rows)
    write_csv(out_dir / "integrated_by_regime.csv", by_regime_rows)
    write_csv(out_dir / "integrated_violation_table.csv", violation_rows)
    write_csv(out_dir / "integrated_timing.csv", timing_rows)
    _save_figures(case_rows, timing_rows, fig_dir)
    summary = _summary(case_rows, timing_rows, cfg)
    write_json(out_dir / "integrated_summary.json", summary)
    return summary


def _dry_run_case_paths(cfg: dict) -> list[Path]:
    return [Path(path) for path in cfg.get("dry_run_cases", [])]


def _generate_fresh_cases(cfg: dict, fresh_dir: Path) -> tuple[list[Path], dict[str, float]]:
    split_rows = _read_csv(Path(cfg["split_table"]))
    selected = _select_cases(split_rows, int(cfg.get("cases_per_option_type", 3)))
    paths: list[Path] = []
    timings: dict[str, float] = {}
    for row in selected:
        solver_cfg = solver_config_from_dict(
            {
                "option_type": row["option_type"],
                "k": float(cfg.get("k", 1.0)),
                "t": float(row["T"]),
                "r": float(row["r"]),
                "q": float(row["q"]),
                "sigma": float(row["sigma"]),
                "s_max": float(cfg.get("s_max", 4.0)),
                "m": int(cfg.get("m", 120)),
                "n": int(cfg.get("n", 120)),
                "omega": float(cfg.get("omega", 1.25)),
                "tol": float(cfg.get("tol", 1e-9)),
                "max_iter": int(cfg.get("max_iter", 20000)),
                "american": True,
            }
        )
        start = time.perf_counter()
        sample = build_surface_samples(
            solver_cfg,
            regime_key_value=row["case_id"],
            split_name="test",
            diagnostic_status="fresh_psor_audit",
            source_stage="stage11",
        )
        psor_seconds = time.perf_counter() - start
        sample["psor_seconds"] = np.array([psor_seconds], dtype=float)
        sample["option_type"] = np.array([row["option_type"].encode("utf-8")])
        path = fresh_dir / f"{row['case_id']}.npz"
        np.savez_compressed(path, **sample)
        paths.append(path)
        timings[row["case_id"]] = psor_seconds
    return paths, timings


def _select_cases(rows: list[dict[str, str]], per_type: int) -> list[dict[str, str]]:
    test_rows = [row for row in rows if row.get("split") == "test"]
    puts = sorted([row for row in test_rows if row.get("option_type") == "put"], key=lambda row: row["case_id"])[:per_type]
    calls = sorted([row for row in test_rows if row.get("regime_class") == "dividend_call"], key=lambda row: row["case_id"])[:per_type]
    return puts + calls


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_case(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _case_id(case: dict[str, np.ndarray], path: Path) -> str:
    if "regime_key" in case and len(case["regime_key"]):
        return str(case["regime_key"].astype(str)[0])
    return path.stem


def _option_type(features: np.ndarray) -> str:
    values = np.where(features[:, 6] > 0.5, "call", "put")
    unique = sorted(set(values.tolist()))
    return unique[0] if len(unique) == 1 else "mixed"


def _summary(case_rows: list[dict], timing_rows: list[dict], cfg: dict) -> dict:
    price_rmse = [row["price_rmse"] for row in case_rows]
    boundary_mae = [row["boundary_mae"] for row in case_rows if np.isfinite(row["boundary_mae"])]
    delta_rmse = [row["strict_delta_rmse"] for row in case_rows if np.isfinite(row["strict_delta_rmse"])]
    speedups = [row["speedup"] for row in timing_rows if np.isfinite(row["speedup"])]
    return {
        "decision": "scenario_screening_with_psor_audit",
        "cases": len(case_rows),
        "mean_price_rmse": float(np.mean(price_rmse)) if price_rmse else float("nan"),
        "mean_boundary_mae": float(np.mean(boundary_mae)) if boundary_mae else float("nan"),
        "mean_strict_delta_rmse": float(np.mean(delta_rmse)) if delta_rmse else float("nan"),
        "max_price_obstacle_violation": float(max(row["max_price_obstacle_violation"] for row in case_rows)) if case_rows else float("nan"),
        "max_delta_bound_violation": float(max(row["max_delta_bound_violation"] for row in case_rows)) if case_rows else float("nan"),
        "mean_speedup": float(np.mean(speedups)) if speedups else float("nan"),
        "claims_psor_replacement": False,
        "allowed_claim": "modular scenario screening with PSOR audit",
        "uses_separate_heads": True,
        "price_model": str(cfg["price_model"]),
        "boundary_model": str(cfg["boundary_model"]),
        "delta_model": str(cfg["delta_model"]),
    }


def _save_figures(case_rows: list[dict], timing_rows: list[dict], figure_root: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(figure_root)
    if case_rows:
        labels = [row["case_id"] for row in case_rows]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, [row["price_rmse"] for row in case_rows])
        ax.set_ylabel("price RMSE")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(figure_root / "integrated_price_rmse.png", dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, [row["strict_delta_rmse"] for row in case_rows])
        ax.set_ylabel("strict Delta RMSE")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(figure_root / "integrated_delta_rmse.png", dpi=160)
        plt.close(fig)
    if timing_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([row["case_id"] for row in timing_rows], [row["speedup"] for row in timing_rows])
        ax.set_ylabel("PSOR / workflow seconds")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(figure_root / "integrated_speedup.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/integrated_workflow.yaml")
    args = parser.parse_args()
    print(run(args.config))
