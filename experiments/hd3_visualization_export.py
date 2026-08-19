"""CLI entry point for deterministic post-hoc HD3 visualization export.

All heavy lifting lives in focused modules under ``src/hd_pipeline``. This
entry point deliberately exposes the tested helper interfaces used by the
HD3 visualization tests while keeping orchestration separate from numerics
and plotting.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from hd_pipeline import hd3_visualization_core as core
from hd_pipeline import hd3_visualization_data as data
from hd_pipeline import hd3_visualization_plots as plots

TRAINING_IMPLEMENTATION_SHA = core.TRAINING_IMPLEMENTATION_SHA
CANONICAL_EVIDENCE_SHA = core.CANONICAL_EVIDENCE_SHA
CheckpointExpectation = core.CheckpointExpectation
OracleAsset = core.OracleAsset
CANONICAL_DIMENSIONS = core.CANONICAL_DIMENSIONS
STRESS_DIMENSIONS = core.STRESS_DIMENSIONS
ALL_DIMENSIONS = core.ALL_DIMENSIONS
MODEL_IDS = core.MODEL_IDS
CRR_DEPTHS = core.CRR_DEPTHS
ROOT = core.ROOT
VISUALIZATION_DATA_DIR = core.VISUALIZATION_DATA_DIR
FIGURE_DIR = core.FIGURE_DIR
DIMENSION_SUMMARY_PATH = core.DIMENSION_SUMMARY_PATH
STRUCTURAL_DIAGNOSTICS_PATH = core.STRUCTURAL_DIAGNOSTICS_PATH

validate_checkpoint_metadata = core.validate_checkpoint_metadata
load_completed_model_for_visualization = core.load_completed_model_for_visualization
state_dict_digest = core.state_dict_digest
infer_prices = core.infer_prices
sha256_path = core.sha256_path
write_rows_csv = core.write_rows_csv
write_json = core.write_json
select_crr_depth = core.select_crr_depth
diagonal_states = core.diagonal_states
effective_contract = core.effective_contract
load_oracle_asset = core.load_oracle_asset
oracle_price = core.oracle_price
run_class_for_dimension = core.run_class_for_dimension
checkpoint_path = core.checkpoint_path
oracle_path = core.oracle_path
load_frozen_evidence = core.load_frozen_evidence
checkpoint_expectation = core.checkpoint_expectation
export_source_inventory = data.export_source_inventory
export_reference_agreement = data.export_reference_agreement
load_all_oracles = data.load_all_oracles
load_all_models = data.load_all_models
export_diagonal_slices = data.export_diagonal_slices
export_error_surfaces = data.export_error_surfaces
export_exercise_boundaries = data.export_exercise_boundaries

build_figure_metrics = plots.build_figure_metrics
generate_all_figures = plots.generate_all_figures
_read_csv = plots._read_csv
_truthy = plots._truthy
_json_default = core._json_default

SLICE_MATURITIES = (1.0, 0.5, 0.1)
SLICE_SPOT_COUNT = 401
SURFACE_SPOT_COUNT = 201
SURFACE_TAU_COUNT = 101
BOUNDARY_TAU_COUNT = 128
BOUNDARY_SPOT_COUNT = 512
INFERENCE_BATCH_SIZE = 4096


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Post-hoc HD3 visualization export; inference only, no retraining."
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--all", action="store_true", help="export derived data and figures")
    modes.add_argument("--reference-only", action="store_true", help="run CN/PSOR vs CRR audit only")
    modes.add_argument("--data-only", action="store_true", help="export derived visualization CSV/JSON only")
    modes.add_argument("--figures-only", action="store_true", help="render figures from existing derived CSVs")
    return parser


def current_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _path_for_manifest(path: Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def write_visualization_manifest(
    *,
    output_path: Path,
    mode: str,
    visualization_git_sha: str,
    config_hash: str,
    objective_hash: str,
    selected_crr: dict[int, dict[str, Any]],
    inference_device: str,
    output_files: list[Path],
) -> Path:
    payload = {
        "status": "completed",
        "mode": str(mode),
        "visualization_code_git_sha": str(visualization_git_sha),
        "training_implementation_sha": TRAINING_IMPLEMENTATION_SHA,
        "canonical_evidence_commit": CANONICAL_EVIDENCE_SHA,
        "config_hash": str(config_hash),
        "resolved_objective_hash": str(objective_hash),
        "canonical_dimensions": list(CANONICAL_DIMENSIONS),
        "stress_dimensions": list(STRESS_DIMENSIONS),
        "selected_crr_depths": {str(k): dict(v) for k, v in sorted(selected_crr.items())},
        "reference_solver": {
            "primary": "CN/PSOR 512x512 oracle",
            "cross_check": "Cox-Ross-Rubinstein American put",
            "candidate_depths": list(CRR_DEPTHS),
            "cnpsor_rmse_threshold": 1.0e-4,
            "previous_depth_rmse_threshold": 5.0e-5,
        },
        "inference_device": str(inference_device),
        "grid_definitions": {
            "diagonal_slices": {
                "S_min": 0.5,
                "S_max": 1.5,
                "S_count": int(SLICE_SPOT_COUNT),
                "maturities": list(SLICE_MATURITIES),
            },
            "error_surfaces": {
                "S_min": 0.5,
                "S_max": 1.5,
                "S_count": int(SURFACE_SPOT_COUNT),
                "tau_min": 0.0025,
                "tau_max": 1.0,
                "tau_count": int(SURFACE_TAU_COUNT),
            },
            "exercise_boundaries": {
                "S_min": 0.5,
                "S_max": 1.2,
                "S_count": int(BOUNDARY_SPOT_COUNT),
                "tau_min": 0.0025,
                "tau_max": 1.0,
                "tau_count": int(BOUNDARY_TAU_COUNT),
                "exercise_gap_tolerance": 1.0e-5,
            },
        },
        "output_files": [_path_for_manifest(path) for path in output_files],
        "optimizer_steps": 0,
        "retraining": False,
        "blind_holdout_access": False,
    }
    return write_json(output_path, payload)


def _required_derived_csvs(data_dir: Path) -> list[Path]:
    data_dir = Path(data_dir)
    return [
        data_dir / "reference_solver_agreement.csv",
        data_dir / "figure_metrics.csv",
        data_dir / "diagonal_slices.csv",
        *(data_dir / f"error_surface_{d}d.csv" for d in CANONICAL_DIMENSIONS),
        data_dir / "exercise_boundaries.csv",
    ]


def require_paths(paths: list[Path]) -> None:
    for path in paths:
        if not Path(path).is_file():
            try:
                label = Path(path).relative_to(ROOT).as_posix()
            except Exception:
                label = str(path)
            raise FileNotFoundError(f"missing_required_asset:{label}")


def _runtime_paths(include_checkpoints: bool) -> list[Path]:
    paths = [oracle_path(d) for d in ALL_DIMENSIONS]
    if include_checkpoints:
        paths.extend(checkpoint_path(d, model_id) for d in ALL_DIMENSIONS for model_id in MODEL_IDS)
    return paths


def _selected_crr_from_csv(path: Path) -> dict[int, dict[str, Any]]:
    rows = _read_csv(path)
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if _truthy(row.get("selected_depth", "false")):
            d = int(row["dimension"])
            result[d] = {
                "depth": int(row["depth"]),
                "passed": _truthy(row["reference_agreement_passed"]),
            }
    return result


def run_export(mode: str) -> dict[str, Any]:
    if mode not in {"all", "reference_only", "data_only", "figures_only"}:
        raise ValueError(f"unknown visualization mode: {mode}")
    data_dir = VISUALIZATION_DATA_DIR
    figure_dir = FIGURE_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    cfg, _objective, cfg_hash, objective_hash = load_frozen_evidence()
    selected_crr: dict[int, dict[str, Any]] = {}
    outputs: list[Path] = []
    inference_device = "not_used"

    if mode == "figures_only":
        require_paths(_required_derived_csvs(data_dir))
        reference_path = data_dir / "reference_solver_agreement.csv"
        selected_crr = _selected_crr_from_csv(reference_path)
        outputs.extend(generate_all_figures(data_dir=data_dir, figure_dir=figure_dir))
    else:
        include_checkpoints = mode in {"all", "data_only"}
        require_paths(_runtime_paths(include_checkpoints=include_checkpoints))
        oracles = load_all_oracles()
        inventory_path = data_dir / "source_inventory.json"
        export_source_inventory(
            output_path=inventory_path,
            config_hash=cfg_hash,
            objective_hash=objective_hash,
            include_checkpoints=include_checkpoints,
        )
        outputs.append(inventory_path)
        reference_path = data_dir / "reference_solver_agreement.csv"
        selected_crr = export_reference_agreement(
            cfg=cfg,
            oracles=oracles,
            output_path=reference_path,
            depths=tuple(CRR_DEPTHS),
            spot_count=int(SLICE_SPOT_COUNT),
        )
        outputs.append(reference_path)

        if include_checkpoints:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            inference_device = str(device)
            models, _checkpoints = load_all_models(
                cfg=cfg, config_hash=cfg_hash, objective_hash=objective_hash, device=device
            )
            slices_path = data_dir / "diagonal_slices.csv"
            export_diagonal_slices(
                cfg=cfg,
                oracles=oracles,
                models=models,
                selected_crr=selected_crr,
                output_path=slices_path,
                maturities=tuple(SLICE_MATURITIES),
                spot_count=int(SLICE_SPOT_COUNT),
                batch_size=int(INFERENCE_BATCH_SIZE),
            )
            outputs.append(slices_path)
            outputs.extend(
                export_error_surfaces(
                    cfg=cfg,
                    oracles=oracles,
                    models=models,
                    output_dir=data_dir,
                    spot_count=int(SURFACE_SPOT_COUNT),
                    tau_count=int(SURFACE_TAU_COUNT),
                    batch_size=int(INFERENCE_BATCH_SIZE),
                )
            )
            boundary_path = data_dir / "exercise_boundaries.csv"
            export_exercise_boundaries(
                cfg=cfg,
                oracles=oracles,
                models=models,
                output_path=boundary_path,
                tau_count=int(BOUNDARY_TAU_COUNT),
                spot_count=int(BOUNDARY_SPOT_COUNT),
                batch_size=int(INFERENCE_BATCH_SIZE),
            )
            outputs.append(boundary_path)
            metrics_path = data_dir / "figure_metrics.csv"
            build_figure_metrics(
                dimension_summary_path=DIMENSION_SUMMARY_PATH,
                structural_path=STRUCTURAL_DIAGNOSTICS_PATH,
                output_path=metrics_path,
            )
            outputs.append(metrics_path)
            if mode == "all":
                outputs.extend(generate_all_figures(data_dir=data_dir, figure_dir=figure_dir))

    manifest_path = data_dir / "visualization_manifest.json"
    write_visualization_manifest(
        output_path=manifest_path,
        mode=mode,
        visualization_git_sha=current_git_sha(),
        config_hash=cfg_hash,
        objective_hash=objective_hash,
        selected_crr=selected_crr,
        inference_device=inference_device,
        output_files=[*outputs, manifest_path],
    )
    outputs.append(manifest_path)
    return {
        "status": "completed",
        "mode": mode,
        "optimizer_steps": 0,
        "retraining": False,
        "blind_holdout_access": False,
        "selected_crr_depths": selected_crr,
        "output_files": [_path_for_manifest(path) for path in outputs],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.all:
        mode = "all"
    elif args.reference_only:
        mode = "reference_only"
    elif args.data_only:
        mode = "data_only"
    else:
        mode = "figures_only"
    result = run_export(mode)
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
