from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

import hd3_visualization_export as export_module
from hd3_visualization_export import TRAINING_IMPLEMENTATION_SHA, sha256_path, write_rows_csv


class TinyModel(torch.nn.Module):
    def __init__(self, d: int):
        super().__init__(); self.linear = torch.nn.Linear(d + 1, 1)
    def forward(self, x, tau):
        return self.linear(torch.cat([x, tau[:, None]], dim=1)).reshape(-1)


def _linear_oracle(d: int):
    spots = np.array([0.5, 1.0, 1.5]); taus = np.array([0.0025, 0.5, 1.0])
    values = np.maximum(1.0 - spots[None, :], 0.0) + 0.01 * taus[:, None]
    return export_module.OracleAsset(d, 0.15, 0.04, spots, taus, values, np.array([0.9, 0.85, 0.8]), 1e-5, 0.1)


def _interp_for_test(spots, taus, values, g, tau):
    f = RegularGridInterpolator((taus, spots), values, bounds_error=False, fill_value=None)
    return np.asarray(f(np.column_stack([tau, g])), dtype=float)


def _patch_numerics(monkeypatch):
    monkeypatch.setattr(export_module.core, "_interp2", _interp_for_test)
    monkeypatch.setattr(export_module.core, "_effective_geometric_basket_parameters", lambda d, sigma, q, rho: (0.15, 0.04))


def test_small_derived_data_export_writes_declared_schemas(tmp_path: Path, monkeypatch):
    _patch_numerics(monkeypatch)
    cfg = {"contract": {"K": 1.0, "T": 1.0, "r": 0.05, "q": 0.02, "sigma": 0.25, "rho": 0.3}}
    oracles = {d: _linear_oracle(d) for d in export_module.ALL_DIMENSIONS}
    torch.manual_seed(2)
    models = {d: {"FC_HVI": TinyModel(d).eval(), "DEEPSETS_HVI": TinyModel(d).eval()} for d in export_module.ALL_DIMENSIONS}
    reference_path = tmp_path / "reference_solver_agreement.csv"
    selected = export_module.export_reference_agreement(cfg=cfg, oracles=oracles, output_path=reference_path, depths=(10, 20), spot_count=7)
    assert set(selected) == {10, 20, 50, 100}
    assert len(list(csv.DictReader(reference_path.open()))) == 8
    selected = {d: {"depth": 10, "passed": False} for d in export_module.ALL_DIMENSIONS}
    slices_path = tmp_path / "diagonal_slices.csv"
    export_module.export_diagonal_slices(cfg=cfg, oracles=oracles, models=models, selected_crr=selected, output_path=slices_path, maturities=(1.0, 0.5), spot_count=7, batch_size=4)
    rows = list(csv.DictReader(slices_path.open()))
    assert len(rows) == 56
    assert {r["run_class"] for r in rows if r["dimension"] == "100"} == {"stress"}
    surface_paths = export_module.export_error_surfaces(cfg=cfg, oracles=oracles, models=models, output_dir=tmp_path, spot_count=5, tau_count=4, batch_size=8)
    assert {p.name for p in surface_paths} == {"error_surface_10d.csv", "error_surface_20d.csv", "error_surface_50d.csv"}
    boundary_path = tmp_path / "exercise_boundaries.csv"
    export_module.export_exercise_boundaries(cfg=cfg, oracles=oracles, models=models, output_path=boundary_path, tau_count=4, spot_count=9, batch_size=8)
    assert len(list(csv.DictReader(boundary_path.open()))) == 12


def test_cpu_synthetic_all_export_is_read_only_and_complete(tmp_path: Path, monkeypatch):
    root = tmp_path / "repo"
    runtime = root / "results" / "hd3_classical_vs_structured_high_dimensional_pinn" / "runtime"
    oracle_dir = runtime / "oracles"; oracle_dir.mkdir(parents=True)
    data_dir = root / "results" / "hd3_visualization"; figure_dir = root / "figures" / "hd3"
    summary_path = root / "dimension_summary.csv"; structural_path = root / "structural_diagnostics.csv"
    cfg = {"seed": 101, "fc_width": 128, "fc_hidden_layers": 6, "contract": {"K": 1.0, "T": 1.0, "r": 0.05, "q": 0.02, "sigma": 0.25, "rho": 0.3}}
    config_hash, objective_hash = "cfg-hash", "obj-hash"
    monkeypatch.setattr(export_module, "ROOT", root); monkeypatch.setattr(export_module.core, "ROOT", root); monkeypatch.setattr(export_module.data, "ROOT", root)
    monkeypatch.setattr(export_module.core, "HD3_RUNTIME_DIR", runtime); monkeypatch.setattr(export_module.core, "HD3_ORACLE_DIR", oracle_dir)
    monkeypatch.setattr(export_module, "VISUALIZATION_DATA_DIR", data_dir); monkeypatch.setattr(export_module, "FIGURE_DIR", figure_dir)
    monkeypatch.setattr(export_module, "DIMENSION_SUMMARY_PATH", summary_path); monkeypatch.setattr(export_module, "STRUCTURAL_DIAGNOSTICS_PATH", structural_path)
    monkeypatch.setattr(export_module, "CRR_DEPTHS", (5, 10)); monkeypatch.setattr(export_module, "SLICE_MATURITIES", (1.0, 0.5)); monkeypatch.setattr(export_module, "SLICE_SPOT_COUNT", 5)
    monkeypatch.setattr(export_module, "SURFACE_SPOT_COUNT", 4); monkeypatch.setattr(export_module, "SURFACE_TAU_COUNT", 3); monkeypatch.setattr(export_module, "BOUNDARY_TAU_COUNT", 3); monkeypatch.setattr(export_module, "BOUNDARY_SPOT_COUNT", 7); monkeypatch.setattr(export_module, "INFERENCE_BATCH_SIZE", 8)
    monkeypatch.setattr(export_module, "current_git_sha", lambda: "visualization-sha")
    monkeypatch.setattr(export_module, "load_frozen_evidence", lambda: (cfg, {"candidate": "HVI-warm-low"}, config_hash, objective_hash))
    monkeypatch.setattr(export_module.core, "_initialize_model", lambda model_id, d, cfg, device: TinyModel(d).to(device)); _patch_numerics(monkeypatch)
    checkpoint_digests = {}; torch.manual_seed(41)
    for d in export_module.ALL_DIMENSIONS:
        run_class = "stress" if d == 100 else "canonical"; steps = 4000 if d == 100 else 6000
        for model_id in export_module.MODEL_IDS:
            state = TinyModel(d).state_dict(); path = runtime / f"{run_class}_d{d}_{model_id}" / "latest_checkpoint.pt"; path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"implementation_sha": TRAINING_IMPLEMENTATION_SHA, "config_hash": config_hash, "resolved_objective_hash": objective_hash, "job_id": f"{run_class}_d{d}_{model_id}", "dimension": d, "model_id": model_id, "run_class": run_class, "completed_step": steps, "target_steps": steps, "model_state": state}, path)
            checkpoint_digests[path] = sha256_path(path)
    spots = np.array([0.5, 1.0, 1.5]); taus = np.array([0.0025, 0.5, 1.0]); values = np.maximum(1.0 - spots[None, :], 0.0) + 0.01 * taus[:, None]
    for d in export_module.ALL_DIMENSIONS:
        np.savez_compressed(oracle_dir / f"d{d}.npz", d=d, sigma_g=0.15, q_g=0.04, spots=spots, taus=taus, values=values, boundary=np.array([0.9, 0.85, 0.8]), refinement_rmse=1e-5, seconds=0.1)
    scaling, structural = [], []
    for d in export_module.ALL_DIMENSIONS:
        run_class = "stress" if d == 100 else "canonical"
        for j, model_id in enumerate(export_module.MODEL_IDS):
            scaling.append({"dimension": d, "model_id": model_id, "run_class": run_class, "normalized_price_rmse": .001+j*.0001, "delta_rmse": .002+j*.0001, "trusted_pde_p95": .05+j*.01, "training_seconds": 10+d})
            structural.append({"dimension": d, "model_id": model_id, "run_class": run_class, "permutation_consistency_rmse": 1e-3 if j == 0 else 1e-8, "transverse_invariance_rmse": 1e-4+j*1e-5})
    write_rows_csv(summary_path, scaling); write_rows_csv(structural_path, structural)
    result = export_module.run_export("all")
    assert result["optimizer_steps"] == 0 and result["retraining"] is False and result["blind_holdout_access"] is False
    declared = {"source_inventory.json", "reference_solver_agreement.csv", "diagonal_slices.csv", "error_surface_10d.csv", "error_surface_20d.csv", "error_surface_50d.csv", "exercise_boundaries.csv", "figure_metrics.csv", "visualization_manifest.json"}
    assert declared <= {p.name for p in data_dir.iterdir()}; assert len(list(figure_dir.glob("*.png"))) == 6
    manifest = json.loads((data_dir / "visualization_manifest.json").read_text()); assert manifest["visualization_code_git_sha"] == "visualization-sha" and manifest["optimizer_steps"] == 0
    assert {path: sha256_path(path) for path in checkpoint_digests} == checkpoint_digests
