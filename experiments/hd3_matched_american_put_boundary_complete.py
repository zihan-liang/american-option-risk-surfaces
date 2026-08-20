"""Final boundary-complete matched-contract American-put HD3 diagnostic."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXPERIMENTS = ROOT / "experiments"
for _p in (SRC, EXPERIMENTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import hd_pipeline.runtime as _runtime
import hd3_matched_american_put_comparison as _old
from hd1_windows_high_dimensional_pinn_48h import (
    Oracle,
    environment_manifest,
    f1_score,
    git_commit,
    interp2,
    peak_cpu_memory_bytes,
)
from hd3_classical_vs_structured_high_dimensional_pinn import _parameter_count
from physics.hd3_high_dimensional_sampling import (
    grouped_transverse_states,
    permutation_consistency_rmse,
    transverse_invariance_rmse,
)
from physics.high_dimensional_black_scholes import black_scholes_vi_residual, geometric_put_payoff
from hd3_matched_boundary_complete_core import (
    BOUNDARY_METHOD,
    COMPONENT_DELTA_DEFINITION,
    GAMMA_CONTRACTION_DEFINITION,
    REDUCED_DELTA_DEFINITION,
    REDUCED_GAMMA_DEFINITION,
    CONFIG_PATH,
    OLD_RESULT_DIR,
    ORACLE_DIR,
    RESULT_DIR,
    RUNTIME_DIR,
    VIS_DIR,
    aggregate_bounds,
    aggregate_spot_bounds,
    assert_oracle_coverage,
    binomial_check_spots,
    binomial_final_steps,
    boundary_axes,
    build_boundary_complete_oracle,
    compute_boundary_metrics,
    config_hash,
    crr_american_put,
    diagonal_value_and_greeks,
    dimension_config,
    load_config,
    load_oracle,
    model_surface_arrays,
    oracle_labels_complete,
    oracle_refinement_axes,
    oracle_surface_arrays,
    parameter_mapping,
    physics_unit_to_states,
    sample_states,
    save_oracle,
    tau_bounds,
    training_domain_metadata,
    transition_boundary,
)


def safe_peak_cpu_memory_bytes(probe: Callable[[], int] | None = None) -> int | None:
    memory_probe = peak_cpu_memory_bytes if probe is None else probe
    try:
        value = int(memory_probe())
    except OSError:
        return None
    return value if value >= 0 else None


def _checkpoint_path(d: int, model_id: str) -> Path:
    return RUNTIME_DIR / f"canonical_d{int(d)}_{model_id}" / "latest_checkpoint.pt"


def _status_path(d: int, model_id: str) -> Path:
    return RUNTIME_DIR / f"canonical_d{int(d)}_{model_id}" / "status.json"


def _job_id(d: int, model_id: str) -> str:
    return f"canonical_d{int(d)}_{model_id}"


def _install_training_namespace() -> None:
    """Point the frozen old trainer at only the new namespace and safe hooks."""
    _old.CONFIG_PATH = CONFIG_PATH
    _old.RESULT_DIR = RESULT_DIR
    _old.RUNTIME_DIR = RUNTIME_DIR
    _old.ORACLE_DIR = ORACLE_DIR
    _old.VIS_DIR = VIS_DIR
    _old.load_config = load_config
    _old.config_hash = config_hash
    _old.parameter_mapping = parameter_mapping
    _old.dimension_config = dimension_config
    _old.load_oracle = load_oracle
    _old.save_oracle = save_oracle
    _old.oracle_labels = oracle_labels_complete
    _old.evaluate_hd3_job = evaluate_job

    raw_save = _runtime.save_checkpoint_atomic
    raw_load = _runtime.load_checkpoint_verified

    def save_checkpoint_with_final_provenance(path: Path, payload: dict[str, Any]) -> None:
        d = int(payload["dimension"])
        cfg = dimension_config(d, load_config())
        enriched = dict(payload)
        enriched.update(
            {
                "experiment_id": cfg["experiment_id"],
                "training_domain": training_domain_metadata(cfg),
                "target_reduced_contract": cfg["target_reduced_contract"],
                "asset_contract": cfg["contract"],
                "rho": float(cfg["rho"]),
                "parameter_mapping": parameter_mapping(d, load_config()),
                "objective": cfg["objective"],
            }
        )
        raw_save(path, enriched)

    def load_checkpoint_with_final_provenance(
        path: Path, *, implementation_sha: str, config_hash: str, job_id: str
    ) -> dict[str, Any]:
        payload = raw_load(
            path,
            implementation_sha=implementation_sha,
            config_hash=config_hash,
            job_id=job_id,
        )
        cfg = load_config()
        if payload.get("experiment_id") != cfg["experiment_id"]:
            raise RuntimeError("stale_experiment_namespace")
        if payload.get("training_domain") != training_domain_metadata(cfg):
            raise RuntimeError("stale_training_domain")
        return payload

    _old.save_checkpoint_atomic = save_checkpoint_with_final_provenance
    _old.load_checkpoint_verified = load_checkpoint_with_final_provenance


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    tmp.replace(path)


def _predict_diagonal_values(
    model: torch.nn.Module, *, d: int, spots: np.ndarray, taus: np.ndarray,
    chunk_size: int = 1024,
) -> np.ndarray:
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    s, t = np.asarray(spots, float).reshape(-1), np.asarray(taus, float).reshape(-1)
    if s.shape != t.shape:
        raise ValueError("spots and taus must have the same shape")
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(s), int(chunk_size)):
            stop = min(start + int(chunk_size), len(s))
            g = torch.tensor(s[start:stop], device=device, dtype=torch.float32)
            tau = torch.tensor(t[start:stop], device=device, dtype=torch.float32)
            x = torch.log(g).reshape(-1, 1).repeat(1, int(d))
            outputs.append(model(x, tau).detach().cpu().numpy().reshape(-1))
    return np.concatenate(outputs)


def _boundary_series_model(
    model: torch.nn.Module, d: int, cfg: dict[str, Any], tolerance: float,
) -> list[dict[str, Any]]:
    spots, taus = boundary_axes(cfg)
    rows: list[dict[str, Any]] = []
    for tau_value in taus:
        values = _predict_diagonal_values(
            model, d=int(d), spots=spots, taus=np.full_like(spots, float(tau_value))
        )
        result = transition_boundary(
            spots, values - np.maximum(1.0 - spots, 0.0),
            tolerance=float(tolerance),
            min_continuation_nodes=int(cfg["boundary"]["min_continuation_nodes"]),
        )
        rows.append(
            {
                "tau_over_T": float(tau_value),
                "S_star_over_K": result["S_star_over_K"],
                "boundary_found": bool(result["found"]),
                "missing_reason": None if result["found"] else result["reason"],
                "boundary_method": BOUNDARY_METHOD,
                "boundary_tolerance": float(tolerance),
            }
        )
    return rows


def _boundary_series_oracle(
    oracle: Oracle, cfg: dict[str, Any], tolerance: float,
) -> list[dict[str, Any]]:
    spots, taus = boundary_axes(cfg)
    rows: list[dict[str, Any]] = []
    for tau_value in taus:
        t = np.full_like(spots, float(tau_value))
        assert_oracle_coverage(oracle, spots, t)
        values = interp2(oracle.spots, oracle.taus, oracle.values, spots, t)
        result = transition_boundary(
            spots, values - np.maximum(1.0 - spots, 0.0),
            tolerance=float(tolerance),
            min_continuation_nodes=int(cfg["boundary"]["min_continuation_nodes"]),
        )
        rows.append(
            {
                "tau_over_T": float(tau_value),
                "S_star_over_K": result["S_star_over_K"],
                "boundary_found": bool(result["found"]),
                "missing_reason": None if result["found"] else result["reason"],
                "boundary_method": BOUNDARY_METHOD,
                "boundary_tolerance": float(tolerance),
            }
        )
    return rows


def _compare_boundary_rows(
    predicted: list[dict[str, Any]], reference: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(predicted) != len(reference):
        raise ValueError("boundary comparison length mismatch")
    output: list[dict[str, Any]] = []
    for pred, ref in zip(predicted, reference):
        if abs(float(pred["tau_over_T"]) - float(ref["tau_over_T"])) > 1e-12:
            raise ValueError("boundary tau mismatch")
        output.append(
            {
                **pred,
                "oracle_S_star_over_K": ref["S_star_over_K"],
                "oracle_boundary_found": bool(ref["boundary_found"]),
                "oracle_missing_reason": ref["missing_reason"],
            }
        )
    return output


def evaluate_job(
    *, d: int, model_id: str, run_class: str, model: torch.nn.Module,
    cfg: dict[str, Any], oracle: Oracle, split: str, n: int,
    training_seconds: float, batch_size: int, probe_count: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    xcpu, tcpu, _ = sample_states(split, n=int(n), d=int(d), cfg=cfg, dtype=torch.float64)
    labels = oracle_labels_complete(oracle, xcpu, tcpu, cfg)
    evaluation_started = time.perf_counter()

    infer_started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, int(n), 512):
            model(
                xcpu[start:start+512].to(device=device, dtype=torch.float32),
                tcpu[start:start+512].to(device=device, dtype=torch.float32),
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - infer_started

    derivative_started = time.perf_counter()
    predictions: list[np.ndarray] = []
    component_deltas: list[np.ndarray] = []
    contractions: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    gaps: list[np.ndarray] = []
    for start in range(0, int(n), 512):
        x = xcpu[start:start+512].to(device=device, dtype=torch.float32).requires_grad_(True)
        tau = tcpu[start:start+512].to(device=device, dtype=torch.float32).requires_grad_(True)
        value = model(x, tau).reshape(-1)
        vx = torch.autograd.grad(value.sum(), x, create_graph=True, retain_graph=True)[0]
        c = cfg["contract"]
        residual, contraction = black_scholes_vi_residual(
            value, x, tau, r=float(c["r"]), q=float(c["q"]),
            sigma=float(c["sigma"]), rho=float(c["rho"]),
            hessian_mode="hutchinson", probe_count=int(probe_count),
            probe_seed=int(cfg["hutchinson_probe_seed"]),
        )
        predictions.append(value.detach().cpu().numpy())
        component_deltas.append((vx / torch.exp(x)).detach().cpu().numpy())
        contractions.append(contraction.detach().cpu().numpy())
        residuals.append(residual.detach().cpu().numpy())
        gaps.append((value - geometric_put_payoff(x)).detach().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    derivative_seconds = time.perf_counter() - derivative_started

    prediction = np.concatenate(predictions)
    component_delta = np.concatenate(component_deltas)
    contraction = np.concatenate(contractions)
    residual = np.concatenate(residuals)
    gap = np.concatenate(gaps)
    price_error = prediction - labels["price"]
    component_delta_error = component_delta - labels["delta"]
    contraction_error = contraction - labels["contraction"]
    phi = np.sqrt(gap**2 + residual**2) - gap - residual
    continuation = labels["g"] > labels["boundary"] + 1e-8
    pde_abs = np.abs(residual[continuation]) if np.any(continuation) else np.abs(residual)

    primary = float(cfg["boundary"]["primary_tolerance"])
    true_exercise = labels["g"] <= labels["boundary"] + 1e-8
    predicted_exercise = gap <= primary
    tube = np.abs(labels["g"] - labels["boundary"]) <= float(cfg["boundary"]["tube_width"])

    _, reduced_delta, reduced_gamma = diagonal_value_and_greeks(
        model, d=int(d), spots=labels["g"], taus=tcpu.numpy(), device=device
    )
    reduced_delta_error = reduced_delta - labels["reduced_delta"]
    reduced_gamma_error = reduced_gamma - labels["reduced_gamma"]

    pred_boundary = _boundary_series_model(model, int(d), cfg, primary)
    ref_boundary = _boundary_series_oracle(oracle, cfg, primary)
    compared = _compare_boundary_rows(pred_boundary, ref_boundary)
    boundary_metrics = compute_boundary_metrics(
        [row["oracle_S_star_over_K"] for row in compared],
        [row["S_star_over_K"] for row in compared],
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg["sobol_seeds"]["structural"]) + int(d))
    permutation = torch.randperm(int(d), generator=generator)
    structural_n = min(1024, int(n))
    permutation_rmse = permutation_consistency_rmse(
        model, xcpu[:structural_n], tcpu[:structural_n], permutation
    )
    x_grouped, tau_grouped, _ = grouped_transverse_states(
        groups=128, repeats=4, d=int(d),
        seed=int(cfg["sobol_seeds"]["structural"]) + 1000 + int(d),
        m_bounds=aggregate_bounds(cfg), tau_bounds=tau_bounds(cfg),
        transverse_scale=float(cfg["transverse_scale"]), dtype=torch.float64,
    )
    transverse_rmse = transverse_invariance_rmse(model, x_grouped, tau_grouped)

    finite = all(
        np.all(np.isfinite(v))
        for v in (
            prediction, component_delta, contraction, residual, gap, price_error,
            component_delta_error, contraction_error, reduced_delta_error,
            reduced_gamma_error,
        )
    )
    peak_cuda = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    scaled_gamma_error = (
        float(cfg["target_reduced_contract"]["K"]) * reduced_gamma
        - labels["scaled_reduced_gamma"]
    )
    return {
        "dimension": int(d), "model_id": model_id, "run_class": run_class, "split": split,
        "normalized_price_rmse": float(np.sqrt(np.mean(price_error**2))),
        "normalized_price_mae": float(np.mean(np.abs(price_error))),
        "normalized_price_p95": float(np.quantile(np.abs(price_error), 0.95)),
        "normalized_price_max": float(np.max(np.abs(price_error))),
        "component_delta_rmse": float(np.sqrt(np.mean(component_delta_error**2))),
        "component_delta_max": float(np.max(np.abs(component_delta_error))),
        "gamma_contraction_rmse": float(np.sqrt(np.mean(contraction_error**2))),
        "reduced_delta_rmse": float(np.sqrt(np.mean(reduced_delta_error**2))),
        "reduced_delta_max": float(np.max(np.abs(reduced_delta_error))),
        "reduced_gamma_rmse": float(np.sqrt(np.mean(reduced_gamma_error**2))),
        "reduced_gamma_max": float(np.max(np.abs(reduced_gamma_error))),
        "scaled_reduced_gamma_rmse": float(np.sqrt(np.mean(scaled_gamma_error**2))),
        "obstacle_violation_max": float(np.max(np.maximum(-gap, 0.0))),
        "obstacle_violation_rate": float(np.mean(gap < -1e-6)),
        "fb_rms": float(np.sqrt(np.mean(phi**2))),
        "fb_p95": float(np.quantile(np.abs(phi), 0.95)),
        "fb_max": float(np.max(np.abs(phi))),
        "trusted_pde_rms": float(np.sqrt(np.mean(pde_abs**2))),
        "trusted_pde_p95": float(np.quantile(pde_abs, 0.95)),
        "exercise_f1": f1_score(true_exercise, predicted_exercise),
        "boundary_tube_f1": f1_score(true_exercise[tube], predicted_exercise[tube]) if np.any(tube) else 0.0,
        **boundary_metrics,
        "permutation_consistency_rmse": permutation_rmse,
        "transverse_invariance_rmse": transverse_rmse,
        "training_seconds": float(training_seconds),
        "inference_seconds": inference_seconds,
        "derivative_seconds": derivative_seconds,
        "evaluation_seconds": time.perf_counter() - evaluation_started,
        "oracle_seconds": float(oracle.seconds),
        "peak_cpu_memory_bytes": safe_peak_cpu_memory_bytes(),
        "peak_cuda_gib": peak_cuda,
        "parameter_count": _parameter_count(model),
        "batch_size": int(batch_size),
        "hutchinson_probe_count": int(probe_count),
        "integrity": bool(finite and peak_cuda < float(cfg["max_cuda_allocation_gib"])),
    }


def _oracle_consistency(oracles: dict[int, Oracle]) -> dict[str, Any]:
    dims = sorted(oracles)
    reference = oracles[dims[0]]
    rows, passed = [], True
    for d in dims[1:]:
        other = oracles[d]
        same_grid = np.array_equal(reference.spots, other.spots) and np.array_equal(reference.taus, other.taus)
        if not same_grid:
            rows.append({"pair": f"d{dims[0]}_vs_d{d}", "same_grid": False, "passed": False})
            passed = False
            continue
        diff = np.asarray(reference.values) - np.asarray(other.values)
        max_abs, rmse = float(np.max(np.abs(diff))), float(np.sqrt(np.mean(diff**2)))
        ok = max_abs <= 1e-10 and rmse <= 1e-11
        rows.append(
            {"pair": f"d{dims[0]}_vs_d{d}", "same_grid": True,
             "max_abs_difference": max_abs, "rmse_difference": rmse, "passed": ok}
        )
        passed = passed and ok
    return {"passed": passed, "comparisons": rows}


def run_binomial(cfg: dict[str, Any]) -> dict[str, Any]:
    target = cfg["target_reduced_contract"]
    k, r, q, sigma = (
        float(target["K"]), float(target["r"]), float(target["q_g"]), float(target["sigma_g"])
    )
    steps = [int(v) for v in cfg["binomial_steps"]]
    spots = np.asarray(binomial_check_spots(cfg), float)
    check_taus = [float(v) for v in cfg["binomial_check_taus"]]
    convergence: list[dict[str, Any]] = []
    for tau_value in check_taus:
        previous: np.ndarray | None = None
        for n in steps:
            values = np.asarray(
                [crr_american_put(float(s), tau_value, k=k, r=r, q=q, sigma=sigma, steps=n) for s in spots]
            )
            if np.any(values + 1e-12 < np.maximum(k-spots, 0.0)):
                raise RuntimeError("binomial_obstacle_violation")
            if np.any(np.diff(values) > 1e-10):
                raise RuntimeError("binomial_monotonicity_violation")
            delta = None if previous is None else values - previous
            convergence.append(
                {
                    "tau_over_T": tau_value, "steps": n,
                    "max_abs_change_from_previous": None if delta is None else float(np.max(np.abs(delta))),
                    "rmse_change_from_previous": None if delta is None else float(np.sqrt(np.mean(delta**2))),
                }
            )
            previous = values

    final_steps = binomial_final_steps(cfg)
    vis = cfg["visualization"]
    anchors = np.linspace(float(vis["slice_s_min"]), float(vis["slice_s_max"]), 101)
    dense = np.linspace(float(vis["slice_s_min"]), float(vis["slice_s_max"]), int(vis["slice_points"]))
    for tau_value in vis["slice_taus"]:
        anchor_values = np.asarray(
            [crr_american_put(float(s), float(tau_value), k=k, r=r, q=q, sigma=sigma, steps=final_steps) for s in anchors]
        )
        values = np.interp(dense, anchors, anchor_values)
        tag = {1.0: "1", 0.5: "05", 0.1: "01"}[float(tau_value)]
        _write_csv(
            VIS_DIR / "binomial" / f"price_slice_tau_{tag}.csv",
            [
                {
                    "S_over_K": float(s), "tau_over_T": float(tau_value), "V_over_K": float(v),
                    "payoff_over_K": float(max(1.0-s,0.0)), "tree_steps": final_steps,
                    "interpolated_from_anchor_grid": True,
                }
                for s,v in zip(dense, values)
            ],
        )
    payload = {
        "contract": target, "check_spots": spots.tolist(), "check_taus": check_taus,
        "convergence": convergence, "final_curve_steps": final_steps,
        "final_curve_anchor_points": len(anchors), "final_curve_output_points": len(dense),
    }
    _runtime.atomic_json(RESULT_DIR / "binomial_convergence.json", payload)
    return payload


def _base_metadata(d: int, cfg: dict[str, Any]) -> dict[str, Any]:
    b, vis = cfg["boundary"], cfg["visualization"]
    return {
        "experiment_id": cfg["experiment_id"], "source_git_commit": git_commit(),
        "config_sha256": config_hash(), "seed": 101, "dimension": int(d),
        "single_seed": True, "diagnostic_only": True, "posthoc_matched_contract": True,
        "random_seed_robustness": False, "candidate_supported": False,
        "blind_holdout_access": False, "training_domain": training_domain_metadata(cfg),
        "target_reduced_contract": cfg["target_reduced_contract"], "asset_contract": cfg["contract"],
        "rho": float(cfg["rho"]), "parameter_mapping": parameter_mapping(int(d), load_config()),
        "objective": cfg["objective"], "boundary_definition": BOUNDARY_METHOD,
        "primary_boundary_tolerance": float(b["primary_tolerance"]),
        "boundary_sensitivity_tolerances": [float(v) for v in b["sensitivity_tolerances"]],
        "boundary_evaluation_domain": {
            "G_over_K": [float(b["s_min"]), float(b["s_max"])],
            "tau_over_T": [float(b["tau_min"]), float(b["tau_max"])],
            "G_points": int(b["s_points"]), "tau_points": int(b["tau_points"]),
        },
        "visualization_domain": vis,
        "component_delta_definition": COMPONENT_DELTA_DEFINITION,
        "gamma_contraction_definition": GAMMA_CONTRACTION_DEFINITION,
        "reduced_delta_definition": REDUCED_DELTA_DEFINITION,
        "reduced_gamma_definition": REDUCED_GAMMA_DEFINITION,
        "scaled_reduced_gamma_definition": "K*Gamma_G; K=1 for this matched contract",
        "pointwise_pde_residual_exported": False, "pointwise_fb_residual_exported": False,
        "residual_export_note": "PDE/FB residuals are summarized in run_metrics.csv but omitted from dense surface.npz.",
        "common_reference_policy": "Later Arm C/Arm D/HD ranking must use Ruixi DIRK+Policy Iteration+sinh-grid M=480,N=960 as one external reference; it is not accessed during HD training.",
    }


def _slice_rows(payload: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    keys = [
        "S_over_K", "tau_over_T", "V_over_K", "payoff_over_K", "premium_over_K",
        "reduced_delta", "reduced_gamma", "scaled_reduced_gamma", "exercise_like_primary",
    ]
    return [
        {key: (bool(value) if key == "exercise_like_primary" else float(value))
         for key,value in zip(keys, values)}
        for values in zip(*(payload[key] for key in keys))
    ]


def _save_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def export_model_visualization(
    d: int, model_id: str, model: torch.nn.Module, oracle: Oracle, cfg: dict[str, Any],
) -> dict[str, Any]:
    vis = cfg["visualization"]
    out_dir = VIS_DIR / f"d{int(d)}" / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    slice_spots = np.linspace(float(vis["slice_s_min"]), float(vis["slice_s_max"]), int(vis["slice_points"]))
    for tau_value in vis["slice_taus"]:
        payload = model_surface_arrays(
            model, d=int(d), spots=slice_spots,
            taus=np.full_like(slice_spots, float(tau_value)), cfg=cfg,
        )
        tag = {1.0:"1",0.5:"05",0.1:"01"}[float(tau_value)]
        _write_csv(out_dir / f"price_slice_tau_{tag}.csv", _slice_rows(payload))

    s1 = np.linspace(float(vis["surface_s_min"]), float(vis["surface_s_max"]), int(vis["surface_s_points"]))
    t1 = np.linspace(float(vis["surface_tau_min"]), float(vis["surface_tau_max"]), int(vis["surface_tau_points"]))
    ss,tt = np.meshgrid(s1,t1)
    surface = model_surface_arrays(model,d=int(d),spots=ss.ravel(),taus=tt.ravel(),cfg=cfg)
    _save_npz(out_dir / "surface.npz", surface)

    primary = float(cfg["boundary"]["primary_tolerance"])
    pred = _boundary_series_model(model,int(d),cfg,primary)
    ref = _boundary_series_oracle(oracle,cfg,primary)
    comparison = _compare_boundary_rows(pred,ref)
    _write_csv(out_dir / "free_boundary.csv", comparison)
    metrics = compute_boundary_metrics(
        [r["oracle_S_star_over_K"] for r in comparison],
        [r["S_star_over_K"] for r in comparison],
    )
    sensitivity: list[dict[str,Any]] = []
    for tolerance in [primary,*[float(v) for v in cfg["boundary"]["sensitivity_tolerances"]]]:
        for row in _compare_boundary_rows(
            _boundary_series_model(model,int(d),cfg,tolerance),
            _boundary_series_oracle(oracle,cfg,tolerance),
        ):
            sensitivity.append({"tolerance_role":"primary" if tolerance == primary else "sensitivity",**row})
    _write_csv(out_dir / "free_boundary_sensitivity.csv", sensitivity)

    metadata = {
        **_base_metadata(int(d),cfg), "model":model_id,
        "source_checkpoint_sha256":_runtime.sha256_file(_checkpoint_path(int(d),model_id)),
        "surface_points":len(surface["V_over_K"]),
        "max_obstacle_violation_on_surface":float(np.max(np.maximum(-surface["premium_over_K"],0.0))),
        **metrics,
    }
    _runtime.atomic_json(out_dir / "metadata.json", metadata)
    _runtime.atomic_json(
        out_dir / "checksums.json",
        {p.name:_runtime.sha256_file(p) for p in sorted(out_dir.iterdir()) if p.is_file() and p.name != "checksums.json"},
    )
    return metadata


def export_oracle_visualization(d: int, oracle: Oracle, cfg: dict[str, Any]) -> dict[str,Any]:
    vis = cfg["visualization"]
    out_dir = VIS_DIR / f"d{int(d)}" / "oracle"
    out_dir.mkdir(parents=True,exist_ok=True)
    slice_spots=np.linspace(float(vis["slice_s_min"]),float(vis["slice_s_max"]),int(vis["slice_points"]))
    for tau_value in vis["slice_taus"]:
        payload=oracle_surface_arrays(
            oracle,spots=slice_spots,taus=np.full_like(slice_spots,float(tau_value)),cfg=cfg
        )
        tag={1.0:"1",0.5:"05",0.1:"01"}[float(tau_value)]
        _write_csv(out_dir/f"price_slice_tau_{tag}.csv",_slice_rows(payload))
    s1=np.linspace(float(vis["surface_s_min"]),float(vis["surface_s_max"]),int(vis["surface_s_points"]))
    t1=np.linspace(float(vis["surface_tau_min"]),float(vis["surface_tau_max"]),int(vis["surface_tau_points"]))
    ss,tt=np.meshgrid(s1,t1)
    surface=oracle_surface_arrays(oracle,spots=ss.ravel(),taus=tt.ravel(),cfg=cfg)
    _save_npz(out_dir/"surface.npz",surface)

    primary=float(cfg["boundary"]["primary_tolerance"])
    _write_csv(out_dir/"free_boundary.csv",_boundary_series_oracle(oracle,cfg,primary))
    sensitivity=[]
    for tolerance in [primary,*[float(v) for v in cfg["boundary"]["sensitivity_tolerances"]]]:
        for row in _boundary_series_oracle(oracle,cfg,tolerance):
            sensitivity.append({"tolerance_role":"primary" if tolerance == primary else "sensitivity",**row})
    _write_csv(out_dir/"free_boundary_sensitivity.csv",sensitivity)
    metadata={
        **_base_metadata(int(d),cfg),"model":"CN_PSOR_reduced_oracle",
        "source_checkpoint_sha256":None,"oracle_runtime_sha256":_runtime.sha256_file(ORACLE_DIR/f"d{int(d)}.npz"),
        "oracle_refinement_rmse":float(oracle.refinement_rmse),
        "oracle_boundary_method":BOUNDARY_METHOD,"oracle_boundary_tolerance":primary,
        "surface_points":len(surface["V_over_K"]),
    }
    _runtime.atomic_json(out_dir/"metadata.json",metadata)
    _runtime.atomic_json(
        out_dir/"checksums.json",
        {p.name:_runtime.sha256_file(p) for p in sorted(out_dir.iterdir()) if p.is_file() and p.name!="checksums.json"},
    )
    return metadata


def run_preflight() -> dict[str, Any]:
    cfg=load_config()
    RESULT_DIR.mkdir(parents=True,exist_ok=True)
    manifest=environment_manifest()
    _runtime.atomic_json(RESULT_DIR/"environment_manifest.json",manifest)
    _runtime.atomic_json(
        RESULT_DIR/"config_manifest.json",
        {
            "experiment_id":cfg["experiment_id"],"config":cfg,"config_sha256":config_hash(),
            "implementation_sha":git_commit(),
            "historical_result_directory_preserved":str(OLD_RESULT_DIR.relative_to(ROOT)),
        },
    )
    mappings=[parameter_mapping(int(d),cfg) for d in cfg["canonical_dimensions"]]
    _write_csv(RESULT_DIR/"parameter_mapping.csv",mappings)
    if not manifest["cuda_available"]:
        raise RuntimeError("cuda_required:boundary-complete matched preflight")
    if int(manifest.get("gpu_total_memory_bytes") or 0) < 7*1024**3:
        raise RuntimeError("cuda_memory_blocked")

    oracles:dict[int,Oracle]={}; refinement=[]
    for d in cfg["canonical_dimensions"]:
        dim_cfg=dimension_config(int(d),cfg)
        oracle,evidence=build_boundary_complete_oracle(int(d),dim_cfg)
        save_oracle(oracle); oracles[int(d)]=oracle; refinement.append(evidence)
    consistency=_oracle_consistency(oracles)
    _runtime.atomic_json(RESULT_DIR/"oracle_consistency.json",consistency)
    _runtime.atomic_json(
        RESULT_DIR/"oracle_refinement.json",
        {"passed":all(float(r["refinement_rmse"])<=float(cfg["oracle_rmse_max"]) for r in refinement),
         "domain":training_domain_metadata(cfg),"rows":refinement},
    )
    if not consistency["passed"]:
        raise RuntimeError("oracle_cross_dimension_consistency_failed")
    binomial=run_binomial(cfg)
    payload={
        "passed":True,"implementation_sha":git_commit(),"config_sha256":config_hash(),
        "new_result_namespace":str(RESULT_DIR.relative_to(ROOT)),
        "old_result_namespace":str(OLD_RESULT_DIR.relative_to(ROOT)),
        "mappings":mappings,"oracle_consistency":consistency,
        "oracle_refinement":refinement,"binomial":binomial,
    }
    _runtime.atomic_json(RESULT_DIR/"preflight.json",payload)
    return payload


def run_canonical() -> list[dict[str,Any]]:
    _install_training_namespace()
    cfg=load_config()
    return [
        _old.train_job(int(d),str(model_id),run_class="canonical")
        for d in cfg["canonical_dimensions"] for model_id in cfg["models"]
    ]


def run_finalize() -> list[dict[str,Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required:boundary-complete matched finalization")
    _install_training_namespace()
    cfg=load_config(); rows=[]
    for d in cfg["canonical_dimensions"]:
        dim_cfg=dimension_config(int(d),cfg); oracle=load_oracle(int(d))
        export_oracle_visualization(int(d),oracle,dim_cfg)
        for model_id in cfg["models"]:
            status=_status_path(int(d),str(model_id))
            if not status.exists() or json.loads(status.read_text(encoding="utf-8")).get("status")!="completed":
                raise RuntimeError(f"missing_completed_job:d{d}:{model_id}")
            model,checkpoint=_old._load_completed_model(int(d),str(model_id),"canonical",dim_cfg)
            row=evaluate_job(
                d=int(d),model_id=str(model_id),run_class="canonical",model=model,cfg=dim_cfg,
                oracle=oracle,split="test",n=int(dim_cfg["test_points"]),
                training_seconds=float(checkpoint["elapsed_training_seconds"]),
                batch_size=int(checkpoint["batch_size"]),probe_count=int(dim_cfg["hutchinson_probe_count"]),
            )
            row.update(parameter_mapping(int(d),cfg)); rows.append(row)
            export_model_visualization(int(d),str(model_id),model,oracle,dim_cfg)

    _write_csv(RESULT_DIR/"run_metrics.csv",rows)
    _write_csv(
        RESULT_DIR/"structural_diagnostics.csv",
        [{k:r[k] for k in ("dimension","model_id","permutation_consistency_rmse","transverse_invariance_rmse")} for r in rows],
    )
    oracle_cost={str(int(d)):float(load_oracle(int(d)).seconds) for d in cfg["canonical_dimensions"]}
    compute=[]
    for r in rows:
        row={k:r[k] for k in (
            "dimension","model_id","training_seconds","inference_seconds","derivative_seconds",
            "evaluation_seconds","parameter_count","batch_size","peak_cuda_gib","oracle_seconds"
        )}
        row["training_plus_oracle_seconds"]=float(r["training_seconds"])+oracle_cost[str(int(r["dimension"]))]
        compute.append(row)
    _write_csv(RESULT_DIR/"compute_accounting.csv",compute)
    _runtime.atomic_json(
        RESULT_DIR/"compute_accounting.json",
        {
            "single_seed":True,"seed":101,"gpu":environment_manifest().get("gpu_name"),
            "jobs":compute,"oracle_seconds_by_dimension":oracle_cost,
            "total_oracle_generation_seconds":float(sum(oracle_cost.values())),
            "total_gpu_training_seconds_sum":float(sum(float(r["training_seconds"]) for r in rows)),
            "training_plus_oracle_seconds_definition":"per-job training time plus that dimension's one-time CN/PSOR oracle generation cost",
            "protocol_disclosure":"HD uses oracle-assisted supervised learning plus an HVI physics term; Ruixi Arm C/D are physics-only PINNs. Report observed accuracy-compute trade-off under the respective protocols, not a pure algorithmic speedup.",
        },
    )
    summary_keys=(
        "dimension","model_id","normalized_price_rmse","component_delta_rmse",
        "gamma_contraction_rmse","reduced_delta_rmse","reduced_gamma_rmse",
        "scaled_reduced_gamma_rmse","trusted_pde_p95","exercise_f1","boundary_tube_f1",
        "boundary_coverage","boundary_conditional_rmse","boundary_full_rmse",
        "training_seconds","parameter_count",
    )
    _write_csv(RESULT_DIR/"dimension_summary.csv",[{k:r[k] for k in summary_keys} for r in rows])
    _runtime.atomic_json(
        RESULT_DIR/"final_status.json",
        {
            "status":"completed","experiment_id":cfg["experiment_id"],
            "implementation_sha":git_commit(),"config_sha256":config_hash(),
            "diagnostic_only":True,"single_seed":True,"seed":101,
            "posthoc_matched_contract":True,"random_seed_robustness":False,
            "candidate_supported":False,"blind_holdout_access":False,
            "canonical_dimensions":cfg["canonical_dimensions"],"models":cfg["models"],
            "rows":len(rows),"target_reduced_contract":cfg["target_reduced_contract"],
            "training_domain":training_domain_metadata(cfg),
            "historical_result_directory_modified":False,
            "common_reference_note":"Cross-repository ranking must later use one shared Ruixi DIRK+Policy Iteration+sinh-grid high-accuracy reference.",
        },
    )
    hashes={}
    for path in RESULT_DIR.rglob("*"):
        if path.is_file() and RUNTIME_DIR not in path.parents and path.name!="artifact_hashes.json":
            hashes[str(path.relative_to(RESULT_DIR))]=_runtime.sha256_file(path)
    _runtime.atomic_json(
        RESULT_DIR/"artifact_hashes.json",
        {"experiment_id":cfg["experiment_id"],"implementation_sha":git_commit(),
         "config_sha256":config_hash(),"sha256":hashes},
    )
    return rows


def status_snapshot() -> dict[str,Any]:
    cfg=load_config(); jobs=[]
    for d in cfg["canonical_dimensions"]:
        for model_id in cfg["models"]:
            path=_status_path(int(d),str(model_id))
            jobs.append(
                json.loads(path.read_text(encoding="utf-8")) if path.exists()
                else {"dimension":int(d),"model_id":str(model_id),"status":"not_started"}
            )
    return {
        "experiment":cfg["experiment_id"],"git_commit":git_commit(),"config_sha256":config_hash(),
        "result_directory":str(RESULT_DIR),"historical_result_directory":str(OLD_RESULT_DIR),
        "preflight":json.loads((RESULT_DIR/"preflight.json").read_text(encoding="utf-8"))
        if (RESULT_DIR/"preflight.json").exists() else {"status":"not_run"},
        "jobs":jobs,
        "final_status":json.loads((RESULT_DIR/"final_status.json").read_text(encoding="utf-8"))
        if (RESULT_DIR/"final_status.json").exists() else {"status":"not_finalized"},
    }


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=["preflight","canonical","finalize","status"],required=True)
    parser.add_argument("--dimension",type=int)
    parser.add_argument("--model",choices=["FC_HVI","DEEPSETS_HVI"])
    parser.add_argument("--max-steps-override",type=int)
    args=parser.parse_args(argv)
    if args.phase=="preflight":
        result:Any=run_preflight()
    elif args.phase=="canonical":
        _install_training_namespace()
        if args.dimension is not None or args.model is not None:
            if args.dimension is None or args.model is None:
                raise SystemExit("--dimension and --model must be supplied together")
            result=_old.train_job(
                args.dimension,args.model,run_class="canonical",
                max_steps_override=args.max_steps_override,
            )
        else:
            result=run_canonical()
    elif args.phase=="finalize":
        result=run_finalize()
    else:
        result=status_snapshot()
    print(json.dumps(result,indent=2,sort_keys=True,default=str))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
