#!/usr/bin/env python3
"""M09 multi-fidelity screen using complete matched CN/PSOR regime surfaces."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from data.splits import three_way_regime_split
from models.mlp import build_mlp
from solvers.boundaries import boundary_curve
from solvers.cn_psor import SolverConfig, fd_operator_matrix, solve_option, time_boundary_values
from solvers.greeks import delta_gamma, greek_diagnostic_masks

from m09_multifidelity_core import (
    ORDINARY_METRICS,
    RESIDUAL_METRICS,
    assert_complete_regime_rows,
    candidate_passes,
    choose_smallest_matching_fraction,
    classify_status,
    matched_accuracy_target,
    nested_regime_subsets,
    paired_bootstrap_mean_ci,
    percent_degradation,
    validate_config,
)

FEATURE_NAMES = ("log_moneyness", "tau_over_T", "r", "q", "sigma", "T", "is_call")
ALL_METRICS = ORDINARY_METRICS + RESIDUAL_METRICS


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _linspace(spec: Mapping[str, Any]) -> np.ndarray:
    return np.linspace(float(spec["min"]), float(spec["max"]), int(spec["count"]), dtype=float)


def build_regimes(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    grid = config["parameter_grid"]
    regimes: list[dict[str, Any]] = []
    flat = 0
    for sigma in _linspace(grid["sigma"]):
        for rate in _linspace(grid["r"]):
            for dividend in _linspace(grid["q"]):
                regimes.append({"id": f"regime_{flat:04d}", "case_id": f"regime_{flat:04d}", "sigma": float(sigma), "r": float(rate), "q": float(dividend)})
                flat += 1
    return regimes


def split_regimes(config: Mapping[str, Any], regimes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    spec = config["split"]
    return three_way_regime_split(regimes, train_fraction=float(spec["train_fraction"]), validation_fraction=float(spec["validation_fraction"]), seed=int(spec["seed"]))


def solver_config(base: Mapping[str, Any], regime: Mapping[str, Any]) -> SolverConfig:
    payload = dict(base)
    payload.update({"sigma": float(regime["sigma"]), "r": float(regime["r"]), "q": float(regime["q"])})
    return SolverConfig(**payload)


def _surface_features(result: Any, cfg: SolverConfig) -> np.ndarray:
    spots_mesh, tau_mesh = np.meshgrid(result.spots, result.taus)
    return np.column_stack([
        np.log(np.maximum(spots_mesh.ravel(), 1e-12) / cfg.k),
        tau_mesh.ravel() / cfg.t if cfg.t > 0.0 else tau_mesh.ravel(),
        np.full(spots_mesh.size, cfg.r), np.full(spots_mesh.size, cfg.q), np.full(spots_mesh.size, cfg.sigma),
        np.full(spots_mesh.size, cfg.t), np.full(spots_mesh.size, 1.0 if cfg.option_type == "call" else 0.0),
    ]).astype(np.float64)


def solve_surface(base: Mapping[str, Any], regime: Mapping[str, Any], fidelity: str) -> dict[str, Any]:
    cfg = solver_config(base, regime)
    started = time.perf_counter()
    result = solve_option(cfg)
    elapsed = float(time.perf_counter() - started)
    if not np.isfinite(result.values).all() or not all(result.lcp_converged):
        raise RuntimeError(f"CN/PSOR failed for {regime['id']} at {fidelity} fidelity")
    features = _surface_features(result, cfg)
    payoff_surface = np.repeat(result.payoff[None, :], len(result.taus), axis=0)
    return {"regime_id": str(regime["id"]), "fidelity": fidelity, "cfg": cfg, "result": result, "features": features,
            "target": (result.values / cfg.k).ravel().astype(np.float64), "payoff": (payoff_surface / cfg.k).ravel().astype(np.float64), "solve_seconds": elapsed}


def build_surface_cache(coarse_solver: Mapping[str, Any], fine_solver: Mapping[str, Any], partitions: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    cache: dict[str, dict[str, dict[str, Any]]] = {"coarse": {}, "fine": {}}
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "validation", "test"):
        for regime in partitions[split_name]:
            fidelities = ("coarse", "fine") if split_name == "train" else ("fine",)
            for fidelity in fidelities:
                surface = solve_surface(coarse_solver if fidelity == "coarse" else fine_solver, regime, fidelity)
                cache[fidelity][str(regime["id"])] = surface
                rows.append({"split": split_name, "regime_id": regime["id"], "fidelity": fidelity, "nodes": int(len(surface["target"])), "solve_seconds": surface["solve_seconds"]})
    return cache, rows


def concatenate_surfaces(cache: Mapping[str, dict[str, Any]], regime_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = tuple(str(x) for x in regime_ids)
    features = np.concatenate([cache[key]["features"] for key in ids], axis=0)
    targets = np.concatenate([cache[key]["target"] for key in ids], axis=0)
    regime_key = np.concatenate([np.full(len(cache[key]["target"]), key, dtype=object) for key in ids])
    return features, targets, regime_key


def fit_normalizer(train_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train_features, axis=0, dtype=np.float64)
    std = np.std(train_features, axis=0, dtype=np.float64)
    return mean, np.where(std > 1e-12, std, 1.0)


def normalize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(features, dtype=np.float64) - mean) / std


def model_parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def train_model(model: Any, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, spec: Mapping[str, Any], *, seed: int, arm: str, fraction: float | None, phase: str) -> tuple[Any, list[dict[str, Any]], float]:
    import torch
    import torch.nn.functional as F
    torch.manual_seed(int(seed))
    model = model.to(device="cpu", dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(spec["learning_rate"]))
    x_train = torch.as_tensor(train_x, dtype=torch.float64); y_train = torch.as_tensor(train_y[:, None], dtype=torch.float64)
    x_val = torch.as_tensor(val_x, dtype=torch.float64); y_val = torch.as_tensor(val_y[:, None], dtype=torch.float64)
    epochs, batch_size, patience = int(spec["epochs"]), int(spec["batch_size"]), int(spec["patience"])
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}; best_val = float("inf"); stale = 0
    history: list[dict[str, Any]] = []; started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        generator = torch.Generator(device="cpu"); generator.manual_seed(int(seed) * 1000 + epoch)
        order = torch.randperm(len(x_train), generator=generator); total = 0.0; model.train()
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]; pred = model(x_train[idx]); loss = F.mse_loss(pred, y_train[idx])
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); total += float(loss.detach().cpu()) * len(idx)
        model.eval()
        with torch.no_grad(): val_rmse = float(torch.sqrt(F.mse_loss(model(x_val), y_val)).cpu())
        history.append({"seed": int(seed), "arm": arm, "fraction": "" if fraction is None else float(fraction), "phase": phase, "epoch": epoch, "train_loss": total / max(len(x_train), 1), "validation_price_rmse_raw": val_rmse})
        if val_rmse < best_val:
            best_val, stale = val_rmse, 0; best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience: break
    model.load_state_dict(best_state)
    return model, history, float(time.perf_counter() - started)


def predict_model(model: Any, features: np.ndarray, *, batch_size: int) -> np.ndarray:
    import torch
    model.eval(); rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), int(batch_size)):
            xb = torch.as_tensor(features[start:start + int(batch_size)], dtype=torch.float64)
            rows.append(model(xb).detach().cpu().numpy().reshape(-1))
    return np.concatenate(rows)


def enforce_known_constraints(values: np.ndarray, result: Any, cfg: SolverConfig) -> np.ndarray:
    candidate = np.maximum(np.asarray(values, dtype=float).copy(), result.payoff[None, :]); candidate[0, :] = result.payoff
    for i, tau in enumerate(result.taus):
        left, right = time_boundary_values(cfg, float(tau)); candidate[i, 0] = left; candidate[i, -1] = right
    return candidate


def _operator_coefficients(cfg: SolverConfig, spots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ds = float(spots[1] - spots[0]); interior = spots[1:-1]; diffusion = 0.5 * cfg.sigma**2 * interior**2; drift = (cfg.r - cfg.q) * interior
    return diffusion / ds**2 - drift / (2.0 * ds), diffusion / ds**2 + drift / (2.0 * ds)


def cn_residual_fields(values: np.ndarray, cfg: SolverConfig, result: Any) -> dict[str, np.ndarray]:
    spots, taus, payoff = result.spots, result.taus, result.payoff; l_mat = fd_operator_matrix(cfg, spots); lower, upper = _operator_coefficients(cfg, spots)
    eye = np.eye(cfg.m - 1); obstacle = payoff[1:-1]; residual = np.empty((cfg.n, cfg.m - 1), dtype=float); gap = values[1:, 1:-1] - obstacle[None, :]
    def source(tau: float) -> np.ndarray:
        out = np.zeros(cfg.m - 1, dtype=float); left, right = time_boundary_values(cfg, tau); out[0] = lower[0] * left; out[-1] = upper[-1] * right; return out
    for step in range(cfg.n):
        dt = float(taus[step + 1] - taus[step]); a = eye - 0.5 * dt * l_mat; b = eye + 0.5 * dt * l_mat
        rhs = b @ values[step, 1:-1] + 0.5 * dt * (source(float(taus[step])) + source(float(taus[step + 1]))); residual[step] = a @ values[step + 1, 1:-1] - rhs
    return {"residual": residual, "gap": gap}


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)))


def surface_metrics(candidate: np.ndarray, surface: Mapping[str, Any], *, contact_tol: float) -> dict[str, float]:
    result = surface["result"]; cfg: SolverConfig = surface["cfg"]; reference = result.values
    reference_boundary = boundary_curve(result.spots, reference, result.payoff, cfg.option_type, tol=contact_tol)
    candidate_boundary = boundary_curve(result.spots, candidate, result.payoff, cfg.option_type, tol=contact_tol)
    ref_delta, ref_gamma = delta_gamma(reference, result.spots); cand_delta, cand_gamma = delta_gamma(candidate, result.spots)
    strict = np.asarray(greek_diagnostic_masks(result.spots, reference_boundary, cfg.k)["strict"], dtype=bool)
    boundary_mask = np.isfinite(reference_boundary) & np.isfinite(candidate_boundary) & (result.taus > 0.0)
    boundary_rmse = _rmse(candidate_boundary[boundary_mask], reference_boundary[boundary_mask]) if np.any(boundary_mask) else 0.0
    fields = cn_residual_fields(candidate, cfg, result); gap, residual = fields["gap"], fields["residual"]
    reference_gap = reference[1:, 1:-1] - result.payoff[1:-1][None, :]; continuation = reference_gap > contact_tol
    return {"price_rmse": _rmse(candidate, reference), "delta_rmse": _rmse(cand_delta[strict], ref_delta[strict]) if np.any(strict) else 0.0,
            "gamma_rmse": _rmse(cand_gamma[strict], ref_gamma[strict]) if np.any(strict) else 0.0, "boundary_rmse": boundary_rmse,
            "obstacle_violation": float(np.max(np.maximum(-gap, 0.0))), "dual_violation": float(np.max(np.maximum(-residual, 0.0))),
            "complementarity": float(np.max(np.abs(gap * residual))), "continuation_residual": float(np.sqrt(np.mean(residual[continuation] ** 2))) if np.any(continuation) else 0.0}


def evaluate_model(model: Any, fine_cache: Mapping[str, dict[str, Any]], regime_ids: Sequence[str], mean: np.ndarray, std: np.ndarray, model_spec: Mapping[str, Any], diagnostics: Mapping[str, Any], *, seed: int, arm: str, fraction: float | None, split_name: str) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []; started = time.perf_counter()
    for regime_id in regime_ids:
        surface = fine_cache[str(regime_id)]; result = surface["result"]; cfg: SolverConfig = surface["cfg"]
        pred = predict_model(model, normalize(surface["features"], mean, std), batch_size=int(model_spec["batch_size"]))
        candidate = enforce_known_constraints((pred * cfg.k).reshape(result.values.shape), result, cfg)
        rows.append({"seed": int(seed), "arm": arm, "fraction": "" if fraction is None else float(fraction), "split": split_name, "regime_id": str(regime_id), **surface_metrics(candidate, surface, contact_tol=float(diagnostics["contact_tol"]))})
    return rows, float(time.perf_counter() - started)


def _aggregate_price(rows: Sequence[Mapping[str, Any]]) -> float:
    return float(np.sqrt(np.mean([float(row["price_rmse"]) ** 2 for row in rows])))


def _fraction_label_seconds(surface_rows: Sequence[Mapping[str, Any]], selected_keys: Sequence[str]) -> float:
    selected = set(str(x) for x in selected_keys)
    return float(sum(float(row["solve_seconds"]) for row in surface_rows if row["split"] == "train" and row["fidelity"] == "fine" and str(row["regime_id"]) in selected))


def _fixed_eval_label_seconds(surface_rows: Sequence[Mapping[str, Any]]) -> float:
    return float(sum(float(row["solve_seconds"]) for row in surface_rows if row["fidelity"] == "fine" and row["split"] in {"validation", "test"}))


def _time_inference(model: Any, features: np.ndarray, mean: np.ndarray, std: np.ndarray, spec: Mapping[str, Any], timing: Mapping[str, Any]) -> tuple[float, float]:
    x = normalize(features, mean, std)
    for _ in range(int(timing["warmups"])): predict_model(model, x, batch_size=int(spec["batch_size"]))
    samples = []
    for _ in range(int(timing["repetitions"])):
        started = time.perf_counter(); predict_model(model, x, batch_size=int(spec["batch_size"])); samples.append(time.perf_counter() - started)
    return float(np.mean(samples)), float(np.std(samples))


def _metric_degradation_summary(selected_rows: Sequence[Mapping[str, Any]], *, bootstrap: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    baseline = [r for r in selected_rows if r["arm"] == "fine_only"]; candidate = [r for r in selected_rows if r["arm"] == "coarse_plus_few_fine"]
    regime_ids = sorted({str(r["regime_id"]) for r in baseline} & {str(r["regime_id"]) for r in candidate}); summary: dict[str, float] = {}; audit_rows: list[dict[str, Any]] = []
    floors = {name: float(diagnostics["ordinary_floor"]) for name in ("price_rmse", "delta_rmse", "gamma_rmse", "boundary_rmse")}; floors["obstacle_violation"] = float(diagnostics["obstacle_floor"])
    for name in RESIDUAL_METRICS: floors[name] = float(diagnostics["residual_floor"])
    for offset, metric in enumerate(ALL_METRICS):
        base_by_regime, cand_by_regime = [], []
        for regime_id in regime_ids:
            b = [float(r[metric]) for r in baseline if str(r["regime_id"]) == regime_id]; c = [float(r[metric]) for r in candidate if str(r["regime_id"]) == regime_id]
            if b and c: base_by_regime.append(float(np.mean(b))); cand_by_regime.append(float(np.mean(c)))
        degradation = percent_degradation(cand_by_regime, base_by_regime, floor=floors[metric])
        point, low, high = paired_bootstrap_mean_ci(degradation, repetitions=int(bootstrap["repetitions"]), seed=int(bootstrap["seed"]) + offset)
        summary[f"degradation_pct_{metric}"] = point; summary[f"degradation_ci_low_pct_{metric}"] = low; summary[f"degradation_ci_upper_pct_{metric}"] = high
        audit_rows.append({"metric": metric, "degradation_pct": point, "ci_low_pct": low, "ci_upper_pct": high})
    return summary, audit_rows


def run(config_path: str | Path, *, mode: str | None = None) -> dict[str, Any]:
    config_path = Path(config_path); config = json.loads(config_path.read_text(encoding="utf-8")); validate_config(config)
    evidence_mode = str(mode or config["evidence_mode"])
    if evidence_mode not in {"smoke", "canonical"}: raise ValueError("mode must be smoke or canonical")
    runtime = copy.deepcopy(config)
    if evidence_mode == "smoke":
        smoke = runtime["smoke"]; runtime["training_seeds"] = list(smoke["training_seeds"]); runtime["fine_fractions"] = list(smoke["fine_fractions"]); runtime["model"]["epochs"] = int(smoke["epochs"])
    out_dir = Path(runtime["output_root"]); out_dir.mkdir(parents=True, exist_ok=True); started_total = time.perf_counter(); regimes = build_regimes(runtime)
    if evidence_mode == "smoke": regimes = regimes[:int(runtime["smoke"]["max_regimes"])]
    partitions = split_regimes(runtime, regimes)
    if min(len(partitions[name]) for name in ("train", "validation", "test")) < 1: raise RuntimeError("M09 requires nonempty train/validation/test regime partitions")
    train_ids = [str(r["id"]) for r in partitions["train"]]; val_ids = [str(r["id"]) for r in partitions["validation"]]; test_ids = [str(r["id"]) for r in partitions["test"]]
    fractions = tuple(float(x) for x in runtime["fine_fractions"]); subsets = nested_regime_subsets(train_ids, fractions, selection_seed=int(runtime["selection_seed"]))
    cache, surface_rows = build_surface_cache(runtime["coarse_solver"], runtime["fine_solver"], partitions)
    coarse_nodes = (int(runtime["coarse_solver"]["m"]) + 1) * (int(runtime["coarse_solver"]["n"]) + 1); fine_nodes = (int(runtime["fine_solver"]["m"]) + 1) * (int(runtime["fine_solver"]["n"]) + 1)
    _, _, coarse_regime_key = concatenate_surfaces(cache["coarse"], train_ids); assert_complete_regime_rows(coarse_regime_key, coarse_nodes, train_ids)
    _, _, fine_regime_key = concatenate_surfaces(cache["fine"], train_ids); assert_complete_regime_rows(fine_regime_key, fine_nodes, train_ids)
    fine_train_x_all, _, _ = concatenate_surfaces(cache["fine"], train_ids); mean, std = fit_normalizer(fine_train_x_all)
    val_x, val_y, _ = concatenate_surfaces(cache["fine"], val_ids); val_x = normalize(val_x, mean, std)
    model_spec = runtime["model"]; hidden, depth = int(model_spec["hidden"]), int(model_spec["depth"])
    all_history: list[dict[str, Any]] = []; metric_rows: list[dict[str, Any]] = []; arm_rows: list[dict[str, Any]] = []; trained: dict[tuple[int, str, float | None], Any] = {}
    for seed in runtime["training_seeds"]:
        seed = int(seed); coarse_x, coarse_y, _ = concatenate_surfaces(cache["coarse"], train_ids); coarse_x = normalize(coarse_x, mean, std)
        coarse_model = build_mlp(len(FEATURE_NAMES), 1, hidden=hidden, depth=depth)
        coarse_model, history, coarse_seconds = train_model(coarse_model, coarse_x, coarse_y, val_x, val_y, model_spec, seed=seed, arm="coarse_only", fraction=None, phase="coarse_pretrain")
        all_history.extend(history); trained[(seed, "coarse_only", None)] = copy.deepcopy(coarse_model)
        val_metrics, val_eval_seconds = evaluate_model(coarse_model, cache["fine"], val_ids, mean, std, model_spec, runtime["diagnostics"], seed=seed, arm="coarse_only", fraction=None, split_name="validation")
        test_metrics, test_eval_seconds = evaluate_model(coarse_model, cache["fine"], test_ids, mean, std, model_spec, runtime["diagnostics"], seed=seed, arm="coarse_only", fraction=None, split_name="test")
        metric_rows.extend(val_metrics + test_metrics); arm_rows.append({"seed": seed, "arm": "coarse_only", "fraction": "", "validation_price_rmse": _aggregate_price(val_metrics), "coarse_pretrain_seconds": coarse_seconds, "fine_tune_seconds": 0.0, "training_seconds": coarse_seconds, "validation_eval_seconds": val_eval_seconds, "test_eval_seconds": test_eval_seconds, "fine_train_label_seconds": 0.0, "interpolation_seconds": 0.0})
        for fraction in fractions:
            selected_ids = subsets[fraction]; train_x, train_y, train_regime_key = concatenate_surfaces(cache["fine"], selected_ids); assert_complete_regime_rows(train_regime_key, fine_nodes, selected_ids); train_x = normalize(train_x, mean, std)
            fine_model = build_mlp(len(FEATURE_NAMES), 1, hidden=hidden, depth=depth)
            fine_model, history, fine_seconds = train_model(fine_model, train_x, train_y, val_x, val_y, model_spec, seed=seed, arm="fine_only", fraction=fraction, phase="fine_train")
            all_history.extend(history); trained[(seed, "fine_only", fraction)] = fine_model
            fine_val, fine_val_seconds = evaluate_model(fine_model, cache["fine"], val_ids, mean, std, model_spec, runtime["diagnostics"], seed=seed, arm="fine_only", fraction=fraction, split_name="validation")
            fine_test, fine_test_seconds = evaluate_model(fine_model, cache["fine"], test_ids, mean, std, model_spec, runtime["diagnostics"], seed=seed, arm="fine_only", fraction=fraction, split_name="test")
            metric_rows.extend(fine_val + fine_test); label_seconds = _fraction_label_seconds(surface_rows, selected_ids)
            arm_rows.append({"seed": seed, "arm": "fine_only", "fraction": fraction, "validation_price_rmse": _aggregate_price(fine_val), "coarse_pretrain_seconds": 0.0, "fine_tune_seconds": fine_seconds, "training_seconds": fine_seconds, "validation_eval_seconds": fine_val_seconds, "test_eval_seconds": fine_test_seconds, "fine_train_label_seconds": label_seconds, "interpolation_seconds": 0.0})
            multi_model = copy.deepcopy(coarse_model)
            multi_model, history, tune_seconds = train_model(multi_model, train_x, train_y, val_x, val_y, model_spec, seed=seed, arm="coarse_plus_few_fine", fraction=fraction, phase="fine_tune")
            all_history.extend(history); trained[(seed, "coarse_plus_few_fine", fraction)] = multi_model
            multi_val, multi_val_seconds = evaluate_model(multi_model, cache["fine"], val_ids, mean, std, model_spec, runtime["diagnostics"], seed=seed, arm="coarse_plus_few_fine", fraction=fraction, split_name="validation")
            multi_test, multi_test_seconds = evaluate_model(multi_model, cache["fine"], test_ids, mean, std, model_spec, runtime["diagnostics"], seed=seed, arm="coarse_plus_few_fine", fraction=fraction, split_name="test")
            metric_rows.extend(multi_val + multi_test); arm_rows.append({"seed": seed, "arm": "coarse_plus_few_fine", "fraction": fraction, "validation_price_rmse": _aggregate_price(multi_val), "coarse_pretrain_seconds": coarse_seconds, "fine_tune_seconds": tune_seconds, "training_seconds": coarse_seconds + tune_seconds, "validation_eval_seconds": multi_val_seconds, "test_eval_seconds": multi_test_seconds, "fine_train_label_seconds": label_seconds, "interpolation_seconds": 0.0})
    selected_by_seed: dict[int, dict[str, float | None]] = {}; selected_test_rows: list[dict[str, Any]] = []; cost_reductions: list[float] = []; selection_rows: list[dict[str, Any]] = []
    for seed in [int(x) for x in runtime["training_seeds"]]:
        fine_100 = next(row for row in arm_rows if row["seed"] == seed and row["arm"] == "fine_only" and float(row["fraction"]) == 1.0)
        target = matched_accuracy_target(float(fine_100["validation_price_rmse"]), float(runtime["matching"]["relative_price_tolerance"])); chosen: dict[str, float | None] = {}
        for arm in ("fine_only", "coarse_plus_few_fine"):
            scores = {float(row["fraction"]): float(row["validation_price_rmse"]) for row in arm_rows if row["seed"] == seed and row["arm"] == arm}; chosen[arm] = choose_smallest_matching_fraction(scores, target)
        selected_by_seed[seed] = chosen; baseline_fraction, candidate_fraction = chosen["fine_only"], chosen["coarse_plus_few_fine"]
        selection_rows.append({"seed": seed, "target_validation_price_rmse": target, "fine_only_fraction": baseline_fraction, "coarse_plus_few_fine_fraction": candidate_fraction})
        if baseline_fraction is None or candidate_fraction is None: continue
        baseline_cost = _fraction_label_seconds(surface_rows, subsets[baseline_fraction]); candidate_cost = _fraction_label_seconds(surface_rows, subsets[candidate_fraction]); cost_reductions.append(100.0 * (baseline_cost - candidate_cost) / max(baseline_cost, 1e-15))
        for row in metric_rows:
            if row["seed"] != seed or row["split"] != "test": continue
            if row["arm"] == "fine_only" and float(row["fraction"]) == baseline_fraction: selected_test_rows.append(row)
            elif row["arm"] == "coarse_plus_few_fine" and float(row["fraction"]) == candidate_fraction: selected_test_rows.append(row)
    matching_complete = len(cost_reductions) == len(runtime["training_seeds"])
    if matching_complete:
        cost_point, cost_low, cost_high = paired_bootstrap_mean_ci(cost_reductions, repetitions=int(runtime["bootstrap"]["repetitions"]), seed=int(runtime["bootstrap"]["seed"]) + 100)
        degradation_summary, degradation_rows = _metric_degradation_summary(selected_test_rows, bootstrap=runtime["bootstrap"], diagnostics=runtime["diagnostics"])
        degradation_upper = {metric: degradation_summary[f"degradation_ci_upper_pct_{metric}"] for metric in ALL_METRICS}
        selected_candidate_obstacle = max(float(r["obstacle_violation"]) for r in selected_test_rows if r["arm"] == "coarse_plus_few_fine")
        passed = candidate_passes(cost_reduction_ci_low_pct=cost_low, degradation_ci_upper_pct=degradation_upper, obstacle_violation=selected_candidate_obstacle, gates=runtime["gates"])
    else:
        cost_point = cost_low = cost_high = float("-inf"); degradation_summary = {}; degradation_rows = []; selected_candidate_obstacle = float("inf"); passed = False
    status = classify_status(evidence_mode, passed=passed); fixed_eval_seconds = _fixed_eval_label_seconds(surface_rows); total_seconds = float(time.perf_counter() - started_total)
    budget_exceeded = evidence_mode == "canonical" and total_seconds > float(runtime["budget_seconds"])
    if budget_exceeded: status = "resource_blocked"; passed = False
    timing_rows: list[dict[str, Any]] = []
    if matching_complete:
        first_test = cache["fine"][test_ids[0]]["features"]
        for seed, chosen in selected_by_seed.items():
            for arm in ("fine_only", "coarse_plus_few_fine"):
                fraction = chosen[arm]
                if fraction is None: continue
                mean_seconds, std_seconds = _time_inference(trained[(seed, arm, fraction)], first_test, mean, std, model_spec, runtime["timing"])
                timing_rows.append({"seed": seed, "arm": arm, "fraction": fraction, "inference_seconds_mean": mean_seconds, "inference_seconds_std": std_seconds})
    manifest = {"method_id": "M09", "phase": "screen", "evidence_mode": evidence_mode, "branch": _git("branch", "--show-current"), "implementation_sha": _git("rev-parse", "HEAD"),
                "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "torch": __import__("torch").__version__, "config_path": str(config_path), "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "feature_names": list(FEATURE_NAMES), "model_parameter_count": model_parameter_count(next(iter(trained.values()))), "training_seeds": list(runtime["training_seeds"]), "fine_fractions": list(runtime["fine_fractions"]),
                "split_regime_keys": {name: [str(r["id"]) for r in partitions[name]] for name in ("train", "validation", "test")}, "nested_fine_regime_subsets": {str(k): list(v) for k, v in subsets.items()},
                "coarse_grid": {"m": runtime["coarse_solver"]["m"], "n": runtime["coarse_solver"]["n"]}, "fine_grid": {"m": runtime["fine_solver"]["m"], "n": runtime["fine_solver"]["n"]},
                "blind_holdout_accessed": False, "scope_limitation": "Classical 1D American-put multi-fidelity falsification screen using complete matched CN/PSOR surfaces only; no local patch solves and no blind holdout access."}
    _write_json(out_dir / "config_manifest.json", manifest); _write_csv(out_dir / "surface_generation.csv", surface_rows); _write_csv(out_dir / "metrics.csv", metric_rows + arm_rows + selection_rows + degradation_rows + timing_rows); _write_csv(out_dir / "training_history.csv", all_history)
    decision = {"method_id": "M09", "status": status, "claim_status": "diagnostic_only" if status != "candidate_supported" else "candidate_supported", "blind_holdout_accessed": False, "matched_accuracy_complete": matching_complete,
                "selected_fractions_by_seed": {str(k): v for k, v in selected_by_seed.items()}, "fine_label_cost_reduction_pct": cost_point if np.isfinite(cost_point) else None, "fine_label_cost_reduction_ci_low_pct": cost_low if np.isfinite(cost_low) else None,
                "fine_label_cost_reduction_ci_high_pct": cost_high if np.isfinite(cost_high) else None, "shared_diagnostic_degradation": degradation_summary, "selected_candidate_obstacle_violation": selected_candidate_obstacle if np.isfinite(selected_candidate_obstacle) else None,
                "numerical_gate_passed": bool(passed), "coarse_label_seconds": float(sum(float(r["solve_seconds"]) for r in surface_rows if r["fidelity"] == "coarse")), "fine_eval_label_seconds": fixed_eval_seconds, "interpolation_seconds": 0.0, "total_seconds": total_seconds,
                "reason": ("Canonical M09 exceeded the frozen one-hour screen budget; no promotion claim is allowed." if budget_exceeded else "Multi-fidelity met matched held-out accuracy with the required fine-label cost CI and all shared diagnostic gates." if passed else "Multi-fidelity did not satisfy matched-accuracy fine-label cost and all shared diagnostic gates."),
                "limitations": ["The primary cost claim counts fine training-label generation only; validation/test fine-oracle generation is fixed audit overhead and reported separately.", "Coarse pretraining uses complete global coarse surfaces and fine tuning uses complete selected fine regimes; no boundary-local patch labels are used.", "Known payoff, terminal, and spatial boundary constraints are imposed identically on every learned arm before diagnostics.", "Interpolation cost is exactly zero because both fidelities train the same continuous-coordinate MLP; no coarse surface is interpolated into a fine label field."]}
    _write_json(out_dir / "decision.json", decision)
    artifact_paths = [out_dir / name for name in ("config_manifest.json", "surface_generation.csv", "metrics.csv", "training_history.csv", "decision.json")]; _write_json(out_dir / "artifact_hashes.json", {path.name: _sha256(path) for path in artifact_paths})
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/standalone/m09_multifidelity.json"); parser.add_argument("--mode", choices=("smoke", "canonical"), default=None); args = parser.parse_args()
    print(json.dumps(run(args.config, mode=args.mode), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
