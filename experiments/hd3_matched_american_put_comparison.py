"""Matched-contract high-dimensional American-put PINN comparison.

This post-hoc single-seed diagnostic reuses the frozen HD3 FC-HVI and
DeepSets-HVI architectures/objective, but maps each high-dimensional geometric
basket to the same reduced 1D American-put contract used by Ruixi's
``put_T100_s060_r005_q010`` regime.  Canonical CUDA execution is intentionally
left to the user's Windows RTX 4070 laptop.  The experiment exports compact
pointwise surfaces and free-boundary data at finalization so no retraining is
needed for the final plotting stage.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXPERIMENTS = ROOT / "experiments"
for p in (SRC, EXPERIMENTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hd_pipeline.runtime import (
    append_jsonl,
    atomic_json,
    capture_rng_state,
    load_checkpoint_verified,
    restore_rng_state,
    save_checkpoint_atomic,
    sha256_file,
    verified_completed_status,
)
from physics.high_dimensional_black_scholes import (
    black_scholes_vi_residual,
    effective_geometric_basket_parameters,
    fischer_burmeister,
    geometric_put_payoff,
)
from hd1_windows_high_dimensional_pinn_48h import (
    Oracle,
    build_oracle,
    environment_manifest,
    git_commit,
    oracle_labels,
)
from hd2_hvi_objective_stabilization import curriculum
from hd3_classical_vs_structured_high_dimensional_pinn import (
    _batch_candidates,
    _initialize_model,
    _optimizer_step,
    _parameter_count,
    _transform_physics_unit,
    _validation_price_rmse,
    evaluate_hd3_job,
)

CONFIG_PATH = ROOT / "configs" / "hd3_matched_american_put_comparison.json"
RESULT_DIR = ROOT / "results" / "hd3_matched_american_put_comparison"
RUNTIME_DIR = RESULT_DIR / "runtime"
ORACLE_DIR = RUNTIME_DIR / "oracles"
VIS_DIR = RESULT_DIR / "visualization"


def load_config() -> dict[str, Any]:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg["experiment_id"] != "hd3_matched_american_put_comparison":
        raise ValueError("wrong experiment config")
    if cfg["seed"] != 101 or cfg["canonical_dimensions"] != [10, 20, 50]:
        raise ValueError("matched experiment seed/dimensions are frozen")
    if cfg["models"] != ["FC_HVI", "DEEPSETS_HVI"]:
        raise ValueError("matched experiment models are frozen")
    if cfg["canonical_steps"] != 6000 or cfg["stress_steps"] != 4000:
        raise ValueError("matched experiment step budgets are frozen")
    if cfg.get("blind_holdout_access") is not False:
        raise ValueError("blind holdout access is forbidden")
    target = cfg["target_reduced_contract"]
    expected = {"option_type": "put", "K": 1.0, "T": 1.0, "r": 0.05, "q_g": 0.10, "sigma_g": 0.60}
    if target != expected or float(cfg["rho"]) != 0.80:
        raise ValueError("matched financial contract is frozen")
    return cfg


def config_hash() -> str:
    return sha256_file(CONFIG_PATH)


def parameter_mapping(d: int, cfg: dict[str, Any]) -> dict[str, float]:
    target = cfg["target_reduced_contract"]
    rho = float(cfg["rho"])
    sigma_g = float(target["sigma_g"])
    q_g = float(target["q_g"])
    factor = rho + (1.0 - rho) / int(d)
    sigma = sigma_g / math.sqrt(factor)
    q = q_g - 0.5 * (sigma * sigma - sigma_g * sigma_g)
    check_sigma_g, check_q_g = effective_geometric_basket_parameters(d, sigma, q, rho)
    if abs(check_sigma_g - sigma_g) > 1e-12 or abs(check_q_g - q_g) > 1e-12:
        raise RuntimeError(f"matched_parameter_assertion_failed:d={d}")
    return {
        "dimension": int(d),
        "rho": rho,
        "asset_sigma": sigma,
        "asset_q": q,
        "effective_sigma_g": check_sigma_g,
        "effective_q_g": check_q_g,
    }


def dimension_config(d: int, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg or load_config())
    mapping = parameter_mapping(d, cfg)
    target = cfg["target_reduced_contract"]
    cfg["contract"] = {
        "K": float(target["K"]),
        "T": float(target["T"]),
        "r": float(target["r"]),
        "q": float(mapping["asset_q"]),
        "sigma": float(mapping["asset_sigma"]),
        "rho": float(mapping["rho"]),
    }
    return cfg


def _oracle_path(d: int) -> Path:
    return ORACLE_DIR / f"d{int(d)}.npz"


def save_oracle(oracle: Oracle) -> None:
    path = _oracle_path(oracle.d)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        d=oracle.d,
        sigma_g=oracle.sigma_g,
        q_g=oracle.q_g,
        spots=oracle.spots,
        taus=oracle.taus,
        values=oracle.values,
        ug=oracle.ug,
        ugg=oracle.ugg,
        boundary=oracle.boundary,
        refinement_rmse=oracle.refinement_rmse,
        seconds=oracle.seconds,
    )


def load_oracle(d: int) -> Oracle:
    z = np.load(_oracle_path(d))
    return Oracle(
        int(z["d"]), float(z["sigma_g"]), float(z["q_g"]), z["spots"], z["taus"],
        z["values"], z["ug"], z["ugg"], z["boundary"],
        float(z["refinement_rmse"]), float(z["seconds"]),
    )


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _oracle_consistency(oracles: dict[int, Oracle]) -> dict[str, Any]:
    dims = sorted(oracles)
    rows = []
    passed = True
    reference = oracles[dims[0]]
    for d in dims[1:]:
        other = oracles[d]
        same_grid = np.array_equal(reference.spots, other.spots) and np.array_equal(reference.taus, other.taus)
        if not same_grid:
            passed = False
            rows.append({"pair": f"d{dims[0]}_vs_d{d}", "same_grid": False})
            continue
        diff = np.asarray(reference.values) - np.asarray(other.values)
        max_abs = float(np.max(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff * diff)))
        pair_pass = max_abs <= 1e-10 and rmse <= 1e-11
        passed = passed and pair_pass
        rows.append({
            "pair": f"d{dims[0]}_vs_d{d}", "same_grid": True,
            "max_abs_difference": max_abs, "rmse_difference": rmse, "passed": pair_pass,
        })
    return {"passed": passed, "comparisons": rows}


def crr_american_put(spot: float, tau: float, *, k: float, r: float, q: float, sigma: float, steps: int) -> float:
    """Cox-Ross-Rubinstein American put value for one initial spot."""
    if tau <= 0.0:
        return max(k - spot, 0.0)
    dt = tau / int(steps)
    u = math.exp(sigma * math.sqrt(dt))
    down = 1.0 / u
    growth = math.exp((r - q) * dt)
    p = (growth - down) / (u - down)
    if not 0.0 <= p <= 1.0:
        raise RuntimeError(f"invalid_crr_probability:{p}")
    disc = math.exp(-r * dt)
    j = np.arange(steps + 1, dtype=float)
    stock = spot * (u ** j) * (down ** (steps - j))
    values = np.maximum(k - stock, 0.0)
    for n in range(steps - 1, -1, -1):
        values = disc * (p * values[1:n + 2] + (1.0 - p) * values[:n + 1])
        jn = np.arange(n + 1, dtype=float)
        stock_n = spot * (u ** jn) * (down ** (n - jn))
        values = np.maximum(values, np.maximum(k - stock_n, 0.0))
    return float(values[0])


def run_binomial(cfg: dict[str, Any]) -> dict[str, Any]:
    target = cfg["target_reduced_contract"]
    k, r, q, sigma = float(target["K"]), float(target["r"]), float(target["q_g"]), float(target["sigma_g"])
    steps = [int(v) for v in cfg["binomial_steps"]]
    check_spots = np.linspace(0.6, 1.4, 9)
    convergence_rows = []
    previous: np.ndarray | None = None
    for n in steps:
        vals = np.asarray([crr_american_put(float(s), 1.0, k=k, r=r, q=q, sigma=sigma, steps=n) for s in check_spots])
        payoff = np.maximum(k - check_spots, 0.0)
        if np.any(vals + 1e-12 < payoff):
            raise RuntimeError("binomial_obstacle_violation")
        delta = None if previous is None else vals - previous
        convergence_rows.append({
            "steps": n,
            "max_abs_change_from_previous": None if delta is None else float(np.max(np.abs(delta))),
            "rmse_change_from_previous": None if delta is None else float(np.sqrt(np.mean(delta * delta))),
        })
        previous = vals
    final_steps = steps[-2] if len(steps) >= 2 else steps[-1]
    anchor_spots = np.linspace(
        float(cfg["visualization"]["slice_s_min"]), float(cfg["visualization"]["slice_s_max"]), 101
    )
    dense_spots = np.linspace(
        float(cfg["visualization"]["slice_s_min"]), float(cfg["visualization"]["slice_s_max"]),
        int(cfg["visualization"]["slice_points"]),
    )
    for tau in cfg["visualization"]["slice_taus"]:
        anchor_values = np.asarray([
            crr_american_put(float(s), float(tau), k=k, r=r, q=q, sigma=sigma, steps=final_steps)
            for s in anchor_spots
        ])
        dense_values = np.interp(dense_spots, anchor_spots, anchor_values)
        tag = {1.0: "1", 0.5: "05", 0.1: "01"}[float(tau)]
        _write_csv(VIS_DIR / "binomial" / f"price_slice_tau_{tag}.csv", [
            {"S_over_K": float(s), "tau_over_T": float(tau), "V_over_K": float(v),
             "payoff_over_K": float(max(1.0 - s, 0.0)), "tree_steps": final_steps,
             "interpolated_from_anchor_grid": True}
            for s, v in zip(dense_spots, dense_values)
        ])
    payload = {
        "contract": target, "check_spots": check_spots.tolist(), "convergence": convergence_rows,
        "final_curve_steps": final_steps, "final_curve_anchor_points": len(anchor_spots),
        "final_curve_output_points": len(dense_spots),
    }
    atomic_json(RESULT_DIR / "binomial_convergence.json", payload)
    return payload


def _job_id(d: int, model_id: str, run_class: str) -> str:
    return f"{run_class}_d{int(d)}_{model_id}"


def _job_dir(d: int, model_id: str, run_class: str) -> Path:
    return RUNTIME_DIR / _job_id(d, model_id, run_class)


def _status_path(d: int, model_id: str, run_class: str) -> Path:
    return _job_dir(d, model_id, run_class) / "status.json"


def _checkpoint_path(d: int, model_id: str, run_class: str) -> Path:
    return _job_dir(d, model_id, run_class) / "latest_checkpoint.pt"


def _fixed_labeled_data(d: int, cfg: dict[str, Any], oracle: Oracle, split: str, n: int, device: torch.device):
    from physics.hd3_high_dimensional_sampling import sample_transverse_states
    m_bounds = tuple(float(v) for v in cfg["aggregate_log_moneyness"])
    tau_bounds = tuple(float(v) for v in cfg["tau"])
    x, tau, _ = sample_transverse_states(
        n, d, seed=int(cfg["sobol_seeds"][split]) + d,
        m_bounds=m_bounds, tau_bounds=tau_bounds,
        transverse_scale=float(cfg["transverse_scale"]), dtype=torch.float64,
    )
    labels = oracle_labels(oracle, x, tau, cfg)["price"]
    return x.to(device=device, dtype=torch.float32), tau.to(device=device, dtype=torch.float32), torch.tensor(labels, device=device, dtype=torch.float32)


def _physics_loss(model: torch.nn.Module, x: torch.Tensor, tau: torch.Tensor, d: int, cfg: dict[str, Any], epsilon: float) -> torch.Tensor:
    value = model(x, tau)
    c = cfg["contract"]
    residual, _ = black_scholes_vi_residual(
        value, x, tau, r=float(c["r"]), q=float(c["q"]), sigma=float(c["sigma"]), rho=float(c["rho"]),
        hessian_mode="hutchinson", probe_count=int(cfg["hutchinson_probe_count"]),
        probe_seed=int(cfg["hutchinson_probe_seed"]),
    )
    return fischer_burmeister(value - geometric_put_payoff(x), residual, epsilon).square().mean()


def train_job(d: int, model_id: str, *, run_class: str = "canonical", max_steps_override: int | None = None) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required:matched HD3 training")
    root_cfg = load_config()
    if run_class == "canonical" and d not in root_cfg["canonical_dimensions"]:
        raise ValueError("dimension outside canonical scope")
    if run_class == "stress" and d not in root_cfg["stress_dimensions"]:
        raise ValueError("dimension outside stress scope")
    if model_id not in root_cfg["models"]:
        raise ValueError("unknown model")
    cfg = dimension_config(d, root_cfg)
    objective = dict(cfg["objective"])
    device = torch.device("cuda:0")
    target_steps = int(cfg["canonical_steps"] if run_class == "canonical" else cfg["stress_steps"])
    target_run = min(target_steps, int(max_steps_override)) if max_steps_override else target_steps
    warm_steps = int(round(target_steps * float(objective["warm_fraction"])))
    oracle = load_oracle(d)
    job_id = _job_id(d, model_id, run_class)
    job_dir = _job_dir(d, model_id, run_class)
    job_dir.mkdir(parents=True, exist_ok=True)
    status_path = _status_path(d, model_id, run_class)
    checkpoint_path = _checkpoint_path(d, model_id, run_class)
    if max_steps_override is None and verified_completed_status(status_path, implementation_sha=git_commit(), config_hash=config_hash()):
        return json.loads((job_dir / "validation_metrics.json").read_text(encoding="utf-8"))

    x_train, tau_train, y_train = _fixed_labeled_data(d, cfg, oracle, "train", int(cfg["supervised_train_points"]), device)
    x_val, tau_val, y_val = _fixed_labeled_data(d, cfg, oracle, "validation", int(cfg["validation_points"]), device)
    batch_candidates = [int(cfg["initial_batch_size"]), *[int(v) for v in cfg["oom_batch_fallbacks"]]]
    implementation_sha, cfg_hash = git_commit(), config_hash()

    for batch_size in batch_candidates:
        model = _initialize_model(model_id, d, cfg, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
        batch_generator = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]) + 1000 * d)
        physics_engine = torch.quasirandom.SobolEngine(d + 2, scramble=True, seed=int(cfg["sobol_seeds"]["physics"]) + d)
        completed_step, physics_consumed, elapsed_before = 0, 0, 0.0
        if checkpoint_path.exists():
            checkpoint = load_checkpoint_verified(checkpoint_path, implementation_sha=implementation_sha, config_hash=cfg_hash, job_id=job_id)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            batch_generator.set_state(checkpoint["batch_generator_state"])
            completed_step = int(checkpoint["completed_step"])
            physics_consumed = int(checkpoint["physics_consumed"])
            elapsed_before = float(checkpoint.get("elapsed_training_seconds", 0.0))
            restore_rng_state(checkpoint["rng_state"])
            saved_batch = int(checkpoint["batch_size"])
            if saved_batch != batch_size:
                if batch_size != saved_batch:
                    continue
        if physics_consumed:
            physics_engine.fast_forward(physics_consumed)
        atomic_json(status_path, {"status": "running", "implementation_sha": implementation_sha, "config_hash": cfg_hash,
                                  "job_id": job_id, "dimension": d, "model_id": model_id, "run_class": run_class,
                                  "completed_step": completed_step, "target_step": target_steps, "batch_size": batch_size})
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            for step in range(completed_step, target_run):
                weight, epsilon = curriculum(
                    step, target_steps, warm_steps, float(objective["final_physics_weight"]),
                    epsilon_initial=float(cfg["fb_epsilon_initial"]), epsilon_final=float(cfg["fb_epsilon_final"]),
                )
                index = torch.randint(0, len(x_train), (batch_size,), generator=batch_generator).to(device)
                sup_loss = (model(x_train[index], tau_train[index]) - y_train[index]).square().mean()
                phys_loss = torch.zeros((), device=device)
                if step >= warm_steps and weight > 0.0:
                    unit = physics_engine.draw(batch_size).to(device=device, dtype=torch.float32)
                    physics_consumed += batch_size
                    xp, tp = _transform_physics_unit(unit, d, cfg)
                    phys_loss = _physics_loss(model, xp, tp, d, cfg, epsilon)
                total, _ = _optimizer_step(
                    model=model, optimizer=optimizer, supervised_loss=sup_loss, physics_loss=phys_loss,
                    weight=weight, mode="fixed", objective=objective, relobralo=None,
                )
                completed_step = step + 1
                elapsed = elapsed_before + time.perf_counter() - started
                peak = torch.cuda.max_memory_allocated(device) / 1024**3
                if peak >= float(cfg["max_cuda_allocation_gib"]):
                    raise RuntimeError(f"cuda_memory_gate_failed:{peak:.4f}GiB")
                if completed_step == 1 or completed_step % int(cfg["train_log_interval"]) == 0 or completed_step == target_run:
                    append_jsonl(job_dir / "training_history.jsonl", {
                        "dimension": d, "model_id": model_id, "run_class": run_class, "step": completed_step,
                        "supervised_mse": float(sup_loss.detach().cpu()), "fb_loss": float(phys_loss.detach().cpu()),
                        "total_loss": float(total.detach().cpu()), "fb_weight": float(weight), "epsilon": float(epsilon),
                        "batch_size": batch_size, "elapsed_seconds": elapsed, "peak_cuda_gib": peak,
                    })
                if completed_step % int(cfg["validation_interval"]) == 0:
                    append_jsonl(job_dir / "validation_history.jsonl", {
                        "dimension": d, "model_id": model_id, "step": completed_step,
                        "price_rmse": _validation_price_rmse(model, x_val, tau_val, y_val), "elapsed_seconds": elapsed,
                    })
                if completed_step % int(cfg["checkpoint_interval"]) == 0 or completed_step == target_run:
                    save_checkpoint_atomic(checkpoint_path, {
                        "implementation_sha": implementation_sha, "config_hash": cfg_hash, "job_id": job_id,
                        "dimension": d, "model_id": model_id, "run_class": run_class,
                        "completed_step": completed_step, "target_steps": target_steps,
                        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                        "batch_generator_state": batch_generator.get_state(), "physics_consumed": physics_consumed,
                        "batch_size": batch_size, "elapsed_training_seconds": elapsed,
                        "rng_state": capture_rng_state(), "parameter_mapping": parameter_mapping(d, root_cfg),
                    })
            elapsed_total = elapsed_before + time.perf_counter() - started
            if target_run < target_steps:
                atomic_json(status_path, {"status": "interrupted_test_hook", "implementation_sha": implementation_sha,
                                          "config_hash": cfg_hash, "job_id": job_id, "completed_step": completed_step,
                                          "target_step": target_steps, "checkpoint": str(checkpoint_path)})
                return {"status": "interrupted_test_hook", "completed_step": completed_step}
            metrics = evaluate_hd3_job(
                d=d, model_id=model_id, run_class=run_class, model=model, cfg=cfg, oracle=oracle,
                split="validation", n=int(cfg["validation_points"]), training_seconds=elapsed_total,
                batch_size=batch_size, probe_count=int(cfg["hutchinson_probe_count"]),
            )
            atomic_json(job_dir / "validation_metrics.json", metrics)
            atomic_json(status_path, {"status": "completed", "implementation_sha": implementation_sha,
                                      "config_hash": cfg_hash, "job_id": job_id, "dimension": d, "model_id": model_id,
                                      "run_class": run_class, "completed_step": target_steps, "target_step": target_steps,
                                      "batch_size": batch_size, "checkpoint_sha256": sha256_file(checkpoint_path)})
            return metrics
        except torch.cuda.OutOfMemoryError:
            append_jsonl(job_dir / "oom_attempts.jsonl", {"batch_size": batch_size, "error": "cuda_oom", "time": time.time()})
            torch.cuda.empty_cache()
            if checkpoint_path.exists():
                saved = load_checkpoint_verified(checkpoint_path, implementation_sha=implementation_sha, config_hash=cfg_hash, job_id=job_id)
                saved_batch = int(saved["batch_size"])
                if saved_batch != batch_size:
                    raise RuntimeError("checkpoint_batch_mismatch_after_oom")
                # Resume with the same scientific stream; batch-size changes are only
                # permitted when no progress has yet been checkpointed.
                if int(saved["completed_step"]) > 0:
                    raise RuntimeError("oom_after_progress:resume_same_batch_or_reduce_before_run")
                checkpoint_path.unlink()
            continue
        except KeyboardInterrupt:
            atomic_json(status_path, {"status": "interrupted", "implementation_sha": implementation_sha,
                                      "config_hash": cfg_hash, "job_id": job_id, "completed_step": completed_step,
                                      "target_step": target_steps, "checkpoint": str(checkpoint_path)})
            raise
    raise RuntimeError("oom_blocked:all_predeclared_batches_exhausted")


def _load_completed_model(d: int, model_id: str, run_class: str, cfg: dict[str, Any]):
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required:finalization")
    checkpoint = load_checkpoint_verified(
        _checkpoint_path(d, model_id, run_class), implementation_sha=git_commit(), config_hash=config_hash(),
        job_id=_job_id(d, model_id, run_class),
    )
    target = int(cfg["canonical_steps"] if run_class == "canonical" else cfg["stress_steps"])
    if int(checkpoint["completed_step"]) != target:
        raise RuntimeError("incomplete_training_checkpoint")
    model = _initialize_model(model_id, d, cfg, torch.device("cuda:0"))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _predict_diagonal(model: torch.nn.Module, d: int, spots: np.ndarray, taus: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    out = np.empty(len(spots), dtype=float)
    with torch.no_grad():
        for start in range(0, len(spots), 1024):
            s = torch.tensor(spots[start:start + 1024], device=device, dtype=torch.float32)
            x = torch.log(s).reshape(-1, 1).repeat(1, d)
            t = torch.tensor(taus[start:start + 1024], device=device, dtype=torch.float32)
            out[start:start + len(s)] = model(x, t).detach().cpu().numpy().reshape(-1)
    return out


def _export_surface_arrays(path: Path, spots: np.ndarray, taus: np.ndarray, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payoff = np.maximum(1.0 - spots, 0.0)
    np.savez_compressed(path, S_over_K=spots, tau_over_T=taus, V_over_K=values,
                        payoff_over_K=payoff, premium_over_K=values - payoff)


def export_visualization(d: int, model_id: str, model: torch.nn.Module, oracle: Oracle, cfg: dict[str, Any]) -> dict[str, Any]:
    vis = cfg["visualization"]
    out_dir = VIS_DIR / f"d{d}" / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    slice_spots = np.linspace(float(vis["slice_s_min"]), float(vis["slice_s_max"]), int(vis["slice_points"]))
    for tau in vis["slice_taus"]:
        taus = np.full_like(slice_spots, float(tau))
        values = _predict_diagonal(model, d, slice_spots, taus)
        tag = {1.0: "1", 0.5: "05", 0.1: "01"}[float(tau)]
        _write_csv(out_dir / f"price_slice_tau_{tag}.csv", [
            {"S_over_K": float(s), "tau_over_T": float(tau), "V_over_K": float(v),
             "payoff_over_K": float(max(1.0 - s, 0.0)), "premium_over_K": float(v - max(1.0 - s, 0.0))}
            for s, v in zip(slice_spots, values)
        ])
    surface_spots_1d = np.linspace(float(vis["surface_s_min"]), float(vis["surface_s_max"]), int(vis["surface_s_points"]))
    surface_taus_1d = np.linspace(float(vis["surface_tau_min"]), float(vis["surface_tau_max"]), int(vis["surface_tau_points"]))
    ss, tt = np.meshgrid(surface_spots_1d, surface_taus_1d)
    vv = _predict_diagonal(model, d, ss.ravel(), tt.ravel())
    _export_surface_arrays(out_dir / "surface.npz", ss.ravel(), tt.ravel(), vv)

    boundary_spots = np.linspace(float(vis["boundary_s_min"]), float(vis["boundary_s_max"]), int(vis["boundary_s_points"]))
    boundary_taus = np.linspace(float(vis["boundary_tau_min"]), float(vis["boundary_tau_max"]), int(vis["boundary_tau_points"]))
    tolerance = float(vis["boundary_contact_tolerance"])
    boundary_rows = []
    predicted = []
    truth = []
    for tau in boundary_taus:
        vals = _predict_diagonal(model, d, boundary_spots, np.full_like(boundary_spots, tau))
        payoff = np.maximum(1.0 - boundary_spots, 0.0)
        ex = np.flatnonzero((boundary_spots <= 1.0) & ((vals - payoff) <= tolerance))
        found = bool(ex.size)
        star = float(boundary_spots[ex[-1]]) if found else float("nan")
        true_star = float(np.interp(tau, oracle.taus, oracle.boundary))
        predicted.append(star)
        truth.append(true_star)
        boundary_rows.append({"tau_over_T": float(tau), "S_star_over_K": star, "boundary_found": found,
                              "boundary_method": "diagonal_obstacle_contact", "boundary_tolerance": tolerance,
                              "oracle_S_star_over_K": true_star})
    _write_csv(out_dir / "free_boundary.csv", boundary_rows)
    pred = np.asarray(predicted); true = np.asarray(truth); valid = np.isfinite(pred)
    boundary_metrics = {
        "boundary_found_fraction": float(np.mean(valid)),
        "boundary_rmse_found": float(np.sqrt(np.mean((pred[valid] - true[valid]) ** 2))) if np.any(valid) else None,
        "boundary_mae_found": float(np.mean(np.abs(pred[valid] - true[valid]))) if np.any(valid) else None,
        "boundary_max_abs_found": float(np.max(np.abs(pred[valid] - true[valid]))) if np.any(valid) else None,
    }
    surface = np.load(out_dir / "surface.npz")
    premium = np.asarray(surface["premium_over_K"])
    metadata = {
        "model": model_id, "dimension": d, "seed": int(cfg["seed"]), "diagnostic_only": True,
        "single_seed": True, "posthoc_matched_contract": True, "random_seed_robustness": False,
        "target_reduced_contract": cfg["target_reduced_contract"], "asset_contract": cfg["contract"],
        "parameter_mapping": parameter_mapping(d, load_config()),
        "slice_definition": "S_1=...=S_d=S; therefore geometric mean G=S",
        "normalization": "K=1; exported S/K, tau/T, V/K",
        "source_git_commit": git_commit(), "source_checkpoint_sha256": sha256_file(_checkpoint_path(d, model_id, "canonical" if d in load_config()["canonical_dimensions"] else "stress")),
        "surface_points": int(len(surface["V_over_K"])),
        "max_obstacle_violation_on_surface": float(np.max(np.maximum(-premium, 0.0))),
        **boundary_metrics,
    }
    atomic_json(out_dir / "metadata.json", metadata)
    hashes = {p.name: sha256_file(p) for p in sorted(out_dir.iterdir()) if p.is_file() and p.name != "checksums.json"}
    atomic_json(out_dir / "checksums.json", hashes)
    return metadata


def export_oracle_visualization(d: int, oracle: Oracle, cfg: dict[str, Any]) -> None:
    from hd1_windows_high_dimensional_pinn_48h import interp2
    vis = cfg["visualization"]
    out_dir = VIS_DIR / f"d{d}" / "oracle"
    out_dir.mkdir(parents=True, exist_ok=True)
    slice_spots = np.linspace(float(vis["slice_s_min"]), float(vis["slice_s_max"]), int(vis["slice_points"]))
    for tau in vis["slice_taus"]:
        taus = np.full_like(slice_spots, float(tau))
        values = interp2(oracle.spots, oracle.taus, oracle.values, slice_spots, taus)
        tag = {1.0: "1", 0.5: "05", 0.1: "01"}[float(tau)]
        _write_csv(out_dir / f"price_slice_tau_{tag}.csv", [
            {"S_over_K": float(s), "tau_over_T": float(tau), "V_over_K": float(v),
             "payoff_over_K": float(max(1.0 - s, 0.0)), "premium_over_K": float(v - max(1.0 - s, 0.0))}
            for s, v in zip(slice_spots, values)
        ])
    s1 = np.linspace(float(vis["surface_s_min"]), float(vis["surface_s_max"]), int(vis["surface_s_points"]))
    t1 = np.linspace(float(vis["surface_tau_min"]), float(vis["surface_tau_max"]), int(vis["surface_tau_points"]))
    ss, tt = np.meshgrid(s1, t1)
    values = interp2(oracle.spots, oracle.taus, oracle.values, ss.ravel(), tt.ravel())
    _export_surface_arrays(out_dir / "surface.npz", ss.ravel(), tt.ravel(), values)
    bt = np.linspace(float(vis["boundary_tau_min"]), float(vis["boundary_tau_max"]), int(vis["boundary_tau_points"]))
    _write_csv(out_dir / "free_boundary.csv", [
        {"tau_over_T": float(t), "S_star_over_K": float(np.interp(t, oracle.taus, oracle.boundary)), "boundary_found": True,
         "boundary_method": "CN_PSOR_oracle_contact"} for t in bt
    ])
    atomic_json(out_dir / "metadata.json", {
        "model": "CN_PSOR_reduced_oracle", "dimension": d, "effective_sigma_g": oracle.sigma_g,
        "effective_q_g": oracle.q_g, "refinement_rmse": oracle.refinement_rmse,
        "source_git_commit": git_commit(), "oracle_runtime_sha256": sha256_file(_oracle_path(d)),
    })
    atomic_json(out_dir / "checksums.json", {p.name: sha256_file(p) for p in sorted(out_dir.iterdir()) if p.is_file() and p.name != "checksums.json"})


def run_preflight() -> dict[str, Any]:
    cfg = load_config()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = environment_manifest()
    atomic_json(RESULT_DIR / "environment_manifest.json", manifest)
    atomic_json(RESULT_DIR / "config_manifest.json", {"config": cfg, "config_sha256": config_hash(), "git_commit": git_commit()})
    mappings = [parameter_mapping(int(d), cfg) for d in [*cfg["canonical_dimensions"], *cfg["stress_dimensions"]]]
    _write_csv(RESULT_DIR / "parameter_mapping.csv", mappings)
    if not manifest["cuda_available"]:
        raise RuntimeError("cuda_required:matched preflight")
    if int(manifest.get("gpu_total_memory_bytes") or 0) < 7 * 1024**3:
        raise RuntimeError("cuda_memory_blocked")
    oracles: dict[int, Oracle] = {}
    for d in cfg["canonical_dimensions"]:
        dim_cfg = dimension_config(int(d), cfg)
        oracle = build_oracle(int(d), dim_cfg)
        save_oracle(oracle)
        oracles[int(d)] = oracle
    consistency = _oracle_consistency(oracles)
    atomic_json(RESULT_DIR / "oracle_consistency.json", consistency)
    if not consistency["passed"]:
        raise RuntimeError("oracle_cross_dimension_consistency_failed")
    binomial = run_binomial(cfg)
    payload = {"passed": True, "mappings": mappings, "oracle_consistency": consistency, "binomial": binomial}
    atomic_json(RESULT_DIR / "preflight.json", payload)
    return payload


def run_canonical() -> list[dict[str, Any]]:
    cfg = load_config()
    return [train_job(int(d), str(model), run_class="canonical") for d in cfg["canonical_dimensions"] for model in cfg["models"]]


def run_stress() -> list[dict[str, Any]]:
    cfg = load_config()
    # Build stress oracle lazily so canonical preflight stays minimal.
    for d in cfg["stress_dimensions"]:
        if not _oracle_path(int(d)).exists():
            save_oracle(build_oracle(int(d), dimension_config(int(d), cfg)))
    return [train_job(int(d), str(model), run_class="stress") for d in cfg["stress_dimensions"] for model in cfg["models"]]


def run_finalize(*, include_stress: bool = False) -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required:matched finalization")
    cfg = load_config()
    rows: list[dict[str, Any]] = []
    classes = [("canonical", cfg["canonical_dimensions"])]
    if include_stress:
        classes.append(("stress", cfg["stress_dimensions"]))
    for run_class, dimensions in classes:
        for d in dimensions:
            oracle = load_oracle(int(d))
            dim_cfg = dimension_config(int(d), cfg)
            export_oracle_visualization(int(d), oracle, dim_cfg)
            for model_id in cfg["models"]:
                status = _status_path(int(d), str(model_id), run_class)
                if not status.exists() or json.loads(status.read_text(encoding="utf-8")).get("status") != "completed":
                    if run_class == "stress":
                        continue
                    raise RuntimeError(f"missing_completed_job:d{d}:{model_id}")
                model, checkpoint = _load_completed_model(int(d), str(model_id), run_class, dim_cfg)
                row = evaluate_hd3_job(
                    d=int(d), model_id=str(model_id), run_class=run_class, model=model, cfg=dim_cfg, oracle=oracle,
                    split="test", n=int(dim_cfg["test_points"]), training_seconds=float(checkpoint["elapsed_training_seconds"]),
                    batch_size=int(checkpoint["batch_size"]), probe_count=int(dim_cfg["hutchinson_probe_count"]),
                )
                row.update(parameter_mapping(int(d), cfg))
                rows.append(row)
                export_visualization(int(d), str(model_id), model, oracle, dim_cfg)
    _write_csv(RESULT_DIR / "run_metrics.csv", rows)
    _write_csv(RESULT_DIR / "structural_diagnostics.csv", [
        {k: row[k] for k in ("dimension", "model_id", "run_class", "permutation_consistency_rmse", "transverse_invariance_rmse")}
        for row in rows
    ])
    compute_rows = [{k: row[k] for k in ("dimension", "model_id", "run_class", "training_seconds", "parameter_count", "batch_size", "peak_cuda_gib", "inference_seconds", "evaluation_seconds")} for row in rows]
    _write_csv(RESULT_DIR / "compute_accounting.csv", compute_rows)
    atomic_json(RESULT_DIR / "compute_accounting.json", {
        "single_seed": True, "seed": 101, "gpu": environment_manifest().get("gpu_name"),
        "jobs": compute_rows, "total_gpu_training_seconds_sum": float(sum(float(r["training_seconds"]) for r in rows)),
        "scope_note": "Engineering compute accounting for this matched pilot; do not interpret raw ratios as pure algorithmic speedups against Ruixi's 67 regime-specific trainings."
    })
    summary = []
    for d in sorted({int(r["dimension"]) for r in rows}):
        for model_id in cfg["models"]:
            match = next((r for r in rows if int(r["dimension"]) == d and r["model_id"] == model_id), None)
            if match:
                summary.append({"dimension": d, "model_id": model_id, "normalized_price_rmse": match["normalized_price_rmse"],
                                "delta_rmse": match["delta_rmse"], "gamma_contraction_rmse": match["gamma_contraction_rmse"],
                                "trusted_pde_p95": match["trusted_pde_p95"], "boundary_rmse": match["boundary_rmse"],
                                "training_seconds": match["training_seconds"], "parameter_count": match["parameter_count"]})
    _write_csv(RESULT_DIR / "dimension_summary.csv", summary)
    atomic_json(RESULT_DIR / "final_status.json", {
        "status": "completed", "diagnostic_only": True, "single_seed": True, "seed": 101,
        "posthoc_matched_contract": True, "random_seed_robustness": False, "candidate_supported": False,
        "canonical_dimensions": cfg["canonical_dimensions"], "stress_included": include_stress,
        "models": cfg["models"], "rows": len(rows), "target_reduced_contract": cfg["target_reduced_contract"],
    })
    hashes = {}
    for path in RESULT_DIR.rglob("*"):
        if path.is_file() and RUNTIME_DIR not in path.parents and path.name != "artifact_hashes.json":
            hashes[str(path.relative_to(RESULT_DIR))] = sha256_file(path)
    atomic_json(RESULT_DIR / "artifact_hashes.json", {"git_commit": git_commit(), "sha256": hashes})
    return rows


def status_snapshot() -> dict[str, Any]:
    cfg = load_config()
    jobs = []
    for run_class, dimensions in (("canonical", cfg["canonical_dimensions"]), ("stress", cfg["stress_dimensions"])):
        for d in dimensions:
            for model in cfg["models"]:
                path = _status_path(int(d), str(model), run_class)
                jobs.append(json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
                    "dimension": int(d), "model_id": str(model), "run_class": run_class, "status": "not_started"
                })
    return {
        "experiment": cfg["experiment_id"], "git_commit": git_commit(),
        "preflight": json.loads((RESULT_DIR / "preflight.json").read_text(encoding="utf-8")) if (RESULT_DIR / "preflight.json").exists() else {"status": "not_run"},
        "jobs": jobs,
        "final_status": json.loads((RESULT_DIR / "final_status.json").read_text(encoding="utf-8")) if (RESULT_DIR / "final_status.json").exists() else {"status": "not_finalized"},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["preflight", "canonical", "stress", "finalize", "status"], required=True)
    parser.add_argument("--dimension", type=int)
    parser.add_argument("--model", choices=["FC_HVI", "DEEPSETS_HVI"])
    parser.add_argument("--max-steps-override", type=int)
    parser.add_argument("--include-stress", action="store_true")
    args = parser.parse_args(argv)
    if args.phase == "preflight":
        result: Any = run_preflight()
    elif args.phase in {"canonical", "stress"}:
        run_class = args.phase
        if args.dimension is not None or args.model is not None:
            if args.dimension is None or args.model is None:
                raise SystemExit("--dimension and --model must be supplied together")
            result = train_job(args.dimension, args.model, run_class=run_class, max_steps_override=args.max_steps_override)
        else:
            result = run_canonical() if run_class == "canonical" else run_stress()
    elif args.phase == "finalize":
        result = run_finalize(include_stress=args.include_stress)
    else:
        result = status_snapshot()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
