"""HD2 HVI objective-conflict diagnosis and stabilization.

This module deliberately keeps the HD1 DeepSets architecture and hybrid
supervised-plus-Fischer--Burmeister objective.  It adds durable resume,
objective-gradient diagnostics, a d=5 exact-vs-Hutchinson gradient audit, and
validation-only deterministic candidate selection.  Test evaluation is gated
on a frozen selection decision.

Canonical CUDA training is intentionally not executed by CI; it runs only on
the user's target Windows GPU.
"""
from __future__ import annotations

import argparse
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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hd_pipeline.gradient_diagnostics import (
    flatten_grads,
    layerwise_gradient_diagnostics,
    objective_gradient_diagnostics,
)
from hd_pipeline.runtime import (
    append_jsonl,
    atomic_json,
    canonical_json_hash,
    capture_rng_state,
    load_checkpoint_verified,
    read_jsonl_recover,
    restore_rng_state,
    save_checkpoint_atomic,
    sha256_file,
    verified_completed_status,
)
from models.high_dimensional_pinn import initialize_model, parameter_count
from physics.high_dimensional_black_scholes import (
    black_scholes_vi_residual,
    covariance_matrix,
    fischer_burmeister,
    full_hessian_contraction,
    geometric_put_payoff,
    hutchinson_hessian_contraction,
)
from hd1_windows_high_dimensional_pinn_48h import (
    Oracle,
    build_oracle,
    environment_manifest,
    f1_score,
    git_commit,
    interp2,
    oracle_labels,
    peak_cpu_memory_bytes,
    sobol_points,
    validate_hutchinson,
)

CONFIG_PATH = ROOT / "configs" / "hd2_hvi_objective_stabilization.json"
HD1_CONFIG_PATH = ROOT / "configs" / "hd1_windows_high_dimensional_pinn_48h.json"
RESULT_DIR = ROOT / "results" / "hd2_hvi_objective_stabilization"
RUNTIME_DIR = RESULT_DIR / "runtime"
ORACLE_DIR = RUNTIME_DIR / "oracles"
SELECTION_PATH = RESULT_DIR / "selection_decision.json"


def load_config() -> dict[str, Any]:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg["seed"] != 101 or cfg["dimensions"] != [2, 5]:
        raise ValueError("HD2 seed/dimensions are frozen")
    if cfg["total_steps"] != 10000:
        raise ValueError("HD2 total step budget is frozen")
    if cfg["hutchinson_probe_count"] != 2:
        raise ValueError("HD2 default Hutchinson probe count is frozen")
    if cfg.get("blind_holdout_access") is not False:
        raise ValueError("HD2 must not access the campaign blind holdout")
    return cfg


def config_hash() -> str:
    return sha256_file(CONFIG_PATH)


def curriculum(
    step: int,
    total_steps: int,
    warm_start_steps: int,
    final_weight: float,
    *,
    epsilon_initial: float = 1e-2,
    epsilon_final: float = 1e-4,
) -> tuple[float, float]:
    """Historical HD1 schedule with a candidate-specific warm start/envelope."""
    if not 0 <= step < total_steps:
        raise ValueError("invalid training step")
    if warm_start_steps < 0 or warm_start_steps > total_steps:
        raise ValueError("invalid warm start")
    if final_weight < 0:
        raise ValueError("final physics weight must be nonnegative")
    if step < warm_start_steps or final_weight == 0.0 or warm_start_steps == total_steps:
        return 0.0, float(epsilon_initial)
    denominator = max(total_steps - warm_start_steps - 1, 1)
    alpha = min(max((step - warm_start_steps) / denominator, 0.0), 1.0)
    weight = float(final_weight) * alpha
    log_eps = math.log10(epsilon_initial) + alpha * (
        math.log10(epsilon_final) - math.log10(epsilon_initial)
    )
    return weight, 10.0**log_eps


def persistent_conflict(
    records: list[dict[str, Any]],
    *,
    warm_start_steps: int,
    median_threshold: float = -0.1,
) -> dict[str, Any]:
    """Apply the frozen operational directional-conflict criterion.

    Sparse HD2 diagnostics occur once per 500-step window, so two adjacent
    negative checkpoints represent two consecutive windows.  We additionally
    require that validation had been improving before physics activation and
    then stalls/worsens across the two conflict windows.
    """
    ordered = sorted(records, key=lambda row: int(row["step"]))
    pre = [row for row in ordered if int(row["step"]) <= warm_start_steps]
    post = [row for row in ordered if int(row["step"]) > warm_start_steps]
    pre_improving = (
        len(pre) >= 2
        and float(pre[-1]["validation_price_rmse"])
        < float(pre[-2]["validation_price_rmse"])
    )
    if not pre_improving:
        return {"triggered": False, "trigger_steps": [], "reason": "no_prephysics_improving_trace"}
    for left, right in zip(post, post[1:]):
        if int(right["step"]) - int(left["step"]) != 500:
            continue
        cosines = [float(left["cosine"]), float(right["cosine"])]
        median = float(np.median(cosines))
        stalled = float(right["validation_price_rmse"]) >= float(left["validation_price_rmse"])
        if all(value < 0.0 for value in cosines) and median <= median_threshold and stalled:
            return {
                "triggered": True,
                "trigger_steps": [int(left["step"]), int(right["step"])],
                "median_cosine": median,
                "validation_stalled": True,
            }
    return {"triggered": False, "trigger_steps": [], "reason": "criterion_not_met"}


def _candidate_settings(candidate: str, cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        return dict(cfg["candidate_settings"][candidate])
    except KeyError as exc:
        raise ValueError(f"unknown HD2 candidate: {candidate}") from exc


def _physics_score(candidate_rows: list[dict[str, Any]], baseline: dict[int, dict[str, Any]]) -> float:
    ratios: list[float] = []
    for row in candidate_rows:
        base = baseline[int(row["dimension"])]
        ratios.append(float(row["pde_p95"]) / max(float(base["pde_p95"]), 1e-15))
        ratios.append(float(row["fb_p95"]) / max(float(base["fb_p95"]), 1e-15))
    return float(math.exp(sum(math.log(max(value, 1e-15)) for value in ratios) / len(ratios)))


def select_hd2_candidate(
    metrics_rows: list[dict[str, Any]],
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Price-first lexicographic selection using validation metrics only."""
    cfg = load_config() if cfg is None else cfg
    dimensions = [int(value) for value in cfg["dimensions"]]
    baseline = {
        int(row["dimension"]): row
        for row in metrics_rows
        if row["candidate"] == "SUP-HD2"
    }
    if sorted(baseline) != sorted(dimensions):
        raise ValueError("SUP-HD2 validation baseline is incomplete")
    gates = cfg["selection_gates"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in metrics_rows:
        candidate = str(row["candidate"])
        if candidate == "SUP-HD2":
            continue
        grouped.setdefault(candidate, []).append(row)

    audit: dict[str, Any] = {}
    eligible: list[tuple[str, float]] = []
    for candidate, rows in grouped.items():
        by_d = {int(row["dimension"]): row for row in rows}
        if sorted(by_d) != sorted(dimensions):
            audit[candidate] = {"eligible": False, "reason": "incomplete_dimensions"}
            continue
        dimension_audit: dict[str, Any] = {}
        candidate_ok = True
        for d in dimensions:
            row = by_d[d]
            base = baseline[d]
            ratios = {
                "price_ratio": float(row["price_rmse"]) / max(float(base["price_rmse"]), 1e-15),
                "gamma_ratio": float(row["gamma_rmse"]) / max(float(base["gamma_rmse"]), 1e-15),
                "boundary_ratio": float(row["boundary_rmse"]) / max(float(base["boundary_rmse"]), 1e-15),
                "pde_p95_ratio": float(row["pde_p95"]) / max(float(base["pde_p95"]), 1e-15),
                "fb_p95_ratio": float(row["fb_p95"]) / max(float(base["fb_p95"]), 1e-15),
            }
            passes = bool(row.get("integrity", False)) and (
                ratios["price_ratio"] <= float(gates["price_ratio_max"])
                and ratios["gamma_ratio"] <= float(gates["gamma_ratio_max"])
                and ratios["boundary_ratio"] <= float(gates["boundary_ratio_max"])
                and ratios["pde_p95_ratio"] <= float(gates["pde_p95_ratio_max"])
                and ratios["fb_p95_ratio"] <= float(gates["fb_p95_ratio_max"])
            )
            candidate_ok = candidate_ok and passes
            dimension_audit[str(d)] = {**ratios, "passes": passes}
        score = _physics_score(rows, baseline)
        audit[candidate] = {
            "eligible": candidate_ok,
            "physics_score": score,
            "dimensions": dimension_audit,
        }
        if candidate_ok:
            eligible.append((candidate, score))

    if not eligible:
        fallback = dict(cfg["fallback_objective"])
        fallback["fallback_objective"] = True
        return {
            "selection_status": "no_stabilized_candidate",
            "eligible_candidates": [],
            "candidate_audit": audit,
            "resolved_objective": fallback,
            "diagnostic_only": True,
        }

    eligible.sort(key=lambda item: (item[1], item[0]))
    selected = eligible[0][0]
    settings = _candidate_settings(selected, cfg)
    return {
        "selection_status": "stabilized_candidate_selected",
        "eligible_candidates": [name for name, _ in eligible],
        "candidate_audit": audit,
        "resolved_objective": {
            "candidate": selected,
            "warm_fraction": float(settings["warm_start_steps"]) / float(cfg["total_steps"]),
            "final_physics_weight": float(settings["final_physics_weight"]),
            "fallback_objective": False,
        },
        "diagnostic_only": True,
    }


def _oracle_path(d: int) -> Path:
    return ORACLE_DIR / f"d{d}.npz"


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
        d,
        float(z["sigma_g"]),
        float(z["q_g"]),
        z["spots"],
        z["taus"],
        z["values"],
        z["ug"],
        z["ugg"],
        z["boundary"],
        float(z["refinement_rmse"]),
        float(z["seconds"]),
    )


def _fixed_supervised_data(
    d: int,
    cfg: dict[str, Any],
    oracle: Oracle,
    split: str,
    n: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, tau = sobol_points(n, d, int(cfg["sobol_seeds"][split]))
    y = oracle_labels(oracle, x, tau, cfg)["price"]
    return (
        x.to(device=device, dtype=torch.float32),
        tau.to(device=device, dtype=torch.float32),
        torch.tensor(y, device=device, dtype=torch.float32),
    )


def _fixed_physics_points(
    d: int,
    n: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    engine = torch.quasirandom.SobolEngine(d + 1, scramble=True, seed=int(seed) + d)
    u = engine.draw(n).to(device=device, dtype=torch.float32)
    lo, hi = math.log(0.5), math.log(1.5)
    x = (lo + (hi - lo) * u[:, :d]).requires_grad_(True)
    tau = (0.0025 + (1.0 - 0.0025) * u[:, d]).requires_grad_(True)
    return x, tau


def _physics_loss_from_points(
    model: torch.nn.Module,
    x: torch.Tensor,
    tau: torch.Tensor,
    d: int,
    cfg: dict[str, Any],
    *,
    epsilon: float,
    probe_count: int,
    exact: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    value = model(x, tau)
    residual, contraction = black_scholes_vi_residual(
        value,
        x,
        tau,
        r=float(cfg["contract"]["r"]),
        q=float(cfg["contract"]["q"]),
        sigma=float(cfg["contract"]["sigma"]),
        rho=float(cfg["contract"]["rho"]),
        hessian_mode="full" if exact or d <= 2 else "hutchinson",
        probe_count=int(probe_count),
        probe_seed=int(cfg["hutchinson_probe_seed"]),
    )
    gap = value - geometric_put_payoff(x)
    loss = fischer_burmeister(gap, residual, epsilon).square().mean()
    return loss, contraction


def _validation_price_rmse(
    model: torch.nn.Module,
    x: torch.Tensor,
    tau: torch.Tensor,
    y: torch.Tensor,
) -> float:
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            chunks.append(model(x[start : start + 1024], tau[start : start + 1024]))
    pred = torch.cat(chunks)
    return float(torch.sqrt((pred - y).square().mean()).detach().cpu())


def _job_id(d: int, candidate: str) -> str:
    return f"d{d}_{candidate}"


def _job_dir(d: int, candidate: str) -> Path:
    return RUNTIME_DIR / _job_id(d, candidate)


def _status_path(d: int, candidate: str) -> Path:
    return _job_dir(d, candidate) / "status.json"


def _checkpoint_path(d: int, candidate: str) -> Path:
    return _job_dir(d, candidate) / "latest_checkpoint.pt"


def _save_training_checkpoint(
    *,
    path: Path,
    implementation_sha: str,
    cfg_hash: str,
    job_id: str,
    d: int,
    candidate: str,
    completed_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_generator: torch.Generator,
    physics_consumed: int,
    batch_size: int,
    elapsed_training_seconds: float,
    weight: float,
    epsilon: float,
) -> None:
    payload = {
        "implementation_sha": implementation_sha,
        "config_hash": cfg_hash,
        "job_id": job_id,
        "dimension": int(d),
        "candidate": candidate,
        "completed_step": int(completed_step),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "batch_generator_state": batch_generator.get_state(),
        "physics_consumed": int(physics_consumed),
        "batch_size": int(batch_size),
        "elapsed_training_seconds": float(elapsed_training_seconds),
        "fb_weight": float(weight),
        "epsilon": float(epsilon),
        "rng_state": capture_rng_state(),
    }
    save_checkpoint_atomic(path, payload)


def _diagnostic_record(
    *,
    model: torch.nn.Module,
    d: int,
    candidate: str,
    completed_step: int,
    weight: float,
    epsilon: float,
    x_train: torch.Tensor,
    tau_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    tau_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    requested = int(cfg["gradient_diagnostic_batch_size"][str(d)])
    n = min(requested, len(x_train))
    pred = model(x_train[:n], tau_train[:n])
    sup_loss = (pred - y_train[:n]).square().mean()
    xp, tp = _fixed_physics_points(d, n, int(cfg["sobol_seeds"]["diagnostic"]), pred.device)
    phys_loss, _ = _physics_loss_from_points(
        model,
        xp,
        tp,
        d,
        cfg,
        epsilon=epsilon,
        probe_count=int(cfg["hutchinson_probe_count"]),
    )
    global_row = objective_gradient_diagnostics(
        sup_loss,
        phys_loss,
        model,
        lambda_weight=weight,
    )
    validation_rmse = _validation_price_rmse(model, x_val, tau_val, y_val)
    row: dict[str, Any] = {
        "dimension": d,
        "candidate": candidate,
        "step": completed_step,
        "lambda": weight,
        "epsilon": epsilon,
        "supervised_loss": float(sup_loss.detach().cpu()),
        "physics_loss": float(phys_loss.detach().cpu()),
        "validation_price_rmse": validation_rmse,
        **global_row,
    }
    layer_rows: list[dict[str, Any]] = []
    if completed_step % int(cfg["layerwise_gradient_interval"]) == 0:
        groups = {
            "asset_encoder": list(model.asset_encoder.parameters()),
            "value_head": list(model.value_head.parameters()),
        }
        for layer_row in layerwise_gradient_diagnostics(
            sup_loss,
            phys_loss,
            groups,
            lambda_weight=weight,
        ):
            layer_rows.append(
                {
                    "dimension": d,
                    "candidate": candidate,
                    "step": completed_step,
                    **layer_row,
                }
            )
    return row, layer_rows


def hutchinson_gradient_diagnostic(
    *,
    model: torch.nn.Module,
    d: int,
    step: int,
    epsilon: float,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Compare exact and two-probe d=5 physics values and parameter gradients."""
    if d != 5:
        raise ValueError("the expensive exact-vs-Hutchinson audit is frozen to d=5")
    device = next(model.parameters()).device
    batch = int(cfg["hutchinson_gradient_batch_size"])
    x0, t0 = _fixed_physics_points(
        d,
        batch,
        int(cfg["sobol_seeds"]["hutchinson_diagnostic"]),
        device,
    )
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    exact_loss, exact_contraction = _physics_loss_from_points(
        model, x0, t0, d, cfg, epsilon=epsilon, probe_count=2, exact=True
    )
    exact_grads = torch.autograd.grad(exact_loss, params, retain_graph=False, allow_unused=True)
    g_exact = flatten_grads(exact_grads, params).detach()

    x1, t1 = _fixed_physics_points(
        d,
        batch,
        int(cfg["sobol_seeds"]["hutchinson_diagnostic"]),
        device,
    )
    hutch_loss, hutch_contraction = _physics_loss_from_points(
        model,
        x1,
        t1,
        d,
        cfg,
        epsilon=epsilon,
        probe_count=int(cfg["hutchinson_probe_count"]),
        exact=False,
    )
    hutch_grads = torch.autograd.grad(hutch_loss, params, retain_graph=False, allow_unused=True)
    g_hutch = flatten_grads(hutch_grads, params).detach()
    relative_error = float(
        (
            torch.linalg.vector_norm(hutch_contraction.detach() - exact_contraction.detach())
            / (torch.linalg.vector_norm(exact_contraction.detach()) + 1e-12)
        ).cpu()
    )
    denom = torch.linalg.vector_norm(g_exact) * torch.linalg.vector_norm(g_hutch)
    cosine = float((torch.dot(g_exact, g_hutch) / (denom + 1e-15)).cpu())
    norm_ratio = float(
        (torch.linalg.vector_norm(g_hutch) / (torch.linalg.vector_norm(g_exact) + 1e-15)).cpu()
    )
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    trigger = (
        relative_error > float(cfg["hutchinson_relative_error_trigger"])
        or cosine < float(cfg["hutchinson_gradient_cosine_trigger"])
    )
    return {
        "dimension": d,
        "step": int(step),
        "probe_count": int(cfg["hutchinson_probe_count"]),
        "relative_contraction_error": relative_error,
        "physics_gradient_cosine": cosine,
        "physics_gradient_norm_ratio": norm_ratio,
        "exact_physics_loss": float(exact_loss.detach().cpu()),
        "hutchinson_physics_loss": float(hutch_loss.detach().cpu()),
        "seconds": elapsed,
        "peak_cuda_gib": peak,
        "conditional_eight_probe_trigger": bool(trigger),
    }


def _training_once(
    d: int,
    candidate: str,
    cfg: dict[str, Any],
    *,
    batch_size: int,
    resume: bool,
    max_steps_override: int | None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required:HD2 training runs only on the target CUDA machine")
    device = torch.device("cuda:0")
    oracle = load_oracle(d)
    x_train, tau_train, y_train = _fixed_supervised_data(
        d, cfg, oracle, "train", int(cfg["supervised_train_points"]), device
    )
    x_val, tau_val, y_val = _fixed_supervised_data(
        d, cfg, oracle, "validation", int(cfg["validation_points"]), device
    )
    settings = _candidate_settings(candidate, cfg)
    total_steps = int(cfg["total_steps"])
    target_steps = min(total_steps, int(max_steps_override)) if max_steps_override else total_steps
    warm_start = int(settings["warm_start_steps"])
    final_weight = float(settings["final_physics_weight"])
    job_id = _job_id(d, candidate)
    job_dir = _job_dir(d, candidate)
    job_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _checkpoint_path(d, candidate)
    status_path = _status_path(d, candidate)
    cfg_hash = config_hash()
    implementation_sha = git_commit()

    model = initialize_model(int(cfg["seed"]), device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    batch_generator = torch.Generator(device="cpu")
    batch_generator.manual_seed(int(cfg["seed"]) + d)
    completed_step = 0
    physics_consumed = 0
    elapsed_before = 0.0

    if resume and checkpoint_path.exists():
        checkpoint = load_checkpoint_verified(
            checkpoint_path,
            implementation_sha=implementation_sha,
            config_hash=cfg_hash,
            job_id=job_id,
        )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        batch_generator.set_state(checkpoint["batch_generator_state"])
        completed_step = int(checkpoint["completed_step"])
        physics_consumed = int(checkpoint["physics_consumed"])
        elapsed_before = float(checkpoint.get("elapsed_training_seconds", 0.0))
        restore_rng_state(checkpoint["rng_state"])

    physics_engine = torch.quasirandom.SobolEngine(
        d + 1,
        scramble=True,
        seed=int(cfg["sobol_seeds"]["physics"]) + d,
    )
    if physics_consumed:
        physics_engine.fast_forward(physics_consumed)

    atomic_json(
        status_path,
        {
            "status": "running",
            "implementation_sha": implementation_sha,
            "config_hash": cfg_hash,
            "job_id": job_id,
            "dimension": d,
            "candidate": candidate,
            "completed_step": completed_step,
            "target_step": target_steps,
            "batch_size": batch_size,
        },
    )
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    latest_weight, latest_epsilon = curriculum(
        max(min(completed_step, total_steps - 1), 0),
        total_steps,
        warm_start,
        final_weight,
        epsilon_initial=float(cfg["fb_epsilon_initial"]),
        epsilon_final=float(cfg["fb_epsilon_final"]),
    )
    try:
        for step in range(completed_step, target_steps):
            weight, epsilon = curriculum(
                step,
                total_steps,
                warm_start,
                final_weight,
                epsilon_initial=float(cfg["fb_epsilon_initial"]),
                epsilon_final=float(cfg["fb_epsilon_final"]),
            )
            latest_weight, latest_epsilon = weight, epsilon
            idx_cpu = torch.randint(
                0,
                len(x_train),
                (batch_size,),
                generator=batch_generator,
            )
            idx = idx_cpu.to(device)
            pred = model(x_train[idx], tau_train[idx])
            supervised = (pred - y_train[idx]).square().mean()
            fb_loss = torch.zeros((), device=device)
            if candidate != "SUP-HD2" and step >= warm_start and weight > 0.0:
                u = physics_engine.draw(batch_size).to(device=device, dtype=torch.float32)
                physics_consumed += batch_size
                lo, hi = math.log(0.5), math.log(1.5)
                xp = (lo + (hi - lo) * u[:, :d]).requires_grad_(True)
                tp = (0.0025 + (1.0 - 0.0025) * u[:, d]).requires_grad_(True)
                fb_loss, _ = _physics_loss_from_points(
                    model,
                    xp,
                    tp,
                    d,
                    cfg,
                    epsilon=epsilon,
                    probe_count=int(cfg["hutchinson_probe_count"]),
                )
            total = supervised + weight * fb_loss
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            completed = step + 1

            peak = torch.cuda.max_memory_allocated(device) / 1024**3
            if peak >= float(cfg["max_cuda_allocation_gib"]):
                raise RuntimeError(f"cuda_memory_gate_failed:{peak:.4f}GiB")

            if completed == 1 or completed % int(cfg["train_log_interval"]) == 0 or completed == target_steps:
                append_jsonl(
                    job_dir / "training_history.jsonl",
                    {
                        "dimension": d,
                        "candidate": candidate,
                        "step": completed,
                        "supervised_mse": float(supervised.detach().cpu()),
                        "fb_loss": float(fb_loss.detach().cpu()),
                        "total_loss": float(total.detach().cpu()),
                        "fb_weight": weight,
                        "epsilon": epsilon,
                        "batch_size": batch_size,
                        "physics_consumed": physics_consumed,
                        "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                        "peak_cuda_gib": peak,
                    },
                )

            if candidate != "SUP-HD2" and completed % int(cfg["gradient_diagnostic_interval"]) == 0:
                diagnostic_row, layer_rows = _diagnostic_record(
                    model=model,
                    d=d,
                    candidate=candidate,
                    completed_step=completed,
                    weight=weight,
                    epsilon=epsilon,
                    x_train=x_train,
                    tau_train=tau_train,
                    y_train=y_train,
                    x_val=x_val,
                    tau_val=tau_val,
                    y_val=y_val,
                    cfg=cfg,
                )
                append_jsonl(job_dir / "gradient_diagnostics.jsonl", diagnostic_row)
                for layer_row in layer_rows:
                    append_jsonl(job_dir / "layerwise_gradient_diagnostics.jsonl", layer_row)

            if completed % int(cfg["validation_interval"]) == 0:
                append_jsonl(
                    job_dir / "validation_history.jsonl",
                    {
                        "dimension": d,
                        "candidate": candidate,
                        "step": completed,
                        "price_rmse": _validation_price_rmse(model, x_val, tau_val, y_val),
                    },
                )

            if (
                d == 5
                and candidate == "HVI-HD1-anchor"
                and completed in {int(value) for value in cfg["hutchinson_gradient_checkpoints"]}
            ):
                append_jsonl(
                    job_dir / "hutchinson_gradient_validation.jsonl",
                    hutchinson_gradient_diagnostic(
                        model=model,
                        d=d,
                        step=completed,
                        epsilon=epsilon,
                        cfg=cfg,
                    ),
                )

            if completed % int(cfg["checkpoint_interval"]) == 0 or completed == target_steps:
                elapsed = elapsed_before + time.perf_counter() - started
                _save_training_checkpoint(
                    path=checkpoint_path,
                    implementation_sha=implementation_sha,
                    cfg_hash=cfg_hash,
                    job_id=job_id,
                    d=d,
                    candidate=candidate,
                    completed_step=completed,
                    model=model,
                    optimizer=optimizer,
                    batch_generator=batch_generator,
                    physics_consumed=physics_consumed,
                    batch_size=batch_size,
                    elapsed_training_seconds=elapsed,
                    weight=weight,
                    epsilon=epsilon,
                )
    except KeyboardInterrupt:
        atomic_json(
            status_path,
            {
                "status": "interrupted",
                "implementation_sha": implementation_sha,
                "config_hash": cfg_hash,
                "job_id": job_id,
                "completed_step": completed_step,
                "checkpoint": str(checkpoint_path),
            },
        )
        raise

    elapsed_total = elapsed_before + time.perf_counter() - started
    if target_steps < total_steps:
        atomic_json(
            status_path,
            {
                "status": "interrupted_test_hook",
                "implementation_sha": implementation_sha,
                "config_hash": cfg_hash,
                "job_id": job_id,
                "completed_step": target_steps,
                "target_step": total_steps,
            },
        )
        return {"status": "interrupted_test_hook", "completed_step": target_steps}

    validation_metrics = evaluate_model_metrics(
        d=d,
        candidate=candidate,
        model=model,
        cfg=cfg,
        oracle=oracle,
        split="validation",
        n=int(cfg["validation_points"]),
        training_seconds=elapsed_total,
        batch_size=batch_size,
    )
    metrics_path = job_dir / "validation_metrics.json"
    atomic_json(metrics_path, validation_metrics)
    atomic_json(
        status_path,
        {
            "status": "completed",
            "implementation_sha": implementation_sha,
            "config_hash": cfg_hash,
            "job_id": job_id,
            "dimension": d,
            "candidate": candidate,
            "completed_step": total_steps,
            "target_step": total_steps,
            "batch_size": batch_size,
            "artifacts": {str(metrics_path): sha256_file(metrics_path)},
        },
    )
    return validation_metrics


def train_candidate(
    d: int,
    candidate: str,
    *,
    resume: bool = True,
    max_steps_override: int | None = None,
) -> dict[str, Any]:
    cfg = load_config()
    if d not in cfg["dimensions"]:
        raise ValueError("dimension is outside frozen HD2 scope")
    _candidate_settings(candidate, cfg)
    status_path = _status_path(d, candidate)
    if max_steps_override is None and verified_completed_status(
        status_path,
        implementation_sha=git_commit(),
        config_hash=config_hash(),
    ):
        return json.loads((_job_dir(d, candidate) / "validation_metrics.json").read_text(encoding="utf-8"))

    attempts_path = _job_dir(d, candidate) / "oom_attempts.jsonl"
    batches = [int(cfg["initial_batch_size"]), *[int(value) for value in cfg["oom_batch_fallbacks"]]]
    checkpoint = _checkpoint_path(d, candidate)
    if resume and checkpoint.exists():
        try:
            current = load_checkpoint_verified(
                checkpoint,
                implementation_sha=git_commit(),
                config_hash=config_hash(),
                job_id=_job_id(d, candidate),
            )
            current_batch = int(current.get("batch_size", batches[0]))
            if current_batch in batches:
                batches = batches[batches.index(current_batch) :]
        except RuntimeError:
            raise
    for batch in batches:
        try:
            return _training_once(
                d,
                candidate,
                cfg,
                batch_size=batch,
                resume=resume,
                max_steps_override=max_steps_override,
            )
        except torch.cuda.OutOfMemoryError:
            append_jsonl(
                attempts_path,
                {"batch_size": batch, "error": "cuda_oom", "time": time.time()},
            )
            torch.cuda.empty_cache()
            resume = True
    raise RuntimeError("oom_blocked:all_predeclared_batches_exhausted")


def evaluate_model_metrics(
    *,
    d: int,
    candidate: str,
    model: torch.nn.Module,
    cfg: dict[str, Any],
    oracle: Oracle,
    split: str,
    n: int,
    training_seconds: float,
    batch_size: int,
) -> dict[str, Any]:
    """Evaluate with the HD1 metric definitions on a named fresh HD2 split."""
    device = next(model.parameters()).device
    xcpu, tcpu = sobol_points(n, d, int(cfg["sobol_seeds"][split]))
    labels = oracle_labels(oracle, xcpu, tcpu, cfg)
    started = time.perf_counter()
    preds: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    contractions: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    gaps: list[np.ndarray] = []
    derivative_started = time.perf_counter()
    chunk_size = 512
    for start in range(0, len(xcpu), chunk_size):
        x = xcpu[start : start + chunk_size].to(device=device, dtype=torch.float32).requires_grad_(True)
        tau = tcpu[start : start + chunk_size].to(device=device, dtype=torch.float32).requires_grad_(True)
        value = model(x, tau)
        vx = torch.autograd.grad(value.sum(), x, create_graph=True, retain_graph=True)[0]
        residual, contraction = black_scholes_vi_residual(
            value,
            x,
            tau,
            r=float(cfg["contract"]["r"]),
            q=float(cfg["contract"]["q"]),
            sigma=float(cfg["contract"]["sigma"]),
            rho=float(cfg["contract"]["rho"]),
            hessian_mode="full" if d <= 2 else "hutchinson",
            probe_count=int(cfg["hutchinson_probe_count"]),
            probe_seed=int(cfg["hutchinson_probe_seed"]),
        )
        preds.append(value.detach().cpu().numpy())
        deltas.append((vx / torch.exp(x)).detach().cpu().numpy())
        contractions.append(contraction.detach().cpu().numpy())
        residuals.append(residual.detach().cpu().numpy())
        gaps.append((value - geometric_put_payoff(x)).detach().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    derivative_seconds = time.perf_counter() - derivative_started
    pred = np.concatenate(preds)
    delta = np.concatenate(deltas)
    contraction = np.concatenate(contractions)
    residual = np.concatenate(residuals)
    gap = np.concatenate(gaps)
    price_error = pred - labels["price"]
    delta_error = delta - labels["delta"]
    gamma_error = contraction - labels["contraction"]
    phi = np.sqrt(gap**2 + residual**2) - gap - residual
    continuation = labels["g"] > labels["boundary"] + 1e-8
    pde_abs = np.abs(residual[continuation]) if np.any(continuation) else np.abs(residual)
    true_exercise = labels["g"] <= labels["boundary"] + 1e-8
    pred_exercise = gap <= 1e-5
    tube = np.abs(labels["g"] - labels["boundary"]) <= 0.02

    taus = np.linspace(0.0025, 1.0, 64)
    gs = np.linspace(0.5, 1.2, 256)
    predicted_boundary: list[float] = []
    true_boundary: list[float] = []
    with torch.no_grad():
        for tau_value in taus:
            xd = torch.log(torch.tensor(gs, device=device, dtype=torch.float32)).reshape(-1, 1).repeat(1, d)
            td = torch.full((len(gs),), float(tau_value), device=device)
            vd = model(xd, td).cpu().numpy()
            payoff = np.maximum(1.0 - gs, 0.0)
            exercise = np.flatnonzero((gs <= 1.0) & ((vd - payoff) <= 1e-5))
            predicted_boundary.append(gs[exercise[-1]] if exercise.size else 0.0)
            true_boundary.append(np.interp(tau_value, oracle.taus, oracle.boundary))
    boundary_rmse = float(
        np.sqrt(np.mean((np.asarray(predicted_boundary) - np.asarray(true_boundary)) ** 2))
    )
    finite = all(
        np.all(np.isfinite(values))
        for values in (pred, delta, contraction, residual, gap, price_error, gamma_error)
    )
    peak_cuda = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    return {
        "dimension": d,
        "candidate": candidate,
        "split": split,
        "price_rmse": float(np.sqrt(np.mean(price_error**2))),
        "price_mae": float(np.mean(np.abs(price_error))),
        "price_p95": float(np.quantile(np.abs(price_error), 0.95)),
        "price_max": float(np.max(np.abs(price_error))),
        "delta_rmse": float(np.sqrt(np.mean(delta_error**2))),
        "delta_max": float(np.max(np.abs(delta_error))),
        "gamma_rmse": float(np.sqrt(np.mean(gamma_error**2))),
        "obstacle_violation_max": float(np.max(np.maximum(-gap, 0.0))),
        "obstacle_violation_rate": float(np.mean(gap < -1e-6)),
        "fb_rms": float(np.sqrt(np.mean(phi**2))),
        "fb_p95": float(np.quantile(np.abs(phi), 0.95)),
        "fb_max": float(np.max(np.abs(phi))),
        "pde_rms": float(np.sqrt(np.mean(pde_abs**2))),
        "pde_p95": float(np.quantile(pde_abs, 0.95)),
        "exercise_f1": f1_score(true_exercise, pred_exercise),
        "boundary_tube_f1": f1_score(true_exercise[tube], pred_exercise[tube]) if np.any(tube) else 0.0,
        "boundary_rmse": boundary_rmse,
        "training_seconds": float(training_seconds),
        "evaluation_seconds": time.perf_counter() - started,
        "derivative_seconds": derivative_seconds,
        "peak_cpu_memory_bytes": peak_cpu_memory_bytes(),
        "peak_cuda_gib": peak_cuda,
        "parameter_count": parameter_count(model),
        "batch_size": int(batch_size),
        "integrity": bool(finite and peak_cuda < float(cfg["max_cuda_allocation_gib"])),
    }


def _load_completed_model(d: int, candidate: str, cfg: dict[str, Any]) -> tuple[torch.nn.Module, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required")
    device = torch.device("cuda:0")
    checkpoint = load_checkpoint_verified(
        _checkpoint_path(d, candidate),
        implementation_sha=git_commit(),
        config_hash=config_hash(),
        job_id=_job_id(d, candidate),
    )
    if int(checkpoint["completed_step"]) != int(cfg["total_steps"]):
        raise RuntimeError("incomplete_training_checkpoint")
    model = initialize_model(int(cfg["seed"]), device=device, dtype=torch.float32)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def run_preflight() -> dict[str, Any]:
    cfg = load_config()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if HD1_CONFIG_PATH.exists():
        hd1_cfg = json.loads(HD1_CONFIG_PATH.read_text(encoding="utf-8"))
        old_seeds = set(int(value) for value in hd1_cfg.get("sobol_seeds", {}).values())
        new_seeds = set(int(value) for value in cfg["sobol_seeds"].values())
        if old_seeds & new_seeds:
            raise RuntimeError("hd2_sobol_seed_reuses_hd1")
    manifest = environment_manifest()
    atomic_json(RESULT_DIR / "environment_manifest.json", manifest)
    atomic_json(
        RESULT_DIR / "config_manifest.json",
        {
            "config": cfg,
            "config_sha256": config_hash(),
            "git_commit": git_commit(),
        },
    )
    if not manifest["cuda_available"]:
        raise RuntimeError("cuda_required:HD2 preflight")
    if int(manifest.get("gpu_total_memory_bytes") or 0) < 7 * 1024**3:
        raise RuntimeError("cuda_memory_blocked")
    oracle_rows = []
    for d in cfg["dimensions"]:
        oracle = build_oracle(int(d), cfg)
        save_oracle(oracle)
        oracle_rows.append(
            {
                "dimension": int(d),
                "refinement_rmse": oracle.refinement_rmse,
                "seconds": oracle.seconds,
            }
        )
    hessian = validate_hutchinson(cfg, device=torch.device("cuda:0"))
    atomic_json(RESULT_DIR / "preflight.json", {"oracles": oracle_rows, "hessian": hessian, "passed": True})
    return {"oracles": oracle_rows, "hessian": hessian, "passed": True}


def run_phase_a() -> list[dict[str, Any]]:
    return [train_candidate(d, "HVI-HD1-anchor") for d in (2, 5)]


def run_screen() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in ("SUP-HD2", "HVI-low", "HVI-warm", "HVI-warm-low"):
        for d in (2, 5):
            rows.append(train_candidate(d, candidate))
    return rows


def collect_validation_metrics() -> list[dict[str, Any]]:
    cfg = load_config()
    rows: list[dict[str, Any]] = []
    for candidate in cfg["candidates"]:
        for d in cfg["dimensions"]:
            path = _job_dir(int(d), candidate) / "validation_metrics.json"
            if path.exists():
                rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def run_select() -> dict[str, Any]:
    cfg = load_config()
    rows = collect_validation_metrics()
    decision = select_hd2_candidate(rows, cfg)
    anchor_records: list[dict[str, Any]] = []
    for d in cfg["dimensions"]:
        gradient_path = _job_dir(int(d), "HVI-HD1-anchor") / "gradient_diagnostics.jsonl"
        if gradient_path.exists():
            diagnostic = persistent_conflict(
                read_jsonl_recover(gradient_path),
                warm_start_steps=int(cfg["candidate_settings"]["HVI-HD1-anchor"]["warm_start_steps"]),
                median_threshold=float(cfg["persistent_conflict_median_cosine"]),
            )
            diagnostic["dimension"] = int(d)
            anchor_records.append(diagnostic)
    hutch_records = read_jsonl_recover(
        _job_dir(5, "HVI-HD1-anchor") / "hutchinson_gradient_validation.jsonl"
    )
    decision["persistent_conflict"] = anchor_records
    decision["hutchinson_diagnostics"] = hutch_records
    decision["cagrad_triggered"] = any(bool(row.get("triggered")) for row in anchor_records)
    decision["eight_probe_triggered"] = any(
        bool(row.get("conditional_eight_probe_trigger")) for row in hutch_records
    )
    decision["implementation_sha"] = git_commit()
    decision["config_hash"] = config_hash()
    decision["validation_metrics_hash"] = canonical_json_hash(rows)
    atomic_json(SELECTION_PATH, decision)
    return decision


def run_finalize() -> list[dict[str, Any]]:
    """Evaluate the untouched HD2 test stream only after selection is frozen."""
    cfg = load_config()
    if not SELECTION_PATH.exists():
        raise RuntimeError("selection_required_before_test_evaluation")
    decision = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if decision.get("implementation_sha") != git_commit() or decision.get("config_hash") != config_hash():
        raise RuntimeError("stale_selection_provenance")
    selected = str(decision["resolved_objective"]["candidate"])
    candidates = []
    for candidate in ("SUP-HD2", "HVI-HD1-anchor", selected):
        if candidate not in candidates:
            candidates.append(candidate)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for d in cfg["dimensions"]:
            model, checkpoint = _load_completed_model(int(d), candidate, cfg)
            oracle = load_oracle(int(d))
            row = evaluate_model_metrics(
                d=int(d),
                candidate=candidate,
                model=model,
                cfg=cfg,
                oracle=oracle,
                split="test",
                n=int(cfg["test_points"]),
                training_seconds=float(checkpoint.get("elapsed_training_seconds", 0.0)),
                batch_size=int(checkpoint.get("batch_size", cfg["initial_batch_size"])),
            )
            row["selection_status"] = decision["selection_status"]
            rows.append(row)
    _write_csv(RESULT_DIR / "test_metrics.csv", rows)
    atomic_json(
        RESULT_DIR / "final_status.json",
        {
            "status": "completed",
            "diagnostic_only": True,
            "candidate_supported": False,
            "selection_status": decision["selection_status"],
            "resolved_objective": decision["resolved_objective"],
            "rows": len(rows),
        },
    )
    return rows


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


def status_snapshot() -> dict[str, Any]:
    cfg = load_config()
    jobs = []
    for candidate in cfg["candidates"]:
        for d in cfg["dimensions"]:
            path = _status_path(int(d), candidate)
            if path.exists():
                jobs.append(json.loads(path.read_text(encoding="utf-8")))
            else:
                jobs.append({"dimension": int(d), "candidate": candidate, "status": "not_started"})
    return {
        "experiment": cfg["experiment_id"],
        "selection": json.loads(SELECTION_PATH.read_text(encoding="utf-8")) if SELECTION_PATH.exists() else None,
        "jobs": jobs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["preflight", "phase-a", "screen", "select", "finalize", "status"],
        required=True,
    )
    args = parser.parse_args(argv)
    if args.phase == "preflight":
        result = run_preflight()
    elif args.phase == "phase-a":
        result = run_phase_a()
    elif args.phase == "screen":
        result = run_screen()
    elif args.phase == "select":
        result = run_select()
    elif args.phase == "finalize":
        result = run_finalize()
    else:
        result = status_snapshot()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
