"""HD3 classical-vs-structured high-dimensional HVI PINN scaling.

The module consumes one frozen HD2 objective, then trains a conventional FC
PINN and the exact HD1 DeepSets PINN on matched data/minibatch/physics streams.
It supports fixed HVI, conditional CAGrad or ReLoBRaLo objectives, a resolved
2/8-probe Hutchinson count, sequential/concurrent batch modes, and durable
checkpoint resume.  CUDA evidence is generated only on the user's laptop.
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

from hd_pipeline.gradient_diagnostics import flatten_grads
from hd_pipeline.objective_optimizers import ReLoBRaLoState, cagrad_combine
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
from models.hd3_high_dimensional_pinn import initialize_fc_model
from models.high_dimensional_pinn import initialize_model as initialize_deepsets
from physics.hd3_high_dimensional_sampling import (
    grouped_transverse_states,
    permutation_consistency_rmse,
    sample_transverse_states,
    transverse_invariance_rmse,
)
from physics.high_dimensional_black_scholes import (
    black_scholes_vi_residual,
    fischer_burmeister,
    geometric_put_payoff,
)
from hd1_windows_high_dimensional_pinn_48h import (
    Oracle,
    build_oracle,
    environment_manifest,
    f1_score,
    git_commit,
    oracle_labels,
    peak_cpu_memory_bytes,
    validate_hutchinson,
)
from hd2_hvi_objective_stabilization import curriculum

CONFIG_PATH = ROOT / "configs" / "hd3_classical_vs_structured_high_dimensional_pinn.json"
HD2_SELECTION_PATH = ROOT / "results" / "hd2_hvi_objective_stabilization" / "selection_decision.json"
RESULT_DIR = ROOT / "results" / "hd3_classical_vs_structured_high_dimensional_pinn"
RUNTIME_DIR = RESULT_DIR / "runtime"
ORACLE_DIR = RUNTIME_DIR / "oracles"
RESOLVED_OBJECTIVE_PATH = RESULT_DIR / "resolved_objective.json"
CANONICAL_STARTED_PATH = RUNTIME_DIR / "canonical_started.json"


def load_config() -> dict[str, Any]:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg["seed"] != 101:
        raise ValueError("HD3 seed is frozen")
    if cfg["canonical_dimensions"] != [10, 20, 50] or cfg["stress_dimensions"] != [100]:
        raise ValueError("HD3 dimensions are frozen")
    if cfg["models"] != ["FC_HVI", "DEEPSETS_HVI"]:
        raise ValueError("HD3 model identities are frozen")
    if cfg["canonical_steps"] != 6000 or cfg["stress_steps"] != 4000:
        raise ValueError("HD3 step budgets are frozen")
    if cfg["hutchinson_probe_count"] != 2:
        raise ValueError("HD3 default Hutchinson probe count is frozen")
    if cfg.get("blind_holdout_access") is not False:
        raise ValueError("HD3 must not access the campaign blind holdout")
    return cfg


def config_hash() -> str:
    return sha256_file(CONFIG_PATH)


def paired_batch_seed(d: int, cfg: dict[str, Any], model_id: str | None = None) -> int:
    """Return the same minibatch RNG seed for both architecture IDs."""
    if model_id is not None and model_id not in cfg["models"]:
        raise ValueError("unknown HD3 model")
    return int(cfg["seed"]) + 1000 * int(d)


def _normalized_objective(payload: dict[str, Any]) -> dict[str, Any]:
    required = {"candidate", "warm_fraction", "final_physics_weight"}
    if not required.issubset(payload):
        raise ValueError("invalid_resolved_objective")
    warm_fraction = float(payload["warm_fraction"])
    final_weight = float(payload["final_physics_weight"])
    if not 0.0 <= warm_fraction <= 1.0 or final_weight < 0.0:
        raise ValueError("invalid_resolved_objective")
    result: dict[str, Any] = {
        "candidate": str(payload["candidate"]),
        "warm_fraction": warm_fraction,
        "final_physics_weight": final_weight,
        "fallback_objective": bool(payload.get("fallback_objective", False)),
    }
    for key in (
        "optimizer_mode",
        "hutchinson_probe_count",
        "cagrad_alpha",
        "cagrad_rescale",
        "relobralo_temperature",
        "relobralo_alpha",
        "relobralo_rho",
    ):
        if key in payload:
            result[key] = payload[key]
    return result


def resolve_objective(selection_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    selection_path = Path(selection_path)
    if not selection_path.is_file():
        raise RuntimeError("hd2_selection_required")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    status = selection.get("selection_status")
    if status == "stabilized_candidate_selected":
        return _normalized_objective(dict(selection["resolved_objective"]))
    if status == "no_stabilized_candidate":
        fallback = dict(cfg["fallback_objective"])
        fallback["fallback_objective"] = True
        return _normalized_objective(fallback)
    raise RuntimeError(f"unexpected_hd2_selection_status:{status}")


def freeze_objective(path: Path, objective: dict[str, Any], *, canonical_started: bool = False) -> str:
    path = Path(path)
    normalized = _normalized_objective(objective)
    if path.exists():
        existing = _normalized_objective(json.loads(path.read_text(encoding="utf-8")))
        if existing != normalized and canonical_started:
            raise RuntimeError("objective_immutable")
        if existing != normalized:
            atomic_json(path, normalized)
    else:
        atomic_json(path, normalized)
    return sha256_file(path)


def resolved_objective() -> tuple[dict[str, Any], str]:
    cfg = load_config()
    objective = resolve_objective(HD2_SELECTION_PATH, cfg)
    objective_hash = freeze_objective(
        RESOLVED_OBJECTIVE_PATH,
        objective,
        canonical_started=CANONICAL_STARTED_PATH.exists(),
    )
    return objective, objective_hash


def objective_mode(objective: dict[str, Any]) -> str:
    mode = str(objective.get("optimizer_mode", "fixed"))
    if mode not in {"fixed", "cagrad", "relobralo"}:
        raise ValueError(f"unsupported objective mode: {mode}")
    return mode


def objective_probe_count(objective: dict[str, Any], cfg: dict[str, Any]) -> int:
    count = int(objective.get("hutchinson_probe_count", cfg["hutchinson_probe_count"]))
    if count not in {2, 8}:
        raise ValueError("HD3 resolved probe count must be 2 or 8")
    return count


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


def _bounds(cfg: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    return tuple(float(v) for v in cfg["aggregate_log_moneyness"]), tuple(float(v) for v in cfg["tau"])


def _fixed_labeled_data(
    d: int,
    cfg: dict[str, Any],
    oracle: Oracle,
    *,
    split: str,
    n: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m_bounds, tau_bounds = _bounds(cfg)
    x, tau, m = sample_transverse_states(
        n,
        d,
        seed=int(cfg["sobol_seeds"][split]) + d,
        m_bounds=m_bounds,
        tau_bounds=tau_bounds,
        transverse_scale=float(cfg["transverse_scale"]),
        dtype=torch.float64,
    )
    labels = oracle_labels(oracle, x, tau, cfg)["price"]
    return (
        x.to(device=device, dtype=torch.float32),
        tau.to(device=device, dtype=torch.float32),
        torch.tensor(labels, device=device, dtype=torch.float32),
        m.to(device=device, dtype=torch.float32),
    )


def _transform_physics_unit(unit: torch.Tensor, d: int, cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    m_bounds, tau_bounds = _bounds(cfg)
    m = m_bounds[0] + (m_bounds[1] - m_bounds[0]) * unit[:, 0]
    raw = 2.0 * unit[:, 1 : d + 1] - 1.0
    delta = float(cfg["transverse_scale"]) * (raw - raw.mean(dim=1, keepdim=True))
    x = (m[:, None] + delta).requires_grad_(True)
    tau = (tau_bounds[0] + (tau_bounds[1] - tau_bounds[0]) * unit[:, d + 1]).requires_grad_(True)
    return x, tau


def _initialize_model(model_id: str, d: int, cfg: dict[str, Any], device: torch.device) -> torch.nn.Module:
    if model_id == "FC_HVI":
        return initialize_fc_model(
            d,
            int(cfg["seed"]),
            width=int(cfg["fc_width"]),
            hidden_layers=int(cfg["fc_hidden_layers"]),
            device=device,
            dtype=torch.float32,
        )
    if model_id == "DEEPSETS_HVI":
        return initialize_deepsets(int(cfg["seed"]), device=device, dtype=torch.float32)
    raise ValueError(f"unknown HD3 model: {model_id}")


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _job_id(d: int, model_id: str, run_class: str) -> str:
    return f"{run_class}_d{d}_{model_id}"


def _job_dir(d: int, model_id: str, run_class: str) -> Path:
    return RUNTIME_DIR / _job_id(d, model_id, run_class)


def _status_path(d: int, model_id: str, run_class: str) -> Path:
    return _job_dir(d, model_id, run_class) / "status.json"


def _checkpoint_path(d: int, model_id: str, run_class: str) -> Path:
    return _job_dir(d, model_id, run_class) / "latest_checkpoint.pt"


def _physics_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    tau: torch.Tensor,
    d: int,
    cfg: dict[str, Any],
    *,
    epsilon: float,
    probe_count: int,
) -> torch.Tensor:
    value = model(x, tau)
    residual, _ = black_scholes_vi_residual(
        value,
        x,
        tau,
        r=float(cfg["contract"]["r"]),
        q=float(cfg["contract"]["q"]),
        sigma=float(cfg["contract"]["sigma"]),
        rho=float(cfg["contract"]["rho"]),
        hessian_mode="hutchinson",
        probe_count=int(probe_count),
        probe_seed=int(cfg["hutchinson_probe_seed"]),
    )
    return fischer_burmeister(value - geometric_put_payoff(x), residual, epsilon).square().mean()


def _validation_price_rmse(model: torch.nn.Module, x: torch.Tensor, tau: torch.Tensor, y: torch.Tensor) -> float:
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            outputs.append(model(x[start : start + 512], tau[start : start + 512]))
    return float(torch.sqrt((torch.cat(outputs) - y).square().mean()).cpu())


def _assign_flat_gradient(parameters: list[torch.nn.Parameter], flat: torch.Tensor) -> None:
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.grad = flat[offset : offset + count].reshape_as(parameter).to(parameter).clone()
        offset += count
    if offset != flat.numel():
        raise RuntimeError("flat_gradient_size_mismatch")


def _make_relobralo(objective: dict[str, Any], state: dict[str, Any] | None = None) -> ReLoBRaLoState:
    if state is not None:
        return ReLoBRaLoState.from_state_dict(state)
    return ReLoBRaLoState(
        task_count=2,
        temperature=float(objective.get("relobralo_temperature", 1.0)),
        alpha=float(objective.get("relobralo_alpha", 0.999)),
        rho=float(objective.get("relobralo_rho", 1.0)),
    )


def _optimizer_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    supervised_loss: torch.Tensor,
    physics_loss: torch.Tensor,
    weight: float,
    mode: str,
    objective: dict[str, Any],
    relobralo: ReLoBRaLoState | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    optimizer.zero_grad(set_to_none=True)
    if mode == "fixed" or weight <= 0.0:
        total = supervised_loss + float(weight) * physics_loss
        total.backward()
        optimizer.step()
        return total, {"optimizer_mode": "fixed" if weight > 0.0 else "supervised"}
    if mode == "relobralo":
        if relobralo is None:
            raise RuntimeError("missing_relobralo_state")
        weights = relobralo.update(
            [float(supervised_loss.detach().cpu()), float(physics_loss.detach().cpu())]
        )
        total = float(weights[0]) * supervised_loss + float(weights[1]) * physics_loss
        total.backward()
        optimizer.step()
        return total, {
            "optimizer_mode": "relobralo",
            "supervised_weight": float(weights[0]),
            "physics_weight": float(weights[1]),
        }
    if mode == "cagrad":
        parameters = [p for p in model.parameters() if p.requires_grad]
        sup_tuple = torch.autograd.grad(
            supervised_loss, parameters, retain_graph=True, create_graph=False, allow_unused=True
        )
        phys_tuple = torch.autograd.grad(
            float(weight) * physics_loss,
            parameters,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        g_sup = flatten_grads(sup_tuple, parameters)
        g_phys = flatten_grads(phys_tuple, parameters)
        combined, meta = cagrad_combine(
            g_sup,
            g_phys,
            alpha=float(objective.get("cagrad_alpha", 0.5)),
            rescale=int(objective.get("cagrad_rescale", 1)),
        )
        _assign_flat_gradient(parameters, combined)
        optimizer.step()
        total = supervised_loss.detach() + float(weight) * physics_loss.detach()
        return total, {"optimizer_mode": "cagrad", **meta}
    raise ValueError(f"unsupported objective mode: {mode}")


def _save_checkpoint(
    *,
    path: Path,
    implementation_sha: str,
    cfg_hash: str,
    objective_hash: str,
    job_id: str,
    d: int,
    model_id: str,
    run_class: str,
    completed_step: int,
    target_steps: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_generator: torch.Generator,
    physics_consumed: int,
    batch_size: int,
    elapsed_training_seconds: float,
    weight: float,
    epsilon: float,
    execution_mode: str,
    objective_state: dict[str, Any] | None,
) -> None:
    save_checkpoint_atomic(
        path,
        {
            "implementation_sha": implementation_sha,
            "config_hash": cfg_hash,
            "resolved_objective_hash": objective_hash,
            "job_id": job_id,
            "dimension": int(d),
            "model_id": model_id,
            "run_class": run_class,
            "completed_step": int(completed_step),
            "target_steps": int(target_steps),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "batch_generator_state": batch_generator.get_state(),
            "physics_consumed": int(physics_consumed),
            "batch_size": int(batch_size),
            "execution_mode": execution_mode,
            "elapsed_training_seconds": float(elapsed_training_seconds),
            "fb_weight": float(weight),
            "epsilon": float(epsilon),
            "objective_state": objective_state,
            "rng_state": capture_rng_state(),
        },
    )


def _batch_candidates(cfg: dict[str, Any], execution_mode: str) -> list[int]:
    if execution_mode == "sequential":
        return [int(cfg["initial_batch_size"]), *[int(v) for v in cfg["oom_batch_fallbacks"]]]
    if execution_mode == "concurrent":
        return [int(cfg["concurrent_batch_size"]), *[int(v) for v in cfg["concurrent_batch_fallbacks"]]]
    raise ValueError("execution_mode must be sequential or concurrent")


def _train_once(
    d: int,
    model_id: str,
    run_class: str,
    cfg: dict[str, Any],
    objective: dict[str, Any],
    objective_hash: str,
    *,
    batch_size: int,
    execution_mode: str,
    resume: bool,
    max_steps_override: int | None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required:HD3 training runs only on the target CUDA machine")
    device = torch.device("cuda:0")
    if run_class not in {"canonical", "stress"}:
        raise ValueError("run_class must be canonical or stress")
    target_steps = int(cfg["canonical_steps"] if run_class == "canonical" else cfg["stress_steps"])
    target_run = min(target_steps, int(max_steps_override)) if max_steps_override else target_steps
    warm_steps = int(round(target_steps * float(objective["warm_fraction"])))
    final_weight = float(objective["final_physics_weight"])
    mode = objective_mode(objective)
    probe_count = objective_probe_count(objective, cfg)
    oracle = load_oracle(d)
    x_train, tau_train, y_train, _ = _fixed_labeled_data(
        d, cfg, oracle, split="train", n=int(cfg["supervised_train_points"]), device=device
    )
    x_val, tau_val, y_val, _ = _fixed_labeled_data(
        d, cfg, oracle, split="validation", n=int(cfg["validation_points"]), device=device
    )
    model = _initialize_model(model_id, d, cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    batch_generator = torch.Generator(device="cpu")
    batch_generator.manual_seed(paired_batch_seed(d, cfg, model_id))
    physics_engine = torch.quasirandom.SobolEngine(
        d + 2, scramble=True, seed=int(cfg["sobol_seeds"]["physics"]) + d
    )
    implementation_sha, cfg_hash = git_commit(), config_hash()
    job_id = _job_id(d, model_id, run_class)
    job_dir = _job_dir(d, model_id, run_class)
    job_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path, status_path = _checkpoint_path(d, model_id, run_class), _status_path(d, model_id, run_class)
    completed_step, physics_consumed, elapsed_before = 0, 0, 0.0
    relobralo = _make_relobralo(objective) if mode == "relobralo" else None

    if resume and checkpoint_path.exists():
        checkpoint = load_checkpoint_verified(
            checkpoint_path,
            implementation_sha=implementation_sha,
            config_hash=cfg_hash,
            job_id=job_id,
        )
        if checkpoint.get("resolved_objective_hash") != objective_hash:
            raise RuntimeError("stale_objective_provenance")
        if int(checkpoint.get("target_steps", target_steps)) != target_steps:
            raise RuntimeError("stale_step_budget")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        batch_generator.set_state(checkpoint["batch_generator_state"])
        completed_step = int(checkpoint["completed_step"])
        physics_consumed = int(checkpoint["physics_consumed"])
        elapsed_before = float(checkpoint.get("elapsed_training_seconds", 0.0))
        restore_rng_state(checkpoint["rng_state"])
        if mode == "relobralo" and checkpoint.get("objective_state"):
            relobralo = _make_relobralo(objective, checkpoint["objective_state"])
    if physics_consumed:
        physics_engine.fast_forward(physics_consumed)

    if run_class == "canonical" and not CANONICAL_STARTED_PATH.exists():
        atomic_json(
            CANONICAL_STARTED_PATH,
            {
                "implementation_sha": implementation_sha,
                "config_hash": cfg_hash,
                "resolved_objective_hash": objective_hash,
                "first_job_id": job_id,
            },
        )
    atomic_json(
        status_path,
        {
            "status": "running",
            "implementation_sha": implementation_sha,
            "config_hash": cfg_hash,
            "resolved_objective_hash": objective_hash,
            "job_id": job_id,
            "dimension": d,
            "model_id": model_id,
            "run_class": run_class,
            "execution_mode": execution_mode,
            "completed_step": completed_step,
            "target_step": target_steps,
            "batch_size": batch_size,
        },
    )
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    latest_weight, latest_epsilon = 0.0, float(cfg["fb_epsilon_initial"])
    try:
        for step in range(completed_step, target_run):
            schedule_final = 1.0 if mode == "relobralo" else final_weight
            weight, epsilon = curriculum(
                step,
                target_steps,
                warm_steps,
                schedule_final,
                epsilon_initial=float(cfg["fb_epsilon_initial"]),
                epsilon_final=float(cfg["fb_epsilon_final"]),
            )
            if mode == "relobralo" and step >= warm_steps:
                weight = 1.0
            latest_weight, latest_epsilon = weight, epsilon
            index = torch.randint(
                0, len(x_train), (batch_size,), generator=batch_generator
            ).to(device)
            prediction = model(x_train[index], tau_train[index])
            supervised_loss = (prediction - y_train[index]).square().mean()
            physics_loss = torch.zeros((), device=device)
            if step >= warm_steps and weight > 0.0:
                unit = physics_engine.draw(batch_size).to(device=device, dtype=torch.float32)
                physics_consumed += batch_size
                xp, tp = _transform_physics_unit(unit, d, cfg)
                physics_loss = _physics_loss(
                    model, xp, tp, d, cfg, epsilon=epsilon, probe_count=probe_count
                )
            total_loss, update_meta = _optimizer_step(
                model=model,
                optimizer=optimizer,
                supervised_loss=supervised_loss,
                physics_loss=physics_loss,
                weight=weight,
                mode=mode,
                objective=objective,
                relobralo=relobralo,
            )
            completed_step = step + 1
            peak = torch.cuda.max_memory_allocated(device) / 1024**3
            if peak >= float(cfg["max_cuda_allocation_gib"]):
                raise RuntimeError(f"cuda_memory_gate_failed:{peak:.4f}GiB")
            if completed_step == 1 or completed_step % int(cfg["train_log_interval"]) == 0 or completed_step == target_run:
                append_jsonl(
                    job_dir / "training_history.jsonl",
                    {
                        "dimension": d,
                        "model_id": model_id,
                        "run_class": run_class,
                        "step": completed_step,
                        "supervised_mse": float(supervised_loss.detach().cpu()),
                        "fb_loss": float(physics_loss.detach().cpu()),
                        "total_loss": float(total_loss.detach().cpu()),
                        "fb_weight": weight,
                        "epsilon": epsilon,
                        "probe_count": probe_count,
                        "batch_size": batch_size,
                        "execution_mode": execution_mode,
                        "physics_consumed": physics_consumed,
                        "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                        "peak_cuda_gib": peak,
                        **update_meta,
                    },
                )
            if completed_step % int(cfg["validation_interval"]) == 0:
                append_jsonl(
                    job_dir / "validation_history.jsonl",
                    {
                        "dimension": d,
                        "model_id": model_id,
                        "run_class": run_class,
                        "step": completed_step,
                        "price_rmse": _validation_price_rmse(model, x_val, tau_val, y_val),
                    },
                )
            if completed_step % int(cfg["checkpoint_interval"]) == 0 or completed_step == target_run:
                _save_checkpoint(
                    path=checkpoint_path,
                    implementation_sha=implementation_sha,
                    cfg_hash=cfg_hash,
                    objective_hash=objective_hash,
                    job_id=job_id,
                    d=d,
                    model_id=model_id,
                    run_class=run_class,
                    completed_step=completed_step,
                    target_steps=target_steps,
                    model=model,
                    optimizer=optimizer,
                    batch_generator=batch_generator,
                    physics_consumed=physics_consumed,
                    batch_size=batch_size,
                    elapsed_training_seconds=elapsed_before + time.perf_counter() - started,
                    weight=weight,
                    epsilon=epsilon,
                    execution_mode=execution_mode,
                    objective_state=relobralo.state_dict() if relobralo is not None else None,
                )
    except KeyboardInterrupt:
        atomic_json(
            status_path,
            {
                "status": "interrupted",
                "implementation_sha": implementation_sha,
                "config_hash": cfg_hash,
                "resolved_objective_hash": objective_hash,
                "job_id": job_id,
                "completed_step": completed_step,
                "checkpoint": str(checkpoint_path),
            },
        )
        raise

    elapsed_total = elapsed_before + time.perf_counter() - started
    if target_run < target_steps:
        atomic_json(
            status_path,
            {
                "status": "interrupted_test_hook",
                "implementation_sha": implementation_sha,
                "config_hash": cfg_hash,
                "resolved_objective_hash": objective_hash,
                "job_id": job_id,
                "completed_step": completed_step,
                "target_step": target_steps,
            },
        )
        return {"status": "interrupted_test_hook", "completed_step": completed_step}

    metrics = evaluate_hd3_job(
        d=d,
        model_id=model_id,
        run_class=run_class,
        model=model,
        cfg=cfg,
        oracle=oracle,
        split="validation",
        n=int(cfg["validation_points"]),
        training_seconds=elapsed_total,
        batch_size=batch_size,
        probe_count=probe_count,
    )
    metrics_path = job_dir / "validation_metrics.json"
    atomic_json(metrics_path, metrics)
    atomic_json(
        status_path,
        {
            "status": "completed",
            "implementation_sha": implementation_sha,
            "config_hash": cfg_hash,
            "resolved_objective_hash": objective_hash,
            "job_id": job_id,
            "dimension": d,
            "model_id": model_id,
            "run_class": run_class,
            "completed_step": target_steps,
            "target_step": target_steps,
            "batch_size": batch_size,
            "execution_mode": execution_mode,
            "artifacts": {str(metrics_path): sha256_file(metrics_path)},
        },
    )
    return metrics


def train_hd3_job(
    d: int,
    model_id: str,
    *,
    run_class: str = "canonical",
    resume: bool = True,
    max_steps_override: int | None = None,
    execution_mode: str = "sequential",
) -> dict[str, Any]:
    cfg = load_config()
    if run_class == "canonical" and d not in cfg["canonical_dimensions"]:
        raise ValueError("dimension is outside canonical HD3 scope")
    if run_class == "stress" and d not in cfg["stress_dimensions"]:
        raise ValueError("dimension is outside stress HD3 scope")
    if model_id not in cfg["models"]:
        raise ValueError("unknown HD3 model")
    objective, objective_hash = resolved_objective()
    status_path = _status_path(d, model_id, run_class)
    if max_steps_override is None and verified_completed_status(
        status_path, implementation_sha=git_commit(), config_hash=config_hash()
    ):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("resolved_objective_hash") != objective_hash:
            raise RuntimeError("stale_objective_provenance")
        return json.loads((_job_dir(d, model_id, run_class) / "validation_metrics.json").read_text())

    batches = _batch_candidates(cfg, execution_mode)
    checkpoint_path = _checkpoint_path(d, model_id, run_class)
    if resume and checkpoint_path.exists():
        checkpoint = load_checkpoint_verified(
            checkpoint_path,
            implementation_sha=git_commit(),
            config_hash=config_hash(),
            job_id=_job_id(d, model_id, run_class),
        )
        if checkpoint.get("resolved_objective_hash") != objective_hash:
            raise RuntimeError("stale_objective_provenance")
        checkpoint_batch = int(checkpoint.get("batch_size", batches[0]))
        if checkpoint_batch in batches:
            batches = batches[batches.index(checkpoint_batch) :]
        elif int(checkpoint.get("completed_step", 0)) > 0:
            # Changing execution mode is allowed only before a canonical job has
            # started; this keeps minibatch size part of job provenance.
            raise RuntimeError("batch_mode_changed_after_job_start")
    for batch_size in batches:
        try:
            return _train_once(
                d,
                model_id,
                run_class,
                cfg,
                objective,
                objective_hash,
                batch_size=batch_size,
                execution_mode=execution_mode,
                resume=resume,
                max_steps_override=max_steps_override,
            )
        except torch.cuda.OutOfMemoryError:
            append_jsonl(
                _job_dir(d, model_id, run_class) / "oom_attempts.jsonl",
                {"batch_size": batch_size, "error": "cuda_oom", "time": time.time()},
            )
            torch.cuda.empty_cache()
            resume = True
    raise RuntimeError("oom_blocked:all_predeclared_batches_exhausted")


def _evaluation_states(d: int, cfg: dict[str, Any], split: str, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    m_bounds, tau_bounds = _bounds(cfg)
    x, tau, _ = sample_transverse_states(
        n,
        d,
        seed=int(cfg["sobol_seeds"][split]) + d,
        m_bounds=m_bounds,
        tau_bounds=tau_bounds,
        transverse_scale=float(cfg["transverse_scale"]),
        dtype=torch.float64,
    )
    return x, tau


def evaluate_hd3_job(
    *,
    d: int,
    model_id: str,
    run_class: str,
    model: torch.nn.Module,
    cfg: dict[str, Any],
    oracle: Oracle,
    split: str,
    n: int,
    training_seconds: float,
    batch_size: int,
    probe_count: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    xcpu, tcpu = _evaluation_states(d, cfg, split, n)
    labels = oracle_labels(oracle, xcpu, tcpu, cfg)
    evaluation_started = time.perf_counter()
    infer_started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, n, 512):
            model(xcpu[start:start+512].to(device=device, dtype=torch.float32), tcpu[start:start+512].to(device=device, dtype=torch.float32))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - infer_started

    derivative_started = time.perf_counter()
    predictions: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    contractions: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    gaps: list[np.ndarray] = []
    for start in range(0, n, 512):
        x = xcpu[start:start+512].to(device=device, dtype=torch.float32).requires_grad_(True)
        tau = tcpu[start:start+512].to(device=device, dtype=torch.float32).requires_grad_(True)
        value = model(x, tau)
        vx = torch.autograd.grad(value.sum(), x, create_graph=True, retain_graph=True)[0]
        residual, contraction = black_scholes_vi_residual(
            value, x, tau,
            r=float(cfg["contract"]["r"]), q=float(cfg["contract"]["q"]),
            sigma=float(cfg["contract"]["sigma"]), rho=float(cfg["contract"]["rho"]),
            hessian_mode="hutchinson", probe_count=int(probe_count),
            probe_seed=int(cfg["hutchinson_probe_seed"]),
        )
        predictions.append(value.detach().cpu().numpy())
        deltas.append((vx / torch.exp(x)).detach().cpu().numpy())
        contractions.append(contraction.detach().cpu().numpy())
        residuals.append(residual.detach().cpu().numpy())
        gaps.append((value - geometric_put_payoff(x)).detach().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    derivative_seconds = time.perf_counter() - derivative_started
    prediction, delta = np.concatenate(predictions), np.concatenate(deltas)
    contraction, residual, gap = np.concatenate(contractions), np.concatenate(residuals), np.concatenate(gaps)
    price_error, delta_error = prediction - labels["price"], delta - labels["delta"]
    gamma_error = contraction - labels["contraction"]
    phi = np.sqrt(gap**2 + residual**2) - gap - residual
    continuation = labels["g"] > labels["boundary"] + 1e-8
    pde_abs = np.abs(residual[continuation]) if np.any(continuation) else np.abs(residual)
    true_exercise = labels["g"] <= labels["boundary"] + 1e-8
    predicted_exercise = gap <= 1e-5
    tube = np.abs(labels["g"] - labels["boundary"]) <= 0.02

    taus, gs = np.linspace(0.0025, 1.0, 64), np.linspace(0.5, 1.2, 256)
    predicted_boundary: list[float] = []
    true_boundary: list[float] = []
    with torch.no_grad():
        for tau_value in taus:
            xd = torch.log(torch.tensor(gs, device=device, dtype=torch.float32)).reshape(-1, 1).repeat(1, d)
            td = torch.full((len(gs),), float(tau_value), device=device)
            vd = model(xd, td).detach().cpu().numpy()
            payoff = np.maximum(1.0 - gs, 0.0)
            exercise = np.flatnonzero((gs <= 1.0) & ((vd - payoff) <= 1e-5))
            predicted_boundary.append(gs[exercise[-1]] if exercise.size else 0.0)
            true_boundary.append(float(np.interp(tau_value, oracle.taus, oracle.boundary)))
    boundary_rmse = float(np.sqrt(np.mean((np.asarray(predicted_boundary) - np.asarray(true_boundary)) ** 2)))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg["sobol_seeds"]["structural"]) + d)
    permutation = torch.randperm(d, generator=generator)
    structural_n = min(1024, n)
    permutation_rmse = permutation_consistency_rmse(model, xcpu[:structural_n], tcpu[:structural_n], permutation)
    m_bounds, tau_bounds = _bounds(cfg)
    x_grouped, tau_grouped, _ = grouped_transverse_states(
        groups=128, repeats=4, d=d,
        seed=int(cfg["sobol_seeds"]["structural"]) + 1000 + d,
        m_bounds=m_bounds, tau_bounds=tau_bounds,
        transverse_scale=float(cfg["transverse_scale"]), dtype=torch.float64,
    )
    transverse_rmse = transverse_invariance_rmse(model, x_grouped, tau_grouped)
    finite = all(np.all(np.isfinite(v)) for v in (prediction, delta, contraction, residual, gap, price_error, gamma_error))
    peak_cuda = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    return {
        "dimension": d, "model_id": model_id, "run_class": run_class, "split": split,
        "normalized_price_rmse": float(np.sqrt(np.mean(price_error**2))),
        "normalized_price_mae": float(np.mean(np.abs(price_error))),
        "normalized_price_p95": float(np.quantile(np.abs(price_error), 0.95)),
        "normalized_price_max": float(np.max(np.abs(price_error))),
        "delta_rmse": float(np.sqrt(np.mean(delta_error**2))), "delta_max": float(np.max(np.abs(delta_error))),
        "gamma_contraction_rmse": float(np.sqrt(np.mean(gamma_error**2))),
        "obstacle_violation_max": float(np.max(np.maximum(-gap, 0.0))),
        "obstacle_violation_rate": float(np.mean(gap < -1e-6)),
        "fb_rms": float(np.sqrt(np.mean(phi**2))), "fb_p95": float(np.quantile(np.abs(phi), 0.95)), "fb_max": float(np.max(np.abs(phi))),
        "trusted_pde_rms": float(np.sqrt(np.mean(pde_abs**2))), "trusted_pde_p95": float(np.quantile(pde_abs, 0.95)),
        "exercise_f1": f1_score(true_exercise, predicted_exercise),
        "boundary_tube_f1": f1_score(true_exercise[tube], predicted_exercise[tube]) if np.any(tube) else 0.0,
        "boundary_rmse": boundary_rmse,
        "permutation_consistency_rmse": permutation_rmse, "transverse_invariance_rmse": transverse_rmse,
        "training_seconds": float(training_seconds), "inference_seconds": inference_seconds,
        "derivative_seconds": derivative_seconds, "evaluation_seconds": time.perf_counter() - evaluation_started,
        "oracle_seconds": float(oracle.seconds), "peak_cpu_memory_bytes": peak_cpu_memory_bytes(),
        "peak_cuda_gib": peak_cuda, "parameter_count": _parameter_count(model), "batch_size": int(batch_size),
        "hutchinson_probe_count": int(probe_count),
        "integrity": bool(finite and peak_cuda < float(cfg["max_cuda_allocation_gib"])),
    }


def _load_completed_model(d: int, model_id: str, run_class: str, cfg: dict[str, Any], objective_hash: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required")
    checkpoint = load_checkpoint_verified(
        _checkpoint_path(d, model_id, run_class), implementation_sha=git_commit(),
        config_hash=config_hash(), job_id=_job_id(d, model_id, run_class)
    )
    if checkpoint.get("resolved_objective_hash") != objective_hash:
        raise RuntimeError("stale_objective_provenance")
    target = int(cfg["canonical_steps"] if run_class == "canonical" else cfg["stress_steps"])
    if int(checkpoint["completed_step"]) != target:
        raise RuntimeError("incomplete_training_checkpoint")
    model = _initialize_model(model_id, d, cfg, torch.device("cuda:0"))
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def run_preflight() -> dict[str, Any]:
    cfg = load_config()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    objective, objective_hash = resolved_objective()
    manifest = environment_manifest()
    atomic_json(RESULT_DIR / "environment_manifest.json", manifest)
    atomic_json(RESULT_DIR / "config_manifest.json", {
        "config": cfg, "config_sha256": config_hash(), "git_commit": git_commit(),
        "resolved_objective_hash": objective_hash,
    })
    if not manifest["cuda_available"]:
        raise RuntimeError("cuda_required:HD3 preflight")
    if int(manifest.get("gpu_total_memory_bytes") or 0) < 7 * 1024**3:
        raise RuntimeError("cuda_memory_blocked")
    oracle_rows = []
    for d in [*cfg["canonical_dimensions"], *cfg["stress_dimensions"]]:
        oracle = build_oracle(int(d), cfg)
        save_oracle(oracle)
        oracle_rows.append({"dimension": int(d), "refinement_rmse": float(oracle.refinement_rmse), "seconds": float(oracle.seconds)})
    hessian = validate_hutchinson(cfg, device=torch.device("cuda:0"))
    payload = {"passed": True, "resolved_objective": objective, "resolved_objective_hash": objective_hash, "oracles": oracle_rows, "hessian": hessian}
    atomic_json(RESULT_DIR / "preflight.json", payload)
    return payload


def run_canonical(*, execution_mode: str = "sequential") -> list[dict[str, Any]]:
    cfg = load_config()
    return [train_hd3_job(int(d), str(model), run_class="canonical", execution_mode=execution_mode) for d in cfg["canonical_dimensions"] for model in cfg["models"]]


def run_stress() -> list[dict[str, Any]]:
    cfg = load_config()
    return [train_hd3_job(int(d), str(model), run_class="stress", execution_mode="sequential") for d in cfg["stress_dimensions"] for model in cfg["models"]]


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
        writer.writeheader(); writer.writerows(rows); handle.flush()
    tmp.replace(path)


def run_finalize() -> list[dict[str, Any]]:
    cfg = load_config()
    objective, objective_hash = resolved_objective()
    probe_count = objective_probe_count(objective, cfg)
    rows: list[dict[str, Any]] = []
    for run_class, dimensions in (("canonical", cfg["canonical_dimensions"]), ("stress", cfg["stress_dimensions"])):
        for d in dimensions:
            for model_id in cfg["models"]:
                status_path = _status_path(int(d), str(model_id), run_class)
                if not status_path.exists():
                    if run_class == "stress": continue
                    raise RuntimeError(f"missing_canonical_job:d{d}:{model_id}")
                status = json.loads(status_path.read_text())
                if status.get("status") != "completed":
                    if run_class == "stress": continue
                    raise RuntimeError(f"incomplete_canonical_job:d{d}:{model_id}")
                model, checkpoint = _load_completed_model(int(d), str(model_id), run_class, cfg, objective_hash)
                row = evaluate_hd3_job(
                    d=int(d), model_id=str(model_id), run_class=run_class, model=model,
                    cfg=cfg, oracle=load_oracle(int(d)), split="test", n=int(cfg["test_points"]),
                    training_seconds=float(checkpoint.get("elapsed_training_seconds", 0.0)),
                    batch_size=int(checkpoint["batch_size"]), probe_count=probe_count,
                )
                row["resolved_objective"] = objective["candidate"]
                row["fallback_objective"] = bool(objective["fallback_objective"])
                row["optimizer_mode"] = objective_mode(objective)
                rows.append(row)
    _write_csv(RESULT_DIR / "run_metrics.csv", rows)
    _write_csv(RESULT_DIR / "structural_diagnostics.csv", [{k: row[k] for k in ("dimension", "model_id", "run_class", "permutation_consistency_rmse", "transverse_invariance_rmse")} for row in rows])
    atomic_json(RESULT_DIR / "final_status.json", {
        "status": "completed", "diagnostic_only": True, "candidate_supported": False,
        "resolved_objective": objective, "resolved_objective_hash": objective_hash,
        "canonical_dimensions": cfg["canonical_dimensions"],
        "stress_dimension_attempted": any(row["run_class"] == "stress" for row in rows), "rows": len(rows),
    })
    return rows


def status_snapshot() -> dict[str, Any]:
    cfg = load_config()
    objective = json.loads(RESOLVED_OBJECTIVE_PATH.read_text()) if RESOLVED_OBJECTIVE_PATH.exists() else None
    jobs = []
    for run_class, dimensions in (("canonical", cfg["canonical_dimensions"]), ("stress", cfg["stress_dimensions"])):
        for d in dimensions:
            for model_id in cfg["models"]:
                path = _status_path(int(d), str(model_id), run_class)
                jobs.append(json.loads(path.read_text()) if path.exists() else {"dimension": int(d), "model_id": str(model_id), "run_class": run_class, "status": "not_started"})
    return {
        "experiment": cfg["experiment_id"], "resolved_objective": objective,
        "resolved_objective_hash": sha256_file(RESOLVED_OBJECTIVE_PATH) if RESOLVED_OBJECTIVE_PATH.exists() else None,
        "canonical_started": CANONICAL_STARTED_PATH.exists(), "jobs": jobs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["preflight", "canonical", "stress", "finalize", "status"], required=True)
    parser.add_argument("--dimension", type=int)
    parser.add_argument("--model", choices=["FC_HVI", "DEEPSETS_HVI"])
    parser.add_argument("--execution-mode", choices=["sequential", "concurrent"], default="sequential")
    parser.add_argument("--max-steps-override", type=int)
    args = parser.parse_args(argv)
    if args.phase == "preflight": result: Any = run_preflight()
    elif args.phase in {"canonical", "stress"}:
        run_class = args.phase
        if args.dimension is not None or args.model is not None:
            if args.dimension is None or args.model is None: raise SystemExit("--dimension and --model must be supplied together")
            result = train_hd3_job(args.dimension, args.model, run_class=run_class, execution_mode=args.execution_mode, max_steps_override=args.max_steps_override)
        else:
            result = run_canonical(execution_mode=args.execution_mode) if run_class == "canonical" else run_stress()
    elif args.phase == "finalize": result = run_finalize()
    else: result = status_snapshot()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
