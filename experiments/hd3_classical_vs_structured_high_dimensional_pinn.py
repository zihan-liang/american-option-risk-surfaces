"""HD3 classical-vs-structured high-dimensional HVI PINN scaling.

HD3 consumes a frozen HD2 objective decision, then compares a conventional
six-layer fully connected PINN with the exact HD1 DeepSets architecture at
10D/20D/50D, with 100D explicitly marked as a stress test.  Training and
runtime persistence are CUDA-only; CPU CI verifies the deterministic contracts
and imports without claiming canonical results.
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
        raise ValueError("HD3 Hutchinson probe count is frozen")
    if cfg.get("blind_holdout_access") is not False:
        raise ValueError("HD3 must not access the campaign blind holdout")
    return cfg


def config_hash() -> str:
    return sha256_file(CONFIG_PATH)


def _normalized_objective(payload: dict[str, Any]) -> dict[str, Any]:
    required = {"candidate", "warm_fraction", "final_physics_weight"}
    if not required.issubset(payload):
        raise ValueError("invalid_resolved_objective")
    warm_fraction = float(payload["warm_fraction"])
    final_weight = float(payload["final_physics_weight"])
    if not 0.0 <= warm_fraction <= 1.0 or final_weight < 0.0:
        raise ValueError("invalid_resolved_objective")
    return {
        "candidate": str(payload["candidate"]),
        "warm_fraction": warm_fraction,
        "final_physics_weight": final_weight,
        "fallback_objective": bool(payload.get("fallback_objective", False)),
    }


def resolve_objective(selection_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve exactly one HD2-selected objective or the preregistered fallback."""
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


def freeze_objective(
    path: Path,
    objective: dict[str, Any],
    *,
    canonical_started: bool = False,
) -> str:
    """Persist the objective once and make it immutable after canonical start."""
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
    return (
        tuple(float(v) for v in cfg["aggregate_log_moneyness"]),
        tuple(float(v) for v in cfg["tau"]),
    )


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


def _transform_physics_unit(
    unit: torch.Tensor,
    d: int,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
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
        probe_count=int(cfg["hutchinson_probe_count"]),
        probe_seed=int(cfg["hutchinson_probe_seed"]),
    )
    gap = value - geometric_put_payoff(x)
    return fischer_burmeister(gap, residual, epsilon).square().mean()


def _validation_price_rmse(
    model: torch.nn.Module,
    x: torch.Tensor,
    tau: torch.Tensor,
    y: torch.Tensor,
) -> float:
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            outputs.append(model(x[start : start + 512], tau[start : start + 512]))
    pred = torch.cat(outputs)
    return float(torch.sqrt((pred - y).square().mean()).cpu())


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
            "elapsed_training_seconds": float(elapsed_training_seconds),
            "fb_weight": float(weight),
            "epsilon": float(epsilon),
            "rng_state": capture_rng_state(),
        },
    )


def _train_once(
    d: int,
    model_id: str,
    run_class: str,
    cfg: dict[str, Any],
    objective: dict[str, Any],
    objective_hash: str,
    *,
    batch_size: int,
    resume: bool,
    max_steps_override: int | None = None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required:HD3 training runs only on the target CUDA machine")
    device = torch.device("cuda:0")
    canonical = run_class == "canonical"
    if run_class not in {"canonical", "stress"}:
        raise ValueError("run_class must be canonical or stress")
    target_steps = int(cfg["canonical_steps"] if canonical else cfg["stress_steps"])
    if max_steps_override is not None:
        target_steps_run = min(target_steps, int(max_steps_override))
    else:
        target_steps_run = target_steps
    warm_steps = int(round(target_steps * float(objective["warm_fraction"])))
    final_weight = float(objective["final_physics_weight"])
    oracle = load_oracle(d)
    x_train, tau_train, y_train, _ = _fixed_labeled_data(
        d,
        cfg,
        oracle,
        split="train",
        n=int(cfg["supervised_train_points"]),
        device=device,
    )
    x_val, tau_val, y_val, _ = _fixed_labeled_data(
        d,
        cfg,
        oracle,
        split="validation",
        n=int(cfg["validation_points"]),
        device=device,
    )
    model = _initialize_model(model_id, d, cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    batch_generator = torch.Generator(device="cpu")
    batch_generator.manual_seed(int(cfg["seed"]) + 1000 * d + (0 if model_id == "FC_HVI" else 1))
    physics_engine = torch.quasirandom.SobolEngine(
        d + 2,
        scramble=True,
        seed=int(cfg["sobol_seeds"]["physics"]) + d,
    )
    implementation_sha = git_commit()
    cfg_hash = config_hash()
    job_id = _job_id(d, model_id, run_class)
    job_dir = _job_dir(d, model_id, run_class)
    job_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _checkpoint_path(d, model_id, run_class)
    status_path = _status_path(d, model_id, run_class)
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
    if physics_consumed:
        physics_engine.fast_forward(physics_consumed)

    if canonical and not CANONICAL_STARTED_PATH.exists():
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
            "completed_step": completed_step,
            "target_step": target_steps,
            "batch_size": batch_size,
        },
    )
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    latest_weight = 0.0
    latest_epsilon = float(cfg["fb_epsilon_initial"])
    try:
        for step in range(completed_step, target_steps_run):
            weight, epsilon = curriculum(
                step,
                target_steps,
                warm_steps,
                final_weight,
                epsilon_initial=float(cfg["fb_epsilon_initial"]),
                epsilon_final=float(cfg["fb_epsilon_final"]),
            )
            latest_weight, latest_epsilon = weight, epsilon
            index_cpu = torch.randint(
                0,
                len(x_train),
                (batch_size,),
                generator=batch_generator,
            )
            index = index_cpu.to(device)
            prediction = model(x_train[index], tau_train[index])
            supervised_loss = (prediction - y_train[index]).square().mean()
            physics_loss = torch.zeros((), device=device)
            if step >= warm_steps and weight > 0.0:
                unit = physics_engine.draw(batch_size).to(device=device, dtype=torch.float32)
                physics_consumed += batch_size
                xp, tp = _transform_physics_unit(unit, d, cfg)
                physics_loss = _physics_loss(model, xp, tp, d, cfg, epsilon=epsilon)
            total_loss = supervised_loss + weight * physics_loss
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            completed_step = step + 1
            peak = torch.cuda.max_memory_allocated(device) / 1024**3
            if peak >= float(cfg["max_cuda_allocation_gib"]):
                raise RuntimeError(f"cuda_memory_gate_failed:{peak:.4f}GiB")

            if (
                completed_step == 1
                or completed_step % int(cfg["train_log_interval"]) == 0
                or completed_step == target_steps_run
            ):
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
                        "batch_size": batch_size,
                        "physics_consumed": physics_consumed,
                        "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                        "peak_cuda_gib": peak,
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
            if (
                completed_step % int(cfg["checkpoint_interval"]) == 0
                or completed_step == target_steps_run
            ):
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
    if target_steps_run < target_steps:
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

    validation_metrics = evaluate_hd3_job(
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
    )
    metrics_path = job_dir / "validation_metrics.json"
    atomic_json(metrics_path, validation_metrics)
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
            "artifacts": {str(metrics_path): sha256_file(metrics_path)},
        },
    )
    return validation_metrics


def train_hd3_job(
    d: int,
    model_id: str,
    *,
    run_class: str = "canonical",
    resume: bool = True,
    max_steps_override: int | None = None,
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
        status_path,
        implementation_sha=git_commit(),
        config_hash=config_hash(),
    ):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("resolved_objective_hash") != objective_hash:
            raise RuntimeError("stale_objective_provenance")
        return json.loads(
            (_job_dir(d, model_id, run_class) / "validation_metrics.json").read_text(encoding="utf-8")
        )

    batches = [int(cfg["initial_batch_size"]), *[int(v) for v in cfg["oom_batch_fallbacks"]]]
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


def _evaluation_states(
    d: int,
    cfg: dict[str, Any],
    split: str,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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
) -> dict[str, Any]:
    device = next(model.parameters()).device
    xcpu, tcpu = _evaluation_states(d, cfg, split, n)
    labels = oracle_labels(oracle, xcpu, tcpu, cfg)
    evaluation_started = time.perf_counter()
    infer_started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, n, 512):
            model(
                xcpu[start : start + 512].to(device=device, dtype=torch.float32),
                tcpu[start : start + 512].to(device=device, dtype=torch.float32),
            )
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
        x = xcpu[start : start + 512].to(device=device, dtype=torch.float32).requires_grad_(True)
        tau = tcpu[start : start + 512].to(device=device, dtype=torch.float32).requires_grad_(True)
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
            hessian_mode="hutchinson",
            probe_count=int(cfg["hutchinson_probe_count"]),
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

    prediction = np.concatenate(predictions)
    delta = np.concatenate(deltas)
    contraction = np.concatenate(contractions)
    residual = np.concatenate(residuals)
    gap = np.concatenate(gaps)
    price_error = prediction - labels["price"]
    delta_error = delta - labels["delta"]
    gamma_error = contraction - labels["contraction"]
    phi = np.sqrt(gap**2 + residual**2) - gap - residual
    continuation = labels["g"] > labels["boundary"] + 1e-8
    pde_abs = np.abs(residual[continuation]) if np.any(continuation) else np.abs(residual)
    true_exercise = labels["g"] <= labels["boundary"] + 1e-8
    predicted_exercise = gap <= 1e-5
    tube = np.abs(labels["g"] - labels["boundary"]) <= 0.02

    taus = np.linspace(0.0025, 1.0, 64)
    gs = np.linspace(0.5, 1.2, 256)
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
    boundary_rmse = float(
        np.sqrt(np.mean((np.asarray(predicted_boundary) - np.asarray(true_boundary)) ** 2))
    )

    # Structural diagnostics use their own fixed stream and do not alter training.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg["sobol_seeds"]["structural"]) + d)
    permutation = torch.randperm(d, generator=generator)
    structural_n = min(1024, n)
    permutation_rmse = permutation_consistency_rmse(
        model,
        xcpu[:structural_n],
        tcpu[:structural_n],
        permutation,
    )
    m_bounds, tau_bounds = _bounds(cfg)
    x_grouped, tau_grouped, _ = grouped_transverse_states(
        groups=128,
        repeats=4,
        d=d,
        seed=int(cfg["sobol_seeds"]["structural"]) + 1000 + d,
        m_bounds=m_bounds,
        tau_bounds=tau_bounds,
        transverse_scale=float(cfg["transverse_scale"]),
        dtype=torch.float64,
    )
    transverse_rmse = transverse_invariance_rmse(model, x_grouped, tau_grouped)

    finite = all(
        np.all(np.isfinite(values))
        for values in (prediction, delta, contraction, residual, gap, price_error, gamma_error)
    )
    peak_cuda = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    return {
        "dimension": d,
        "model_id": model_id,
        "run_class": run_class,
        "split": split,
        "normalized_price_rmse": float(np.sqrt(np.mean(price_error**2))),
        "normalized_price_mae": float(np.mean(np.abs(price_error))),
        "normalized_price_p95": float(np.quantile(np.abs(price_error), 0.95)),
        "normalized_price_max": float(np.max(np.abs(price_error))),
        "delta_rmse": float(np.sqrt(np.mean(delta_error**2))),
        "delta_max": float(np.max(np.abs(delta_error))),
        "gamma_contraction_rmse": float(np.sqrt(np.mean(gamma_error**2))),
        "obstacle_violation_max": float(np.max(np.maximum(-gap, 0.0))),
        "obstacle_violation_rate": float(np.mean(gap < -1e-6)),
        "fb_rms": float(np.sqrt(np.mean(phi**2))),
        "fb_p95": float(np.quantile(np.abs(phi), 0.95)),
        "fb_max": float(np.max(np.abs(phi))),
        "trusted_pde_rms": float(np.sqrt(np.mean(pde_abs**2))),
        "trusted_pde_p95": float(np.quantile(pde_abs, 0.95)),
        "exercise_f1": f1_score(true_exercise, predicted_exercise),
        "boundary_tube_f1": (
            f1_score(true_exercise[tube], predicted_exercise[tube]) if np.any(tube) else 0.0
        ),
        "boundary_rmse": boundary_rmse,
        "permutation_consistency_rmse": permutation_rmse,
        "transverse_invariance_rmse": transverse_rmse,
        "training_seconds": float(training_seconds),
        "inference_seconds": inference_seconds,
        "derivative_seconds": derivative_seconds,
        "evaluation_seconds": time.perf_counter() - evaluation_started,
        "oracle_seconds": float(oracle.seconds),
        "peak_cpu_memory_bytes": peak_cpu_memory_bytes(),
        "peak_cuda_gib": peak_cuda,
        "parameter_count": _parameter_count(model),
        "batch_size": int(batch_size),
        "integrity": bool(finite and peak_cuda < float(cfg["max_cuda_allocation_gib"])),
    }


def _load_completed_model(
    d: int,
    model_id: str,
    run_class: str,
    cfg: dict[str, Any],
    objective_hash: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required")
    checkpoint = load_checkpoint_verified(
        _checkpoint_path(d, model_id, run_class),
        implementation_sha=git_commit(),
        config_hash=config_hash(),
        job_id=_job_id(d, model_id, run_class),
    )
    if checkpoint.get("resolved_objective_hash") != objective_hash:
        raise RuntimeError("stale_objective_provenance")
    target_steps = int(cfg["canonical_steps"] if run_class == "canonical" else cfg["stress_steps"])
    if int(checkpoint["completed_step"]) != target_steps:
        raise RuntimeError("incomplete_training_checkpoint")
    device = torch.device("cuda:0")
    model = _initialize_model(model_id, d, cfg, device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def run_preflight() -> dict[str, Any]:
    cfg = load_config()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    objective, objective_hash = resolved_objective()
    manifest = environment_manifest()
    atomic_json(RESULT_DIR / "environment_manifest.json", manifest)
    atomic_json(
        RESULT_DIR / "config_manifest.json",
        {
            "config": cfg,
            "config_sha256": config_hash(),
            "git_commit": git_commit(),
            "resolved_objective_hash": objective_hash,
        },
    )
    if not manifest["cuda_available"]:
        raise RuntimeError("cuda_required:HD3 preflight")
    if int(manifest.get("gpu_total_memory_bytes") or 0) < 7 * 1024**3:
        raise RuntimeError("cuda_memory_blocked")
    oracle_rows = []
    for d in [*cfg["canonical_dimensions"], *cfg["stress_dimensions"]]:
        oracle = build_oracle(int(d), cfg)
        save_oracle(oracle)
        oracle_rows.append(
            {
                "dimension": int(d),
                "refinement_rmse": float(oracle.refinement_rmse),
                "seconds": float(oracle.seconds),
            }
        )
    hessian = validate_hutchinson(
        {
            **cfg,
            "hutchinson_probe_seed": int(cfg["hutchinson_probe_seed"]),
        },
        device=torch.device("cuda:0"),
    )
    payload = {
        "passed": True,
        "resolved_objective": objective,
        "resolved_objective_hash": objective_hash,
        "oracles": oracle_rows,
        "hessian": hessian,
    }
    atomic_json(RESULT_DIR / "preflight.json", payload)
    return payload


def run_canonical() -> list[dict[str, Any]]:
    cfg = load_config()
    rows: list[dict[str, Any]] = []
    for d in cfg["canonical_dimensions"]:
        for model_id in cfg["models"]:
            rows.append(train_hd3_job(int(d), str(model_id), run_class="canonical"))
    return rows


def run_stress() -> list[dict[str, Any]]:
    cfg = load_config()
    rows: list[dict[str, Any]] = []
    for d in cfg["stress_dimensions"]:
        for model_id in cfg["models"]:
            rows.append(train_hd3_job(int(d), str(model_id), run_class="stress"))
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


def run_finalize() -> list[dict[str, Any]]:
    """Evaluate fresh fixed HD3 test states after all architecture choices are frozen."""
    cfg = load_config()
    objective, objective_hash = resolved_objective()
    rows: list[dict[str, Any]] = []
    for run_class, dimensions in (
        ("canonical", cfg["canonical_dimensions"]),
        ("stress", cfg["stress_dimensions"]),
    ):
        for d in dimensions:
            for model_id in cfg["models"]:
                status_path = _status_path(int(d), str(model_id), run_class)
                if not status_path.exists():
                    if run_class == "stress":
                        continue
                    raise RuntimeError(f"missing_canonical_job:d{d}:{model_id}")
                status = json.loads(status_path.read_text(encoding="utf-8"))
                if status.get("status") != "completed":
                    if run_class == "stress":
                        continue
                    raise RuntimeError(f"incomplete_canonical_job:d{d}:{model_id}")
                model, checkpoint = _load_completed_model(
                    int(d), str(model_id), run_class, cfg, objective_hash
                )
                oracle = load_oracle(int(d))
                row = evaluate_hd3_job(
                    d=int(d),
                    model_id=str(model_id),
                    run_class=run_class,
                    model=model,
                    cfg=cfg,
                    oracle=oracle,
                    split="test",
                    n=int(cfg["test_points"]),
                    training_seconds=float(checkpoint.get("elapsed_training_seconds", 0.0)),
                    batch_size=int(checkpoint.get("batch_size", cfg["initial_batch_size"])),
                )
                row["resolved_objective"] = objective["candidate"]
                row["fallback_objective"] = bool(objective["fallback_objective"])
                rows.append(row)
    _write_csv(RESULT_DIR / "run_metrics.csv", rows)
    structural_rows = [
        {
            "dimension": row["dimension"],
            "model_id": row["model_id"],
            "run_class": row["run_class"],
            "permutation_consistency_rmse": row["permutation_consistency_rmse"],
            "transverse_invariance_rmse": row["transverse_invariance_rmse"],
        }
        for row in rows
    ]
    _write_csv(RESULT_DIR / "structural_diagnostics.csv", structural_rows)
    atomic_json(
        RESULT_DIR / "final_status.json",
        {
            "status": "completed",
            "diagnostic_only": True,
            "candidate_supported": False,
            "resolved_objective": objective,
            "resolved_objective_hash": objective_hash,
            "canonical_dimensions": cfg["canonical_dimensions"],
            "stress_dimension_attempted": any(row["run_class"] == "stress" for row in rows),
            "rows": len(rows),
        },
    )
    return rows


def status_snapshot() -> dict[str, Any]:
    cfg = load_config()
    objective = None
    objective_hash = None
    if RESOLVED_OBJECTIVE_PATH.exists():
        objective = json.loads(RESOLVED_OBJECTIVE_PATH.read_text(encoding="utf-8"))
        objective_hash = sha256_file(RESOLVED_OBJECTIVE_PATH)
    jobs: list[dict[str, Any]] = []
    for run_class, dimensions in (
        ("canonical", cfg["canonical_dimensions"]),
        ("stress", cfg["stress_dimensions"]),
    ):
        for d in dimensions:
            for model_id in cfg["models"]:
                path = _status_path(int(d), str(model_id), run_class)
                if path.exists():
                    jobs.append(json.loads(path.read_text(encoding="utf-8")))
                else:
                    jobs.append(
                        {
                            "dimension": int(d),
                            "model_id": str(model_id),
                            "run_class": run_class,
                            "status": "not_started",
                        }
                    )
    return {
        "experiment": cfg["experiment_id"],
        "resolved_objective": objective,
        "resolved_objective_hash": objective_hash,
        "canonical_started": CANONICAL_STARTED_PATH.exists(),
        "jobs": jobs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["preflight", "canonical", "stress", "finalize", "status"],
        required=True,
    )
    parser.add_argument("--dimension", type=int)
    parser.add_argument("--model", choices=["FC_HVI", "DEEPSETS_HVI"])
    args = parser.parse_args(argv)
    if args.phase == "preflight":
        result: Any = run_preflight()
    elif args.phase == "canonical":
        if args.dimension is not None or args.model is not None:
            if args.dimension is None or args.model is None:
                raise SystemExit("--dimension and --model must be supplied together")
            result = train_hd3_job(args.dimension, args.model, run_class="canonical")
        else:
            result = run_canonical()
    elif args.phase == "stress":
        if args.dimension is not None or args.model is not None:
            if args.dimension is None or args.model is None:
                raise SystemExit("--dimension and --model must be supplied together")
            result = train_hd3_job(args.dimension, args.model, run_class="stress")
        else:
            result = run_stress()
    elif args.phase == "finalize":
        result = run_finalize()
    else:
        result = status_snapshot()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
