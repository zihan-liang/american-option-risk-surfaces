"""Conditional HD2 candidates activated only by frozen diagnostic triggers.

This module keeps the mandatory HD2 screen causal and small.  CAGrad is run
only after persistent measured gradient conflict, the 8-probe variant only
after a material d=5 Hutchinson-gradient discrepancy, and ReLoBRaLo only when
no fixed-schedule candidate passes and the global budget permits it.
"""
from __future__ import annotations

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

import hd2_hvi_objective_stabilization as hd2
from hd_pipeline.gradient_diagnostics import flatten_grads
from hd_pipeline.objective_optimizers import ReLoBRaLoState, cagrad_combine
from hd_pipeline.runtime import (
    append_jsonl,
    atomic_json,
    capture_rng_state,
    load_checkpoint_verified,
    read_jsonl_recover,
    restore_rng_state,
    save_checkpoint_atomic,
    sha256_file,
    verified_completed_status,
)
from hd1_windows_high_dimensional_pinn_48h import git_commit
from models.high_dimensional_pinn import initialize_model

CONDITIONAL_RESULT_PATH = hd2.RESULT_DIR / "conditional_methods.json"
CONDITIONAL_CANDIDATES = (
    "HVI-CAGrad",
    "HVI-HD1-anchor-8probe",
    "HVI-ReLoBRaLo",
)


def _conditional_spec(candidate: str, cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        return dict(cfg["conditional_methods"][candidate])
    except KeyError as exc:
        raise ValueError(f"unknown conditional candidate: {candidate}") from exc


def build_conditional_plan(
    cfg: dict[str, Any],
    *,
    mandatory_decision: dict[str, Any],
    conflict_trigger: bool,
    eight_probe_trigger: bool,
    relobralo_budget_safe: bool,
) -> list[str]:
    plan: list[str] = []
    if conflict_trigger:
        plan.append("HVI-CAGrad")
    if eight_probe_trigger:
        plan.append("HVI-HD1-anchor-8probe")
    if mandatory_decision.get("selection_status") == "no_stabilized_candidate" and relobralo_budget_safe:
        plan.append("HVI-ReLoBRaLo")
    return plan


def resolved_objective_for_candidate(candidate: str, cfg: dict[str, Any]) -> dict[str, Any]:
    if candidate in cfg["candidate_settings"] and candidate != "SUP-HD2":
        settings = cfg["candidate_settings"][candidate]
        return {
            "candidate": candidate,
            "warm_fraction": float(settings["warm_start_steps"]) / float(cfg["total_steps"]),
            "final_physics_weight": float(settings["final_physics_weight"]),
            "fallback_objective": False,
            "optimizer_mode": "fixed",
            "hutchinson_probe_count": int(cfg["hutchinson_probe_count"]),
        }
    spec = _conditional_spec(candidate, cfg)
    resolved: dict[str, Any] = {
        "candidate": candidate,
        "warm_fraction": float(spec["warm_start_steps"]) / float(cfg["total_steps"]),
        "final_physics_weight": float(spec["final_physics_weight"]),
        "fallback_objective": False,
        "optimizer_mode": str(spec["optimizer_mode"]),
    }
    if int(spec.get("hutchinson_probe_count", cfg["hutchinson_probe_count"])) != int(cfg["hutchinson_probe_count"]):
        resolved["hutchinson_probe_count"] = int(spec["hutchinson_probe_count"])
    if resolved["optimizer_mode"] == "cagrad":
        resolved["cagrad_alpha"] = float(spec["cagrad_alpha"])
        resolved["cagrad_rescale"] = int(spec["cagrad_rescale"])
    elif resolved["optimizer_mode"] == "relobralo":
        resolved["relobralo_temperature"] = float(spec["relobralo_temperature"])
        resolved["relobralo_alpha"] = float(spec["relobralo_alpha"])
        resolved["relobralo_rho"] = float(spec["relobralo_rho"])
    return resolved


def _fallback_objective(cfg: dict[str, Any]) -> dict[str, Any]:
    fallback = dict(cfg["fallback_objective"])
    fallback.update(
        {
            "fallback_objective": True,
            "optimizer_mode": "fixed",
            "hutchinson_probe_count": int(cfg["hutchinson_probe_count"]),
        }
    )
    return fallback


def _candidate_rows_with_aliases(
    metrics_rows: list[dict[str, Any]],
    candidate: str,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    by_d = {
        int(row["dimension"]): dict(row)
        for row in metrics_rows
        if str(row["candidate"]) == candidate
    }
    alias_info: dict[str, Any] = {}
    if candidate == "HVI-HD1-anchor-8probe" and 2 not in by_d:
        anchor = next(
            (
                dict(row)
                for row in metrics_rows
                if str(row["candidate"]) == "HVI-HD1-anchor" and int(row["dimension"]) == 2
            ),
            None,
        )
        if anchor is not None:
            anchor["candidate"] = candidate
            anchor["reused_anchor_d2"] = True
            by_d[2] = anchor
            alias_info["reused_anchor_d2"] = True
    return by_d, alias_info


def select_with_conditionals(
    metrics_rows: list[dict[str, Any]],
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = hd2.load_config() if cfg is None else cfg
    dims = [int(v) for v in cfg["dimensions"]]
    baseline = {
        int(row["dimension"]): row
        for row in metrics_rows
        if str(row["candidate"]) == "SUP-HD2"
    }
    if sorted(baseline) != sorted(dims):
        raise ValueError("SUP-HD2 validation baseline is incomplete")
    candidates = sorted(
        {
            str(row["candidate"])
            for row in metrics_rows
            if str(row["candidate"]) != "SUP-HD2"
        }
        | {
            "HVI-HD1-anchor-8probe"
            if any(str(row["candidate"]) == "HVI-HD1-anchor-8probe" for row in metrics_rows)
            else ""
        }
    )
    candidates = [candidate for candidate in candidates if candidate]
    gates = cfg["selection_gates"]
    audit: dict[str, Any] = {}
    eligible: list[tuple[float, int, str]] = []
    fixed_candidates = set(cfg["candidate_settings"]) - {"SUP-HD2"}
    for candidate in candidates:
        by_d, alias_info = _candidate_rows_with_aliases(metrics_rows, candidate)
        if sorted(by_d) != sorted(dims):
            audit[candidate] = {"eligible": False, "reason": "incomplete_dimensions"}
            continue
        dimension_audit: dict[str, Any] = {}
        physics_ratios: list[float] = []
        candidate_ok = True
        for d in dims:
            row, base = by_d[d], baseline[d]
            ratios = {
                "price_ratio": float(row["price_rmse"]) / max(float(base["price_rmse"]), 1e-15),
                "gamma_ratio": float(row["gamma_rmse"]) / max(float(base["gamma_rmse"]), 1e-15),
                "boundary_ratio": float(row["boundary_rmse"]) / max(float(base["boundary_rmse"]), 1e-15),
                "pde_p95_ratio": float(row["pde_p95"]) / max(float(base["pde_p95"]), 1e-15),
                "fb_p95_ratio": float(row["fb_p95"]) / max(float(base["fb_p95"]), 1e-15),
            }
            passed = bool(row.get("integrity", False)) and (
                ratios["price_ratio"] <= float(gates["price_ratio_max"])
                and ratios["gamma_ratio"] <= float(gates["gamma_ratio_max"])
                and ratios["boundary_ratio"] <= float(gates["boundary_ratio_max"])
                and ratios["pde_p95_ratio"] <= float(gates["pde_p95_ratio_max"])
                and ratios["fb_p95_ratio"] <= float(gates["fb_p95_ratio_max"])
            )
            candidate_ok = candidate_ok and passed
            physics_ratios.extend((ratios["pde_p95_ratio"], ratios["fb_p95_ratio"]))
            dimension_audit[str(d)] = {
                **ratios,
                "passes": passed,
                "reused_anchor_d2": bool(row.get("reused_anchor_d2", False)),
            }
        score = float(
            math.exp(
                sum(math.log(max(value, 1e-15)) for value in physics_ratios)
                / len(physics_ratios)
            )
        )
        audit[candidate] = {
            "eligible": candidate_ok,
            "physics_score": score,
            "dimensions": dimension_audit,
            **alias_info,
        }
        if candidate_ok:
            priority = 0 if candidate in fixed_candidates else 1
            eligible.append((score, priority, candidate))
    if not eligible:
        return {
            "selection_status": "no_stabilized_candidate",
            "eligible_candidates": [],
            "candidate_audit": audit,
            "resolved_objective": _fallback_objective(cfg),
            "diagnostic_only": True,
        }
    eligible.sort(key=lambda item: (round(item[0], 14), item[1], item[2]))
    selected = eligible[0][2]
    return {
        "selection_status": "stabilized_candidate_selected",
        "eligible_candidates": [item[2] for item in eligible],
        "candidate_audit": audit,
        "resolved_objective": resolved_objective_for_candidate(selected, cfg),
        "diagnostic_only": True,
    }


def _assign_flat_gradient(parameters: list[torch.nn.Parameter], flat: torch.Tensor) -> None:
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.grad = flat[offset : offset + count].reshape_as(parameter).to(parameter).clone()
        offset += count
    if offset != flat.numel():
        raise RuntimeError("flat_gradient_size_mismatch")


def _optimizer_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    supervised: torch.Tensor,
    physics: torch.Tensor,
    scalar_weight: float,
    spec: dict[str, Any],
    relobralo: ReLoBRaLoState | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    mode = str(spec["optimizer_mode"])
    optimizer.zero_grad(set_to_none=True)
    if scalar_weight <= 0.0:
        supervised.backward()
        optimizer.step()
        return supervised.detach(), {"optimizer_mode": "supervised"}
    if mode == "fixed":
        total = supervised + scalar_weight * physics
        total.backward()
        optimizer.step()
        return total.detach(), {"optimizer_mode": "fixed"}
    if mode == "cagrad":
        params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        sup_grads = torch.autograd.grad(
            supervised, params, retain_graph=True, create_graph=False, allow_unused=True
        )
        phys_grads = torch.autograd.grad(
            scalar_weight * physics,
            params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        combined, meta = cagrad_combine(
            flatten_grads(sup_grads, params),
            flatten_grads(phys_grads, params),
            alpha=float(spec["cagrad_alpha"]),
            rescale=int(spec["cagrad_rescale"]),
        )
        _assign_flat_gradient(params, combined)
        optimizer.step()
        return (supervised.detach() + scalar_weight * physics.detach()), {
            "optimizer_mode": "cagrad",
            **meta,
        }
    if mode == "relobralo":
        if relobralo is None:
            raise RuntimeError("missing_relobralo_state")
        weights = relobralo.update(
            [float(supervised.detach().cpu()), float(physics.detach().cpu())]
        )
        total = weights[0] * supervised + weights[1] * physics
        total.backward()
        optimizer.step()
        return total.detach(), {
            "optimizer_mode": "relobralo",
            "supervised_weight": float(weights[0]),
            "physics_weight": float(weights[1]),
        }
    raise ValueError(f"unsupported conditional optimizer mode: {mode}")


def _save_checkpoint(
    *,
    path: Path,
    candidate: str,
    d: int,
    completed_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_generator: torch.Generator,
    physics_consumed: int,
    batch_size: int,
    elapsed_seconds: float,
    relobralo: ReLoBRaLoState | None,
) -> None:
    save_checkpoint_atomic(
        path,
        {
            "implementation_sha": git_commit(),
            "config_hash": hd2.config_hash(),
            "job_id": hd2._job_id(d, candidate),
            "dimension": d,
            "candidate": candidate,
            "completed_step": completed_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "batch_generator_state": batch_generator.get_state(),
            "physics_consumed": physics_consumed,
            "batch_size": batch_size,
            "elapsed_training_seconds": elapsed_seconds,
            "objective_state": relobralo.state_dict() if relobralo is not None else None,
            "rng_state": capture_rng_state(),
        },
    )


def _train_once(
    d: int,
    candidate: str,
    cfg: dict[str, Any],
    *,
    batch_size: int,
    resume: bool,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required:conditional HD2 training")
    spec = _conditional_spec(candidate, cfg)
    if candidate == "HVI-HD1-anchor-8probe" and d != 5:
        raise ValueError("8-probe candidate trains only at d=5; d=2 reuses the exact anchor")
    device = torch.device("cuda:0")
    oracle = hd2.load_oracle(d)
    x_train, tau_train, y_train = hd2._fixed_supervised_data(
        d, cfg, oracle, "train", int(cfg["supervised_train_points"]), device
    )
    total_steps = int(cfg["total_steps"])
    warm_start = int(spec["warm_start_steps"])
    final_weight = float(spec["final_physics_weight"])
    probe_count = int(spec.get("hutchinson_probe_count", cfg["hutchinson_probe_count"]))
    mode = str(spec["optimizer_mode"])
    model = initialize_model(int(cfg["seed"]), device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    batch_generator = torch.Generator(device="cpu")
    batch_generator.manual_seed(int(cfg["seed"]) + d)
    physics_engine = torch.quasirandom.SobolEngine(
        d + 1, scramble=True, seed=int(cfg["sobol_seeds"]["physics"]) + d
    )
    checkpoint_path = hd2._checkpoint_path(d, candidate)
    status_path = hd2._status_path(d, candidate)
    job_dir = hd2._job_dir(d, candidate)
    job_dir.mkdir(parents=True, exist_ok=True)
    completed_step, physics_consumed, elapsed_before = 0, 0, 0.0
    relobralo = (
        ReLoBRaLoState(
            task_count=2,
            temperature=float(spec["relobralo_temperature"]),
            alpha=float(spec["relobralo_alpha"]),
            rho=float(spec["relobralo_rho"]),
        )
        if mode == "relobralo"
        else None
    )
    if resume and checkpoint_path.exists():
        checkpoint = load_checkpoint_verified(
            checkpoint_path,
            implementation_sha=git_commit(),
            config_hash=hd2.config_hash(),
            job_id=hd2._job_id(d, candidate),
        )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        batch_generator.set_state(checkpoint["batch_generator_state"])
        completed_step = int(checkpoint["completed_step"])
        physics_consumed = int(checkpoint["physics_consumed"])
        elapsed_before = float(checkpoint.get("elapsed_training_seconds", 0.0))
        restore_rng_state(checkpoint["rng_state"])
        if relobralo is not None and checkpoint.get("objective_state"):
            relobralo = ReLoBRaLoState.from_state_dict(checkpoint["objective_state"])
    if physics_consumed:
        physics_engine.fast_forward(physics_consumed)
    atomic_json(
        status_path,
        {
            "status": "running",
            "implementation_sha": git_commit(),
            "config_hash": hd2.config_hash(),
            "job_id": hd2._job_id(d, candidate),
            "dimension": d,
            "candidate": candidate,
            "completed_step": completed_step,
            "target_step": total_steps,
            "batch_size": batch_size,
        },
    )
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for step in range(completed_step, total_steps):
            if mode == "relobralo":
                _, epsilon = hd2.curriculum(
                    step,
                    total_steps,
                    warm_start,
                    1.0,
                    epsilon_initial=float(cfg["fb_epsilon_initial"]),
                    epsilon_final=float(cfg["fb_epsilon_final"]),
                )
                scalar_weight = 0.0 if step < warm_start else 1.0
            else:
                scalar_weight, epsilon = hd2.curriculum(
                    step,
                    total_steps,
                    warm_start,
                    final_weight,
                    epsilon_initial=float(cfg["fb_epsilon_initial"]),
                    epsilon_final=float(cfg["fb_epsilon_final"]),
                )
            index = torch.randint(
                0, len(x_train), (batch_size,), generator=batch_generator
            ).to(device)
            prediction = model(x_train[index], tau_train[index])
            supervised = (prediction - y_train[index]).square().mean()
            physics = torch.zeros((), device=device)
            if step >= warm_start and scalar_weight > 0.0:
                unit = physics_engine.draw(batch_size).to(device=device, dtype=torch.float32)
                physics_consumed += batch_size
                lo, hi = math.log(0.5), math.log(1.5)
                xp = (lo + (hi - lo) * unit[:, :d]).requires_grad_(True)
                tp = (0.0025 + (1.0 - 0.0025) * unit[:, d]).requires_grad_(True)
                physics, _ = hd2._physics_loss_from_points(
                    model,
                    xp,
                    tp,
                    d,
                    cfg,
                    epsilon=epsilon,
                    probe_count=probe_count,
                )
            total, update_meta = _optimizer_step(
                model=model,
                optimizer=optimizer,
                supervised=supervised,
                physics=physics,
                scalar_weight=scalar_weight,
                spec=spec,
                relobralo=relobralo,
            )
            completed_step = step + 1
            peak = torch.cuda.max_memory_allocated(device) / 1024**3
            if peak >= float(cfg["max_cuda_allocation_gib"]):
                raise RuntimeError(f"cuda_memory_gate_failed:{peak:.4f}GiB")
            if completed_step == 1 or completed_step % int(cfg["train_log_interval"]) == 0 or completed_step == total_steps:
                append_jsonl(
                    job_dir / "training_history.jsonl",
                    {
                        "dimension": d,
                        "candidate": candidate,
                        "step": completed_step,
                        "supervised_mse": float(supervised.detach().cpu()),
                        "fb_loss": float(physics.detach().cpu()),
                        "total_loss": float(total.detach().cpu()),
                        "scalar_weight": scalar_weight,
                        "epsilon": epsilon,
                        "probe_count": probe_count,
                        "batch_size": batch_size,
                        "physics_consumed": physics_consumed,
                        "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                        "peak_cuda_gib": peak,
                        **update_meta,
                    },
                )
            if completed_step % int(cfg["validation_interval"]) == 0:
                x_val, tau_val, y_val = hd2._fixed_supervised_data(
                    d, cfg, oracle, "validation", int(cfg["validation_points"]), device
                )
                append_jsonl(
                    job_dir / "validation_history.jsonl",
                    {
                        "dimension": d,
                        "candidate": candidate,
                        "step": completed_step,
                        "price_rmse": hd2._validation_price_rmse(model, x_val, tau_val, y_val),
                    },
                )
            if completed_step % int(cfg["checkpoint_interval"]) == 0 or completed_step == total_steps:
                _save_checkpoint(
                    path=checkpoint_path,
                    candidate=candidate,
                    d=d,
                    completed_step=completed_step,
                    model=model,
                    optimizer=optimizer,
                    batch_generator=batch_generator,
                    physics_consumed=physics_consumed,
                    batch_size=batch_size,
                    elapsed_seconds=elapsed_before + time.perf_counter() - started,
                    relobralo=relobralo,
                )
    except KeyboardInterrupt:
        atomic_json(
            status_path,
            {
                "status": "interrupted",
                "implementation_sha": git_commit(),
                "config_hash": hd2.config_hash(),
                "job_id": hd2._job_id(d, candidate),
                "completed_step": completed_step,
                "checkpoint": str(checkpoint_path),
            },
        )
        raise
    elapsed = elapsed_before + time.perf_counter() - started
    metrics = hd2.evaluate_model_metrics(
        d=d,
        candidate=candidate,
        model=model,
        cfg=cfg,
        oracle=oracle,
        split="validation",
        n=int(cfg["validation_points"]),
        training_seconds=elapsed,
        batch_size=batch_size,
    )
    metrics["training_probe_count"] = probe_count
    metrics["optimizer_mode"] = mode
    metrics_path = job_dir / "validation_metrics.json"
    atomic_json(metrics_path, metrics)
    atomic_json(
        status_path,
        {
            "status": "completed",
            "implementation_sha": git_commit(),
            "config_hash": hd2.config_hash(),
            "job_id": hd2._job_id(d, candidate),
            "dimension": d,
            "candidate": candidate,
            "completed_step": total_steps,
            "target_step": total_steps,
            "batch_size": batch_size,
            "artifacts": {str(metrics_path): sha256_file(metrics_path)},
        },
    )
    return metrics


def train_conditional_candidate(
    d: int,
    candidate: str,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    cfg = hd2.load_config()
    _conditional_spec(candidate, cfg)
    if candidate == "HVI-HD1-anchor-8probe" and d == 2:
        anchor = hd2._job_dir(2, "HVI-HD1-anchor") / "validation_metrics.json"
        if not anchor.exists():
            raise RuntimeError("d2_anchor_required_for_8probe_alias")
        row = json.loads(anchor.read_text(encoding="utf-8"))
        row["candidate"] = candidate
        row["reused_anchor_d2"] = True
        return row
    status_path = hd2._status_path(d, candidate)
    if verified_completed_status(
        status_path,
        implementation_sha=git_commit(),
        config_hash=hd2.config_hash(),
    ):
        return json.loads(
            (hd2._job_dir(d, candidate) / "validation_metrics.json").read_text(encoding="utf-8")
        )
    batches = [
        int(cfg["initial_batch_size"]),
        *[int(value) for value in cfg["oom_batch_fallbacks"]],
    ]
    checkpoint = hd2._checkpoint_path(d, candidate)
    if resume and checkpoint.exists():
        saved = load_checkpoint_verified(
            checkpoint,
            implementation_sha=git_commit(),
            config_hash=hd2.config_hash(),
            job_id=hd2._job_id(d, candidate),
        )
        batch = int(saved.get("batch_size", batches[0]))
        if batch in batches:
            batches = batches[batches.index(batch) :]
    for batch_size in batches:
        try:
            return _train_once(d, candidate, cfg, batch_size=batch_size, resume=resume)
        except torch.cuda.OutOfMemoryError:
            append_jsonl(
                hd2._job_dir(d, candidate) / "oom_attempts.jsonl",
                {"batch_size": batch_size, "error": "cuda_oom", "time": time.time()},
            )
            torch.cuda.empty_cache()
            resume = True
    raise RuntimeError("oom_blocked:conditional_batches_exhausted")


def _trigger_evidence(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool, bool]:
    conflict_rows: list[dict[str, Any]] = []
    for d in cfg["dimensions"]:
        path = hd2._job_dir(int(d), "HVI-HD1-anchor") / "gradient_diagnostics.jsonl"
        if path.exists():
            row = hd2.persistent_conflict(
                read_jsonl_recover(path),
                warm_start_steps=int(cfg["candidate_settings"]["HVI-HD1-anchor"]["warm_start_steps"]),
                median_threshold=float(cfg["persistent_conflict_median_cosine"]),
            )
            row["dimension"] = int(d)
            conflict_rows.append(row)
    hutch_rows = read_jsonl_recover(
        hd2._job_dir(5, "HVI-HD1-anchor") / "hutchinson_gradient_validation.jsonl"
    )
    return (
        conflict_rows,
        hutch_rows,
        any(bool(row.get("triggered")) for row in conflict_rows),
        any(bool(row.get("conditional_eight_probe_trigger")) for row in hutch_rows),
    )


def run_triggered_conditionals(*, relobralo_budget_safe: bool) -> dict[str, Any]:
    cfg = hd2.load_config()
    mandatory_rows = hd2.collect_validation_metrics()
    mandatory_decision = hd2.select_hd2_candidate(mandatory_rows, cfg)
    conflict_rows, hutch_rows, conflict, eight_probe = _trigger_evidence(cfg)
    plan = build_conditional_plan(
        cfg,
        mandatory_decision=mandatory_decision,
        conflict_trigger=conflict,
        eight_probe_trigger=eight_probe,
        relobralo_budget_safe=relobralo_budget_safe,
    )
    metrics: list[dict[str, Any]] = []
    for candidate in plan:
        dimensions = [5] if candidate == "HVI-HD1-anchor-8probe" else [2, 5]
        for d in dimensions:
            metrics.append(train_conditional_candidate(d, candidate))
    payload = {
        "status": "completed",
        "mandatory_selection_status": mandatory_decision["selection_status"],
        "persistent_conflict": conflict_rows,
        "hutchinson_diagnostics": hutch_rows,
        "cagrad_triggered": conflict,
        "eight_probe_triggered": eight_probe,
        "relobralo_budget_safe": bool(relobralo_budget_safe),
        "executed_candidates": plan,
        "validation_metrics": metrics,
    }
    atomic_json(CONDITIONAL_RESULT_PATH, payload)
    return payload


def collect_all_validation_metrics() -> list[dict[str, Any]]:
    rows = hd2.collect_validation_metrics()
    for candidate in CONDITIONAL_CANDIDATES:
        dimensions = [5] if candidate == "HVI-HD1-anchor-8probe" else [2, 5]
        for d in dimensions:
            path = hd2._job_dir(d, candidate) / "validation_metrics.json"
            if path.exists():
                rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def run_select_with_conditionals() -> dict[str, Any]:
    cfg = hd2.load_config()
    rows = collect_all_validation_metrics()
    decision = select_with_conditionals(rows, cfg)
    conflict_rows, hutch_rows, conflict, eight_probe = _trigger_evidence(cfg)
    conditional = (
        json.loads(CONDITIONAL_RESULT_PATH.read_text(encoding="utf-8"))
        if CONDITIONAL_RESULT_PATH.exists()
        else {"executed_candidates": []}
    )
    decision.update(
        {
            "persistent_conflict": conflict_rows,
            "hutchinson_diagnostics": hutch_rows,
            "cagrad_triggered": conflict,
            "eight_probe_triggered": eight_probe,
            "conditional_candidates_executed": conditional.get("executed_candidates", []),
            "implementation_sha": git_commit(),
            "config_hash": hd2.config_hash(),
        }
    )
    atomic_json(hd2.SELECTION_PATH, decision)
    return decision
