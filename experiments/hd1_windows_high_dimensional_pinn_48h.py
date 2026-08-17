"""Frozen HD1 Windows high-dimensional PINN pilot.

Implementation-only module: canonical and smoke training are CUDA-only.  This
module never accesses the campaign blind holdout and never fabricates canonical
metrics.  The geometric-basket control reduces to a one-dimensional American
put oracle; it is not a claim about general high-dimensional American options.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.high_dimensional_pinn import initialize_model, parameter_count
from physics.high_dimensional_black_scholes import (
    black_scholes_vi_residual,
    covariance_matrix,
    effective_geometric_basket_parameters,
    fischer_burmeister,
    full_hessian_contraction,
    geometric_put_payoff,
    hutchinson_hessian_contraction,
)

CONFIG_PATH = ROOT / "configs" / "hd1_windows_high_dimensional_pinn_48h.json"
RESULT_DIR = ROOT / "results" / "hd1_windows_high_dimensional_pinn_48h"
STATUS_DIR = RESULT_DIR / "status"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"
REQUIRED_OUTPUTS = {
    "environment_manifest.json", "config_manifest.json", "run_metrics.csv",
    "dimension_summary.csv", "training_history.csv", "hessian_validation.csv",
    "decision.json", "artifact_hashes.json",
}


def load_config() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    required = {
        "seed", "dimensions", "models", "contract", "adam_steps", "warm_start_steps",
        "initial_batch_size", "oom_batch_fallbacks", "canonical_order", "max_run_hours",
        "max_gpu_hours", "max_cuda_allocation_gib", "hutchinson_probe_count",
        "hutchinson_probe_seed", "oracle_grids", "oracle_rmse_max", "sobol_seeds",
        "supervised_train_points", "validation_points", "test_points", "fallback_adam_steps",
    }
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"missing frozen config fields: {missing}")
    if cfg["seed"] != 101 or cfg["dimensions"] != [1, 2, 5, 10] or cfg["models"] != ["SUP", "HVI"]:
        raise ValueError("HD1 seed/dimensions/models are frozen")
    if cfg["contract"] != {"K": 1.0, "T": 1.0, "r": 0.05, "q": 0.02, "sigma": 0.25, "rho": 0.30}:
        raise ValueError("HD1 contract is frozen")
    if cfg["adam_steps"] != 10000 or cfg["fallback_adam_steps"] != 6000 or cfg["warm_start_steps"] != 2000:
        raise ValueError("HD1 training schedule is frozen")
    if cfg["initial_batch_size"] != 2048 or cfg["oom_batch_fallbacks"] != [1024, 512]:
        raise ValueError("HD1 batch/OOM schedule is frozen")
    if cfg["hutchinson_probe_count"] != 2 or cfg["hutchinson_probe_seed"] != 101:
        raise ValueError("HD1 Hutchinson estimator is frozen")
    if cfg["oracle_grids"] != [[256, 256], [512, 512]] or cfg["oracle_rmse_max"] != 1e-4:
        raise ValueError("HD1 oracle refinement gate is frozen")
    if cfg["max_run_hours"] != 3.0 or cfg["max_gpu_hours"] != 24.0 or cfg["max_cuda_allocation_gib"] != 7.0:
        raise ValueError("HD1 resource budgets are frozen")
    if cfg["canonical_order"] != ["d1_SUP", "d1_HVI", "d2_SUP", "d2_HVI", "d5_SUP", "d5_HVI", "d10_SUP", "d10_HVI"]:
        raise ValueError("canonical order is frozen")


def curriculum(step: int, total_steps: int) -> tuple[float, float]:
    """2000-step SUP warm start, then frozen weight/epsilon annealing."""
    if total_steps < 2001 or not 0 <= step < total_steps:
        raise ValueError("invalid training step")
    if step < 2000:
        return 0.0, 1.0e-2
    alpha = (step - 2000) / max(total_steps - 2001, 1)
    weight = 0.1 * alpha
    epsilon = 10.0 ** (math.log10(1e-2) + alpha * (math.log10(1e-4) - math.log10(1e-2)))
    return weight, epsilon


def classify_decision(metrics: dict) -> dict:
    """Apply the frozen diagnostic gate; candidate_supported is impossible."""
    gates = {
        "finite_metrics_and_hashes": bool(metrics["finite_metrics"] and metrics["hashes_valid"]),
        "obstacle_violation": float(metrics["obstacle_max"]) <= 1e-6,
        "price_nonregression": float(metrics["hvi_price_rmse"]) <= 1.05 * float(metrics["sup_price_rmse"]),
        "pde_nonregression": float(metrics["hvi_pde_p95"]) <= 1.05 * float(metrics["sup_pde_p95"]),
        "gamma_nonregression": float(metrics["hvi_gamma_rmse"]) <= 1.10 * float(metrics["sup_gamma_rmse"]),
        "boundary_nonregression": float(metrics["hvi_boundary_rmse"]) <= 1.10 * float(metrics["sup_boundary_rmse"]),
        "cuda_allocation": float(metrics["cuda_alloc_gib"]) < 7.0,
        "no_hidden_failures": not bool(metrics["hidden_failures"]),
    }
    improves = (
        float(metrics["hvi_price_rmse"]) <= 0.95 * float(metrics["sup_price_rmse"])
        or float(metrics["hvi_boundary_rmse"]) <= 0.95 * float(metrics["sup_boundary_rmse"])
    )
    classification = "pilot_promising" if all(gates.values()) and improves else "diagnostic_only"
    return {"classification": classification, "diagnostic_only": True, "candidate_supported": False, "gates": gates}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return os.environ.get("HD1_GIT_COMMIT", "unknown")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def environment_manifest() -> dict:
    import scipy
    cuda = bool(torch.cuda.is_available())
    driver = None
    if cuda:
        try:
            driver = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True, stderr=subprocess.DEVNULL,
            ).splitlines()[0].strip()
        except Exception:
            pass
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
        "cuda_available": cuda, "torch_cuda": torch.version.cuda, "cuda_driver": driver,
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory if cuda else None,
        "git_commit": git_commit(), "mixed_precision": False,
    }


def peak_cpu_memory_bytes() -> int:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]
        counters = PMC(); counters.cb = ctypes.sizeof(counters)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        if not ok:
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def sobol_points(n: int, d: int, seed: int, *, device="cpu", dtype=torch.float64):
    engine = torch.quasirandom.SobolEngine(d + 1, scramble=True, seed=int(seed))
    u = engine.draw(n).to(device=device, dtype=dtype)
    lo, hi = math.log(0.5), math.log(1.5)
    return lo + (hi - lo) * u[:, :d], 0.0025 + (1.0 - 0.0025) * u[:, d]


@dataclass
class Oracle:
    d: int
    sigma_g: float
    q_g: float
    spots: np.ndarray
    taus: np.ndarray
    values: np.ndarray
    ug: np.ndarray
    ugg: np.ndarray
    boundary: np.ndarray
    refinement_rmse: float
    seconds: float


def solve_oracle_grid(d: int, m: int, n: int, cfg: dict):
    from solvers.cn_psor import SolverConfig, solve_option
    c = cfg["contract"]
    sigma_g, q_g = effective_geometric_basket_parameters(d, c["sigma"], c["q"], c["rho"])
    return solve_option(SolverConfig(
        option_type="put", k=1.0, t=1.0, r=c["r"], q=q_g, sigma=sigma_g,
        s_max=4.0, m=m, n=n, omega=1.2, tol=1e-10, max_iter=20000,
        american=True, time_scheme="cn", lcp_solver="psor",
    ))


def interp2(spots, taus, values, g, tau):
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((taus, spots), values, bounds_error=False, fill_value=None)
    return np.asarray(f(np.column_stack([tau, g])), dtype=float)


def build_oracle(d: int, cfg: dict) -> Oracle:
    started = time.perf_counter()
    coarse = solve_oracle_grid(d, 256, 256, cfg)
    fine = solve_oracle_grid(d, 512, 512, cfg)
    ge = np.linspace(0.5, 1.5, 513); te = np.linspace(0.0025, 1.0, 129)
    gg, tt = np.meshgrid(ge, te)
    vc = interp2(coarse.spots, coarse.taus, coarse.values, gg.ravel(), tt.ravel())
    vf = interp2(fine.spots, fine.taus, fine.values, gg.ravel(), tt.ravel())
    refinement_rmse = float(np.sqrt(np.mean((vc - vf) ** 2)))
    if not np.isfinite(refinement_rmse) or refinement_rmse > cfg["oracle_rmse_max"]:
        raise RuntimeError(f"oracle_blocked:d={d}:refinement_rmse={refinement_rmse:.6g}")
    values = np.asarray(fine.values, dtype=float)
    ds = float(fine.spots[1] - fine.spots[0])
    ug = np.gradient(values, ds, axis=1, edge_order=2)
    ugg = np.gradient(ug, ds, axis=1, edge_order=2)
    payoff = np.maximum(1.0 - fine.spots, 0.0)
    boundary = np.zeros_like(fine.taus)
    for k, row in enumerate(values):
        ex = np.flatnonzero((fine.spots <= 1.0) & ((row - payoff) <= 1e-7))
        boundary[k] = fine.spots[ex[-1]] if ex.size else 0.0
    sigma_g, q_g = effective_geometric_basket_parameters(d, cfg["contract"]["sigma"], cfg["contract"]["q"], cfg["contract"]["rho"])
    return Oracle(d, sigma_g, q_g, fine.spots, fine.taus, values, ug, ugg, boundary, refinement_rmse, time.perf_counter() - started)


def save_oracle(o: Oracle) -> None:
    path = RESULT_DIR / "oracles" / f"d{o.d}.npz"; path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, d=o.d, sigma_g=o.sigma_g, q_g=o.q_g, spots=o.spots, taus=o.taus,
                        values=o.values, ug=o.ug, ugg=o.ugg, boundary=o.boundary,
                        refinement_rmse=o.refinement_rmse, seconds=o.seconds)


def load_oracle(d: int) -> Oracle:
    z = np.load(RESULT_DIR / "oracles" / f"d{d}.npz")
    return Oracle(d, float(z["sigma_g"]), float(z["q_g"]), z["spots"], z["taus"], z["values"],
                  z["ug"], z["ugg"], z["boundary"], float(z["refinement_rmse"]), float(z["seconds"]))


def oracle_labels(o: Oracle, x: torch.Tensor, tau: torch.Tensor, cfg: dict) -> dict:
    xn = x.detach().cpu().numpy(); tn = tau.detach().cpu().numpy(); d = o.d
    g = np.exp(xn.mean(axis=1))
    u = interp2(o.spots, o.taus, o.values, g, tn)
    ug = interp2(o.spots, o.taus, o.ug, g, tn)
    ugg = interp2(o.spots, o.taus, o.ugg, g, tn)
    boundary = np.interp(tn, o.taus, o.boundary)
    delta = ug[:, None] * (g[:, None] / (d * np.exp(xn)))
    covariance = covariance_matrix(d, cfg["contract"]["sigma"], cfg["contract"]["rho"], dtype=torch.float64).numpy()
    # dG/dx_i=G/d and d2G/dx_i dx_j=G/d^2 for every i,j.
    common = (ugg * g**2 + ug * g) / d**2
    h_log = np.broadcast_to(common[:, None, None], (len(g), d, d))
    contraction = np.einsum("ij,nij->n", covariance, h_log)
    return {"price": u, "delta": delta, "contraction": contraction, "g": g, "boundary": boundary}


def validate_hutchinson(cfg: dict, *, device: torch.device) -> dict:
    x = torch.randn(128, 2, device=device, dtype=torch.float32, requires_grad=True)
    tau = torch.full((128,), 0.5, device=device, dtype=torch.float32, requires_grad=True)
    model = initialize_model(cfg["seed"], device=device, dtype=torch.float32)
    value = model(x, tau)
    cov = covariance_matrix(2, cfg["contract"]["sigma"], cfg["contract"]["rho"], device=device, dtype=x.dtype)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter(); exact = full_hessian_contraction(value, x, cov); exact_seconds = time.perf_counter() - t0
    t1 = time.perf_counter(); estimate = hutchinson_hessian_contraction(value, x, cov, probe_count=2, probe_seed=101); estimator_seconds = time.perf_counter() - t1
    rel = float((torch.linalg.vector_norm(estimate - exact) / (torch.linalg.vector_norm(exact) + 1e-12)).detach().cpu())
    peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    row = {"dimension": 2, "probe_seed": 101, "probe_count": 2, "relative_error": rel,
           "full_seconds": exact_seconds, "hutchinson_seconds": estimator_seconds,
           "peak_cuda_gib": peak, "passed": rel <= 0.05}
    if not row["passed"]:
        raise RuntimeError(f"hessian_check_failed:{rel:.6g}")
    return row


def fixed_supervised_data(d: int, cfg: dict, o: Oracle, split: str, n: int, device):
    x, tau = sobol_points(n, d, cfg["sobol_seeds"][split])
    y = oracle_labels(o, x, tau, cfg)["price"]
    return x.to(device=device, dtype=torch.float32), tau.to(device=device, dtype=torch.float32), torch.tensor(y, device=device, dtype=torch.float32)


def train_model(d: int, variant: str, cfg: dict, *, steps: int, batch_size: int):
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required: smoke/canonical training must run on target Windows CUDA")
    device = torch.device("cuda:0"); o = load_oracle(d)
    x_train, tau_train, y_train = fixed_supervised_data(d, cfg, o, "train", cfg["supervised_train_points"], device)
    model = initialize_model(cfg["seed"], device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    generator = torch.Generator(device="cpu"); generator.manual_seed(cfg["seed"] + d)
    physics = torch.quasirandom.SobolEngine(d + 1, scramble=True, seed=cfg["sobol_seeds"]["physics"] + d)
    history = []; started = time.perf_counter(); points = 0
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(steps):
        idx = torch.randint(0, len(x_train), (batch_size,), generator=generator).to(device)
        pred = model(x_train[idx], tau_train[idx]); supervised = (pred - y_train[idx]).square().mean()
        weight, epsilon = curriculum(step, steps) if steps >= 2001 else (0.0, 1e-2)
        fb_loss = torch.zeros((), device=device)
        if variant == "HVI" and step >= 2000 and weight > 0:
            u = physics.draw(batch_size).to(device=device, dtype=torch.float32)
            lo, hi = math.log(0.5), math.log(1.5)
            xp = (lo + (hi - lo) * u[:, :d]).requires_grad_(True)
            tp = (0.0025 + (1.0 - 0.0025) * u[:, d]).requires_grad_(True)
            vp = model(xp, tp)
            residual, _ = black_scholes_vi_residual(
                vp, xp, tp, r=cfg["contract"]["r"], q=cfg["contract"]["q"],
                sigma=cfg["contract"]["sigma"], rho=cfg["contract"]["rho"],
                hessian_mode="full" if d <= 2 else "hutchinson", probe_count=2, probe_seed=101,
            )
            fb_loss = fischer_burmeister(vp - geometric_put_payoff(xp), residual, epsilon).square().mean()
        loss = supervised + weight * fb_loss
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); points += batch_size
        elapsed = time.perf_counter() - started; peak = torch.cuda.max_memory_allocated(device) / 1024**3
        if peak >= cfg["max_cuda_allocation_gib"]:
            raise RuntimeError(f"cuda_memory_gate_failed:{peak:.4f}GiB")
        if elapsed > cfg["max_run_hours"] * 3600:
            raise TimeoutError("resource_blocked:three-hour-run-cap")
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
            history.append({"dimension": d, "model": variant, "step": step + 1,
                            "supervised_mse": float(supervised.detach().cpu()), "fb_loss": float(fb_loss.detach().cpu()),
                            "fb_weight": weight, "epsilon": epsilon, "elapsed_seconds": elapsed,
                            "peak_cuda_gib": peak, "batch_size": batch_size})
    xv, tv, yv = fixed_supervised_data(d, cfg, o, "validation", cfg["validation_points"], device)
    with torch.no_grad(): validation_rmse = float(torch.sqrt((model(xv, tv) - yv).square().mean()).cpu())
    meta = {"training_seconds": time.perf_counter() - started, "peak_cuda_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "points_processed": points, "parameter_count": parameter_count(model), "batch_size": batch_size,
            "steps": steps, "validation_price_rmse": validation_rmse}
    return model, meta, history


def train_with_oom_fallback(d: int, variant: str, cfg: dict, *, steps: int):
    failures = []
    for batch in [cfg["initial_batch_size"]] + cfg["oom_batch_fallbacks"]:
        try:
            model, meta, history = train_model(d, variant, cfg, steps=steps, batch_size=batch)
            return model, meta, history, failures
        except torch.cuda.OutOfMemoryError:
            failures.append({"batch_size": batch, "error": "cuda_oom"}); torch.cuda.empty_cache()
    raise RuntimeError("oom_blocked:exhausted-2048-1024-512")


def f1_score(truth, pred) -> float:
    tp = np.sum(truth & pred); fp = np.sum(~truth & pred); fn = np.sum(truth & ~pred)
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def evaluate_model(d: int, variant: str, model, meta: dict, cfg: dict) -> dict:
    device = next(model.parameters()).device; o = load_oracle(d)
    xcpu, tcpu = sobol_points(cfg["test_points"], d, cfg["sobol_seeds"]["test"])
    labels = oracle_labels(o, xcpu, tcpu, cfg); evaluation_start = time.perf_counter()
    infer_start = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(xcpu), 512):
            model(xcpu[start:start+512].to(device=device, dtype=torch.float32), tcpu[start:start+512].to(device=device, dtype=torch.float32))
    torch.cuda.synchronize(device); inference_seconds = time.perf_counter() - infer_start
    derivative_start = time.perf_counter(); preds=[]; deltas=[]; contractions=[]; residuals=[]; gaps=[]
    for start in range(0, len(xcpu), 512):
        x = xcpu[start:start+512].to(device=device, dtype=torch.float32).requires_grad_(True)
        tau = tcpu[start:start+512].to(device=device, dtype=torch.float32).requires_grad_(True)
        v = model(x, tau); vx = torch.autograd.grad(v.sum(), x, create_graph=True, retain_graph=True)[0]
        residual, contraction = black_scholes_vi_residual(
            v, x, tau, r=cfg["contract"]["r"], q=cfg["contract"]["q"], sigma=cfg["contract"]["sigma"],
            rho=cfg["contract"]["rho"], hessian_mode="full" if d <= 2 else "hutchinson", probe_count=2, probe_seed=101,
        )
        preds.append(v.detach().cpu().numpy()); deltas.append((vx / torch.exp(x)).detach().cpu().numpy())
        contractions.append(contraction.detach().cpu().numpy()); residuals.append(residual.detach().cpu().numpy())
        gaps.append((v - geometric_put_payoff(x)).detach().cpu().numpy())
    torch.cuda.synchronize(device); derivative_seconds = time.perf_counter() - derivative_start
    pred=np.concatenate(preds); delta=np.concatenate(deltas); contraction=np.concatenate(contractions); residual=np.concatenate(residuals); gap=np.concatenate(gaps)
    price_err=pred-labels["price"]; delta_err=delta-labels["delta"]; gamma_err=contraction-labels["contraction"]
    phi=np.sqrt(gap**2+residual**2)-gap-residual
    continuation=labels["g"]>labels["boundary"]+1e-8; pde_abs=np.abs(residual[continuation]) if np.any(continuation) else np.abs(residual)
    true_ex=labels["g"]<=labels["boundary"]+1e-8; pred_ex=gap<=1e-5; tube=np.abs(labels["g"]-labels["boundary"])<=0.02
    taus=np.linspace(0.0025,1.0,64); gs=np.linspace(0.5,1.2,256); predicted_boundary=[]; true_boundary=[]
    with torch.no_grad():
        for t in taus:
            xd=torch.log(torch.tensor(gs,device=device,dtype=torch.float32)).reshape(-1,1).repeat(1,d); td=torch.full((len(gs),),float(t),device=device)
            vd=model(xd,td).cpu().numpy(); payoff=np.maximum(1-gs,0); ex=np.flatnonzero((gs<=1)&((vd-payoff)<=1e-5))
            predicted_boundary.append(gs[ex[-1]] if ex.size else 0.0); true_boundary.append(np.interp(t,o.taus,o.boundary))
    boundary_rmse=float(np.sqrt(np.mean((np.asarray(predicted_boundary)-np.asarray(true_boundary))**2)))
    sensitivity=max(float(np.mean((gap<=t)!=(gap<=1e-5))) for t in (5e-6,2e-5))
    return {
        "dimension": d, "model": variant,
        "normalized_price_rmse": float(np.sqrt(np.mean(price_err**2))), "normalized_price_mae": float(np.mean(np.abs(price_err))),
        "normalized_price_p95": float(np.quantile(np.abs(price_err),.95)), "normalized_price_max": float(np.max(np.abs(price_err))),
        "delta_rmse": float(np.sqrt(np.mean(delta_err**2))), "delta_max": float(np.max(np.abs(delta_err))),
        "gamma_contraction_rmse": float(np.sqrt(np.mean(gamma_err**2))),
        "obstacle_violation_max": float(np.max(np.maximum(-gap,0))), "obstacle_violation_rate": float(np.mean(gap < -1e-6)),
        "fb_rms": float(np.sqrt(np.mean(phi**2))), "fb_p95": float(np.quantile(np.abs(phi),.95)), "fb_max": float(np.max(np.abs(phi))),
        "trusted_pde_rms": float(np.sqrt(np.mean(pde_abs**2))), "trusted_pde_p95": float(np.quantile(pde_abs,.95)),
        "exercise_f1": f1_score(true_ex,pred_ex), "boundary_tube_f1": f1_score(true_ex[tube],pred_ex[tube]) if np.any(tube) else 0.0,
        "boundary_rmse": boundary_rmse, "boundary_sensitivity": sensitivity, "training_seconds": meta["training_seconds"],
        "inference_seconds": inference_seconds, "derivative_seconds": derivative_seconds, "oracle_seconds": o.seconds,
        "evaluation_seconds": time.perf_counter()-evaluation_start, "peak_cpu_memory_bytes": peak_cpu_memory_bytes(),
        "peak_cuda_gib": meta["peak_cuda_gib"], "parameter_count": meta["parameter_count"], "points_processed": meta["points_processed"],
        "validation_price_rmse": meta["validation_price_rmse"], "batch_size": meta["batch_size"], "steps": meta["steps"], "status": "completed",
    }


def write_config_manifest(cfg: dict) -> None:
    atomic_json(RESULT_DIR / "config_manifest.json", {"config": cfg, "config_sha256": sha256_file(CONFIG_PATH), "git_commit": git_commit()})


def run_preflight() -> int:
    cfg=load_config(); RESULT_DIR.mkdir(parents=True,exist_ok=True); write_config_manifest(cfg)
    env=environment_manifest(); atomic_json(RESULT_DIR/"environment_manifest.json",env)
    if not env["cuda_available"]: raise RuntimeError("cuda_required:target-Windows-preflight")
    if env["gpu_total_memory_bytes"] is None or env["gpu_total_memory_bytes"] < 7*1024**3: raise RuntimeError("cuda_memory_blocked")
    rows=[]
    for d in cfg["dimensions"]:
        o=build_oracle(d,cfg); save_oracle(o); rows.append({"dimension":d,"oracle_refinement_rmse":o.refinement_rmse,"oracle_seconds":o.seconds})
    hrow=validate_hutchinson(cfg,device=torch.device("cuda:0"))
    with (RESULT_DIR/"hessian_validation.csv").open("w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=list(hrow)); w.writeheader(); w.writerow(hrow)
    atomic_json(STATUS_DIR/"preflight.json",{"status":"passed","git_commit":git_commit(),"oracle":rows,"hessian":hrow})
    return 0


def run_smokes() -> int:
    cfg=load_config(); p=STATUS_DIR/"preflight.json"
    if not p.exists() or json.loads(p.read_text())["status"]!="passed": raise RuntimeError("preflight_not_passed")
    rows=[]
    for d in (1,2):
        _,meta,_,oom=train_with_oom_fallback(d,"HVI",cfg,steps=500)
        rows.append({"dimension":d,"model":"HVI","steps":500,"training_seconds":meta["training_seconds"],"peak_cuda_gib":meta["peak_cuda_gib"],"batch_size":meta["batch_size"],"oom_attempts":oom,"status":"passed"})
    projected=max(r["training_seconds"] for r in rows)*(cfg["adam_steps"]/500)*8/3600.0
    steps=cfg["fallback_adam_steps"] if projected>cfg["max_gpu_hours"] else cfg["adam_steps"]
    atomic_json(STATUS_DIR/"smoke.json",{"status":"passed","smokes":rows,"projected_gpu_hours":projected,"canonical_steps":steps,"git_commit":git_commit()})
    return 0


def job_status_path(job: str) -> Path: return STATUS_DIR / f"canonical_{job}.json"


def verified_job(job: str) -> bool:
    p=job_status_path(job)
    if not p.exists(): return False
    try: data=json.loads(p.read_text())
    except Exception: return False
    if data.get("status")!="completed" or data.get("git_commit")!=git_commit(): return False
    artifacts=data.get("artifacts",{})
    return bool(artifacts) and all((ROOT/rel).exists() and sha256_file(ROOT/rel)==digest for rel,digest in artifacts.items())


def run_canonical_job(job: str) -> dict:
    cfg=load_config(); smoke_path=STATUS_DIR/"smoke.json"
    if not smoke_path.exists(): raise RuntimeError("smoke_not_passed")
    smoke=json.loads(smoke_path.read_text())
    if smoke.get("status")!="passed" or smoke.get("git_commit")!=git_commit(): raise RuntimeError("smoke_not_passed_or_stale")
    if job not in cfg["canonical_order"]: raise ValueError("unknown canonical job")
    if verified_job(job): return {"status":"skipped_verified","job":job}
    d=int(job.split("_")[0][1:]); variant=job.split("_")[1]
    model,meta,history,oom=train_with_oom_fallback(d,variant,cfg,steps=int(smoke["canonical_steps"]))
    metrics=evaluate_model(d,variant,model,meta,cfg); metrics["oom_attempt_count"]=len(oom)
    ckpt=CHECKPOINT_DIR/f"{job}.pt"; ckpt.parent.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":model.state_dict(),"meta":meta,"git_commit":git_commit()},ckpt)
    per_job=RESULT_DIR/"jobs"/f"{job}.json"; atomic_json(per_job,{"metrics":metrics,"oom_attempts":oom,"git_commit":git_commit()})
    for row in history: append_csv(RESULT_DIR/"training_history.csv",row)
    append_csv(RESULT_DIR/"run_metrics.csv",metrics)
    artifacts={str(ckpt.relative_to(ROOT)):sha256_file(ckpt),str(per_job.relative_to(ROOT)):sha256_file(per_job)}
    atomic_json(job_status_path(job),{"status":"completed","job":job,"artifacts":artifacts,"metrics":metrics,"git_commit":git_commit()})
    return {"status":"completed","job":job,"metrics":metrics,"artifacts":artifacts}


def read_metrics() -> list[dict]:
    p=RESULT_DIR/"run_metrics.csv"
    if not p.exists(): return []
    with p.open(newline="",encoding="utf-8") as fh: return list(csv.DictReader(fh))


def finalize_outputs() -> None:
    cfg=load_config(); rows=read_metrics(); by={(int(r["dimension"]),r["model"]):r for r in rows if r.get("status")=="completed"}
    summaries=[]; decisions=[]
    for d in cfg["dimensions"]:
        if (d,"SUP") not in by or (d,"HVI") not in by: continue
        s,h=by[(d,"SUP")],by[(d,"HVI")]
        finite_keys=["normalized_price_rmse","trusted_pde_p95","gamma_contraction_rmse","boundary_rmse"]
        hidden=not (verified_job(f"d{d}_SUP") and verified_job(f"d{d}_HVI"))
        m={"finite_metrics":all(np.isfinite(float(s[k])) and np.isfinite(float(h[k])) for k in finite_keys),"hashes_valid":not hidden,
           "obstacle_max":max(float(s["obstacle_violation_max"]),float(h["obstacle_violation_max"])),
           "hvi_price_rmse":float(h["normalized_price_rmse"]),"sup_price_rmse":float(s["normalized_price_rmse"]),
           "hvi_pde_p95":float(h["trusted_pde_p95"]),"sup_pde_p95":float(s["trusted_pde_p95"]),
           "hvi_gamma_rmse":float(h["gamma_contraction_rmse"]),"sup_gamma_rmse":float(s["gamma_contraction_rmse"]),
           "hvi_boundary_rmse":float(h["boundary_rmse"]),"sup_boundary_rmse":float(s["boundary_rmse"]),
           "cuda_alloc_gib":max(float(s["peak_cuda_gib"]),float(h["peak_cuda_gib"])),"hidden_failures":hidden}
        dec=classify_decision(m); decisions.append({"dimension":d,**dec})
        summaries.append({"dimension":d,"SUP_price_rmse":m["sup_price_rmse"],"HVI_price_rmse":m["hvi_price_rmse"],
                          "SUP_pde_p95":m["sup_pde_p95"],"HVI_pde_p95":m["hvi_pde_p95"],"SUP_gamma_rmse":m["sup_gamma_rmse"],"HVI_gamma_rmse":m["hvi_gamma_rmse"],
                          "SUP_boundary_rmse":m["sup_boundary_rmse"],"HVI_boundary_rmse":m["hvi_boundary_rmse"],
                          "SUP_oom_attempts":s.get("oom_attempt_count",0),"HVI_oom_attempts":h.get("oom_attempt_count",0),"classification":dec["classification"]})
    if summaries:
        with (RESULT_DIR/"dimension_summary.csv").open("w",newline="",encoding="utf-8") as fh:
            w=csv.DictWriter(fh,fieldnames=list(summaries[0])); w.writeheader(); w.writerows(summaries)
    atomic_json(RESULT_DIR/"decision.json",{"diagnostic_only":True,"candidate_supported":False,"seed":101,"dimensions":decisions})
    atomic_json(RESULT_DIR/"m02_bridge.json",{"status":"not_comparable","reason":"HD1 fixed geometric-basket contract/test distribution does not match the committed M02 experimental definition/distribution.","numerical_ranking_performed":False})
    hashes={}
    for name in sorted(REQUIRED_OUTPUTS-{"artifact_hashes.json"}):
        p=RESULT_DIR/name
        if p.exists(): hashes[name]=sha256_file(p)
    missing=sorted((REQUIRED_OUTPUTS-{"artifact_hashes.json"})-set(hashes))
    if missing: raise RuntimeError(f"required_outputs_missing:{missing}")
    atomic_json(RESULT_DIR/"artifact_hashes.json",{"sha256":hashes,"git_commit":git_commit()})


def status_payload() -> dict:
    cfg=load_config(); out={"git_commit":git_commit(),"preflight":{"status":"not_run"},"smoke":{"status":"not_run"},"canonical":{}}
    for phase in ("preflight","smoke"):
        p=STATUS_DIR/f"{phase}.json"
        if p.exists(): out[phase]=json.loads(p.read_text())
    for job in cfg["canonical_order"]:
        p=job_status_path(job); out["canonical"][job]=json.loads(p.read_text()) if p.exists() else {"status":"not_run"}
    return out


def cli(argv: Iterable[str] | None = None) -> int:
    p=argparse.ArgumentParser(); p.add_argument("--phase",choices=["preflight","smoke","canonical-job","finalize","status"],required=True); p.add_argument("--job")
    a=p.parse_args(argv)
    if a.phase=="preflight": return run_preflight()
    if a.phase=="smoke": return run_smokes()
    if a.phase=="canonical-job":
        if not a.job: p.error("--job required")
        print(json.dumps(run_canonical_job(a.job),indent=2,default=str)); return 0
    if a.phase=="finalize": finalize_outputs(); return 0
    print(json.dumps(status_payload(),indent=2,default=str)); return 0


if __name__ == "__main__":
    raise SystemExit(cli())
