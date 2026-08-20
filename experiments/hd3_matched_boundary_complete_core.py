"""Scientific primitives for the final boundary-complete matched HD3 diagnostic."""
from __future__ import annotations

import json
import math
import sys
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

from hd_pipeline.runtime import sha256_file
from physics.hd3_high_dimensional_sampling import sample_transverse_states
from physics.high_dimensional_black_scholes import (
    covariance_matrix,
    effective_geometric_basket_parameters,
)
from hd1_windows_high_dimensional_pinn_48h import (
    Oracle,
    interp2,
    peak_cpu_memory_bytes,
    solve_oracle_grid,
)
from hd3_matched_american_put_comparison import crr_american_put

CONFIG_PATH = ROOT / "configs" / "hd3_matched_american_put_boundary_complete.json"
RESULT_DIR = ROOT / "results" / "hd3_matched_american_put_boundary_complete"
OLD_RESULT_DIR = ROOT / "results" / "hd3_matched_american_put_comparison"
RUNTIME_DIR = RESULT_DIR / "runtime"
ORACLE_DIR = RUNTIME_DIR / "oracles"
VIS_DIR = RESULT_DIR / "visualization"

BOUNDARY_METHOD = "contiguous_low_G_exercise_to_continuation_linear_interpolation"
REDUCED_DELTA_DEFINITION = "Delta_G=dV(G,...,G,tau)/dG=sum_i dV/dS_i"
REDUCED_GAMMA_DEFINITION = "Gamma_G=d2V(G,...,G,tau)/dG2"
COMPONENT_DELTA_DEFINITION = "component_delta=dV/dS_i on sampled high-dimensional states"
GAMMA_CONTRACTION_DEFINITION = "covariance-weighted Hessian contraction in log coordinates"


def load_config() -> dict[str, Any]:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected_contract = {
        "option_type": "put", "K": 1.0, "T": 1.0, "r": 0.05,
        "q_g": 0.10, "sigma_g": 0.60,
    }
    if cfg.get("experiment_id") != "hd3_matched_american_put_boundary_complete":
        raise ValueError("wrong boundary-complete experiment id")
    if cfg.get("target_reduced_contract") != expected_contract or float(cfg.get("rho")) != 0.80:
        raise ValueError("boundary-complete financial contract is frozen")
    frozen = {
        "seed": 101, "canonical_dimensions": [10, 20, 50],
        "models": ["FC_HVI", "DEEPSETS_HVI"], "canonical_steps": 6000,
        "supervised_train_points": 20000, "validation_points": 8192,
        "test_points": 16384, "initial_batch_size": 512,
        "oom_batch_fallbacks": [256, 128], "hutchinson_probe_count": 2,
        "hutchinson_probe_seed": 101, "oracle_grids": [[256, 256], [512, 512]],
        "mixed_precision": False, "canonical_dtype": "float32", "oracle_dtype": "float64",
    }
    for key, value in frozen.items():
        if cfg.get(key) != value:
            raise ValueError(f"boundary-complete setting is frozen: {key}")
    if float(cfg.get("learning_rate")) != 0.001 or float(cfg.get("oracle_rmse_max")) != 1e-4:
        raise ValueError("boundary-complete numeric schedule is frozen")
    expected_objective = {
        "candidate": "HVI-warm-low", "warm_fraction": 0.5,
        "final_physics_weight": 0.01, "fallback_objective": True,
        "optimizer_mode": "fixed", "hutchinson_probe_count": 2,
    }
    if cfg.get("objective") != expected_objective:
        raise ValueError("boundary-complete objective is frozen")
    for key, value in {
        "single_seed": True, "diagnostic_only": True,
        "posthoc_matched_contract": True, "random_seed_robustness": False,
        "candidate_supported": False, "blind_holdout_access": False,
    }.items():
        if cfg.get(key) is not value:
            raise ValueError(f"boundary-complete claim flag is frozen: {key}")
    slo, shi = aggregate_spot_bounds(cfg)
    if abs(slo - 0.2) > 1e-14 or abs(shi - 1.5) > 1e-14:
        raise ValueError("boundary-complete aggregate domain is frozen")
    if tau_bounds(cfg) != (0.0025, 1.0):
        raise ValueError("boundary-complete time domain is frozen")
    b = cfg["boundary"]
    if (
        (float(b["s_min"]), float(b["s_max"]), int(b["s_points"])) != (0.2, 1.2, 801)
        or (float(b["tau_min"]), float(b["tau_max"]), int(b["tau_points"])) != (0.01, 1.0, 201)
        or float(b["primary_tolerance"]) != 1e-6
        or [float(v) for v in b["sensitivity_tolerances"]] != [1e-5, 1e-4]
    ):
        raise ValueError("boundary-complete boundary definition is frozen")
    if binomial_final_steps(cfg) != 2048:
        raise ValueError("final CRR curve must use 2048 steps")
    return cfg


def config_hash() -> str:
    return sha256_file(CONFIG_PATH)


def aggregate_bounds(cfg: dict[str, Any]) -> tuple[float, float]:
    return tuple(float(v) for v in cfg["aggregate_log_moneyness"])


def aggregate_spot_bounds(cfg: dict[str, Any]) -> tuple[float, float]:
    lo, hi = aggregate_bounds(cfg)
    return math.exp(lo), math.exp(hi)


def tau_bounds(cfg: dict[str, Any]) -> tuple[float, float]:
    return tuple(float(v) for v in cfg["tau"])


def training_domain_metadata(cfg: dict[str, Any]) -> dict[str, list[float]]:
    slo, shi = aggregate_spot_bounds(cfg)
    tlo, thi = tau_bounds(cfg)
    return {"G_over_K": [slo, shi], "tau_over_T": [tlo, thi]}


def parameter_mapping(d: int, cfg: dict[str, Any]) -> dict[str, float]:
    target, rho = cfg["target_reduced_contract"], float(cfg["rho"])
    sigma_g, q_g = float(target["sigma_g"]), float(target["q_g"])
    factor = rho + (1.0 - rho) / int(d)
    sigma = sigma_g / math.sqrt(factor)
    q = q_g - 0.5 * (sigma * sigma - sigma_g * sigma_g)
    check_sigma, check_q = effective_geometric_basket_parameters(int(d), sigma, q, rho)
    if abs(check_sigma - sigma_g) > 1e-12 or abs(check_q - q_g) > 1e-12:
        raise RuntimeError(f"matched_parameter_assertion_failed:d={d}")
    return {
        "dimension": int(d), "rho": rho, "asset_sigma": sigma, "asset_q": q,
        "effective_sigma_g": check_sigma, "effective_q_g": check_q,
    }


def dimension_config(d: int, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    import copy
    result = copy.deepcopy(cfg or load_config())
    mapping, target = parameter_mapping(int(d), result), result["target_reduced_contract"]
    result["contract"] = {
        "K": float(target["K"]), "T": float(target["T"]), "r": float(target["r"]),
        "q": float(mapping["asset_q"]), "sigma": float(mapping["asset_sigma"]),
        "rho": float(mapping["rho"]),
    }
    return result


def sample_states(
    split: str, *, n: int, d: int, cfg: dict[str, Any],
    device: str | torch.device = "cpu", dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if split not in {"train", "validation", "test", "structural"}:
        raise ValueError("unknown boundary-complete split")
    x, tau, m = sample_transverse_states(
        int(n), int(d), seed=int(cfg["sobol_seeds"][split]) + int(d),
        m_bounds=aggregate_bounds(cfg), tau_bounds=tau_bounds(cfg),
        transverse_scale=float(cfg["transverse_scale"]), device=device, dtype=dtype,
    )
    return x, tau, torch.exp(m)


def physics_unit_to_states(
    unit: torch.Tensor, *, d: int, cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    if unit.ndim != 2 or unit.shape[1] != int(d) + 2:
        raise ValueError("physics unit tensor must have shape (batch,d+2)")
    lo, hi = aggregate_bounds(cfg)
    tlo, thi = tau_bounds(cfg)
    m = lo + (hi - lo) * unit[:, 0]
    raw = 2.0 * unit[:, 1:int(d)+1] - 1.0
    delta = float(cfg["transverse_scale"]) * (raw - raw.mean(dim=1, keepdim=True))
    x = (m[:, None] + delta).requires_grad_(True)
    tau = (tlo + (thi - tlo) * unit[:, int(d)+1]).requires_grad_(True)
    return x, tau


def boundary_axes(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    b = cfg["boundary"]
    return (
        np.linspace(float(b["s_min"]), float(b["s_max"]), int(b["s_points"])),
        np.linspace(float(b["tau_min"]), float(b["tau_max"]), int(b["tau_points"])),
    )


def oracle_refinement_axes(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    slo, shi = aggregate_spot_bounds(cfg)
    tlo, thi = tau_bounds(cfg)
    return (
        np.linspace(slo, shi, int(cfg["oracle_refinement_g_points"])),
        np.linspace(tlo, thi, int(cfg["oracle_refinement_tau_points"])),
    )


def assert_oracle_coverage(oracle: Oracle, g: np.ndarray, tau: np.ndarray) -> None:
    ga, ta, eps = np.asarray(g, float), np.asarray(tau, float), 1e-12
    if (
        np.any(~np.isfinite(ga)) or np.any(~np.isfinite(ta))
        or np.min(ga) < float(oracle.spots[0]) - eps
        or np.max(ga) > float(oracle.spots[-1]) + eps
        or np.min(ta) < float(oracle.taus[0]) - eps
        or np.max(ta) > float(oracle.taus[-1]) + eps
    ):
        raise RuntimeError("oracle_extrapolation_blocked")


def transition_boundary(
    spots: np.ndarray, premium: np.ndarray, *, tolerance: float,
    min_continuation_nodes: int = 2,
) -> dict[str, Any]:
    s, p = np.asarray(spots, float).reshape(-1), np.asarray(premium, float).reshape(-1)
    if len(s) != len(p) or len(s) < 2 or np.any(~np.isfinite(s)) or np.any(~np.isfinite(p)):
        return {"found": False, "S_star_over_K": None, "reason": "insufficient_nodes"}
    if np.any(np.diff(s) <= 0):
        raise ValueError("boundary spots must be strictly increasing")
    ex = p <= float(tolerance)
    if not np.any(ex):
        return {"found": False, "S_star_over_K": None, "reason": "all_continuation_like"}
    if np.all(ex):
        return {"found": False, "S_star_over_K": None, "reason": "all_exercise_like"}
    if not bool(ex[0]):
        return {"found": False, "S_star_over_K": None, "reason": "expected_exercise_side_absent"}
    first_cont = int(np.flatnonzero(~ex)[0])
    needed = first_cont + max(1, int(min_continuation_nodes))
    if (
        first_cont < 1 or not np.all(ex[:first_cont]) or needed > len(ex)
        or not np.all(~ex[first_cont:needed]) or np.any(ex[needed:])
    ):
        return {"found": False, "S_star_over_K": None, "reason": "no_clean_transition"}
    left, right = first_cont - 1, first_cont
    p0, p1, s0, s1 = float(p[left]), float(p[right]), float(s[left]), float(s[right])
    if p1 > p0 and p0 <= tolerance <= p1:
        star = s0 + (float(tolerance) - p0) / (p1 - p0) * (s1 - s0)
    else:
        star = 0.5 * (s0 + s1)
    return {
        "found": True, "S_star_over_K": float(star), "reason": "clean_transition",
        "left_node": s0, "right_node": s1,
    }


def compute_boundary_metrics(reference: list[Any] | np.ndarray, predicted: list[Any] | np.ndarray) -> dict[str, Any]:
    if len(reference) != len(predicted):
        raise ValueError("boundary series length mismatch")
    valid = lambda v: v is not None and np.isfinite(float(v))
    rm = np.asarray([valid(v) for v in reference], bool)
    pm = np.asarray([valid(v) for v in predicted], bool)
    mm = rm & pm
    ref_count, pred_count, matched = int(rm.sum()), int(pm.sum()), int(mm.sum())
    errors = np.asarray([float(p)-float(r) for r,p,ok in zip(reference,predicted,mm) if ok], float)
    cond_mae = float(np.mean(np.abs(errors))) if errors.size else None
    cond_rmse = float(np.sqrt(np.mean(errors**2))) if errors.size else None
    cond_max = float(np.max(np.abs(errors))) if errors.size else None
    false_count = int((~rm & pm).sum())
    return {
        "reference_boundary_count": ref_count, "predicted_boundary_count": pred_count,
        "matched_boundary_count": matched,
        "boundary_coverage": float(matched/ref_count) if ref_count else None,
        "boundary_missed_count": int((rm & ~pm).sum()), "boundary_false_count": false_count,
        "boundary_conditional_mae": cond_mae, "boundary_conditional_rmse": cond_rmse,
        "boundary_conditional_max_abs": cond_max,
        "boundary_full_rmse": cond_rmse if ref_count and matched == ref_count and false_count == 0 else None,
    }


def _oracle_path(d: int) -> Path:
    return ORACLE_DIR / f"d{int(d)}.npz"


def save_oracle(oracle: Oracle) -> None:
    path = _oracle_path(oracle.d)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, d=oracle.d, sigma_g=oracle.sigma_g, q_g=oracle.q_g,
        spots=oracle.spots, taus=oracle.taus, values=oracle.values, ug=oracle.ug,
        ugg=oracle.ugg, boundary=oracle.boundary,
        refinement_rmse=oracle.refinement_rmse, seconds=oracle.seconds,
    )


def load_oracle(d: int) -> Oracle:
    z = np.load(_oracle_path(d))
    return Oracle(
        int(z["d"]), float(z["sigma_g"]), float(z["q_g"]), z["spots"], z["taus"],
        z["values"], z["ug"], z["ugg"], z["boundary"],
        float(z["refinement_rmse"]), float(z["seconds"]),
    )


def _oracle_shell(d: int, result: Any) -> Oracle:
    zeros = np.zeros_like(result.taus, dtype=float)
    return Oracle(d, 0.0, 0.0, result.spots, result.taus, result.values, result.values, result.values, zeros, 0.0, 0.0)


def build_boundary_complete_oracle(d: int, cfg: dict[str, Any]) -> tuple[Oracle, dict[str, Any]]:
    import time
    started = time.perf_counter()
    coarse_shape, fine_shape = cfg["oracle_grids"]
    coarse = solve_oracle_grid(int(d), int(coarse_shape[0]), int(coarse_shape[1]), cfg)
    fine = solve_oracle_grid(int(d), int(fine_shape[0]), int(fine_shape[1]), cfg)
    ge, te = oracle_refinement_axes(cfg)
    gg, tt = np.meshgrid(ge, te)
    assert_oracle_coverage(_oracle_shell(d, coarse), gg.ravel(), tt.ravel())
    assert_oracle_coverage(_oracle_shell(d, fine), gg.ravel(), tt.ravel())
    vc = interp2(coarse.spots, coarse.taus, coarse.values, gg.ravel(), tt.ravel())
    vf = interp2(fine.spots, fine.taus, fine.values, gg.ravel(), tt.ravel())
    refinement_rmse = float(np.sqrt(np.mean((vc-vf)**2)))
    if not np.isfinite(refinement_rmse) or refinement_rmse > float(cfg["oracle_rmse_max"]):
        raise RuntimeError(f"oracle_blocked:d={d}:refinement_rmse={refinement_rmse:.6g}")
    values = np.asarray(fine.values, float)
    ds = float(fine.spots[1] - fine.spots[0])
    ug = np.gradient(values, ds, axis=1, edge_order=2)
    ugg = np.gradient(ug, ds, axis=1, edge_order=2)
    bspots, _ = boundary_axes(cfg)
    tol, min_cont = float(cfg["boundary"]["primary_tolerance"]), int(cfg["boundary"]["min_continuation_nodes"])
    boundary = np.full_like(np.asarray(fine.taus, float), np.nan)
    reasons: dict[str, int] = {}
    shell = Oracle(d, 0.0, 0.0, fine.spots, fine.taus, values, ug, ugg, boundary, refinement_rmse, 0.0)
    for i, tau_value in enumerate(np.asarray(fine.taus, float)):
        if tau_value <= 1e-14:
            boundary[i] = 1.0
            continue
        t = np.full_like(bspots, tau_value)
        assert_oracle_coverage(shell, bspots, t)
        row = interp2(fine.spots, fine.taus, values, bspots, t)
        result = transition_boundary(
            bspots, row - np.maximum(1.0-bspots, 0.0),
            tolerance=tol, min_continuation_nodes=min_cont,
        )
        reasons[result["reason"]] = reasons.get(result["reason"], 0) + 1
        if result["found"]:
            boundary[i] = float(result["S_star_over_K"])
    if np.any(~np.isfinite(boundary[np.asarray(fine.taus) > 1e-14])):
        raise RuntimeError(f"oracle_boundary_blocked:d={d}:missing_transition")
    sigma_g, q_g = effective_geometric_basket_parameters(
        int(d), float(cfg["contract"]["sigma"]), float(cfg["contract"]["q"]), float(cfg["contract"]["rho"])
    )
    seconds = time.perf_counter() - started
    oracle = Oracle(int(d), sigma_g, q_g, np.asarray(fine.spots,float), np.asarray(fine.taus,float),
                    values, ug, ugg, boundary, refinement_rmse, seconds)
    evidence = {
        "dimension": int(d), "refinement_rmse": refinement_rmse,
        "refinement_rmse_max": float(cfg["oracle_rmse_max"]),
        "refinement_domain": {
            "G_over_K": [float(ge[0]), float(ge[-1])],
            "tau_over_T": [float(te[0]), float(te[-1])],
            "G_points": len(ge), "tau_points": len(te),
        },
        "coarse_grid": [int(v) for v in coarse_shape], "fine_grid": [int(v) for v in fine_shape],
        "solver_s_max": 4.0, "boundary_method": BOUNDARY_METHOD,
        "boundary_tolerance": tol, "boundary_missing_reasons": reasons,
        "oracle_seconds": seconds, "no_extrapolation_asserted": True,
    }
    return oracle, evidence


def oracle_labels_complete(oracle: Oracle, x: torch.Tensor, tau: torch.Tensor, cfg: dict[str, Any]) -> dict[str, Any]:
    xn, tn, d = x.detach().cpu().numpy(), tau.detach().cpu().numpy().reshape(-1), int(oracle.d)
    g = np.exp(xn.mean(axis=1))
    assert_oracle_coverage(oracle, g, tn)
    price = interp2(oracle.spots, oracle.taus, oracle.values, g, tn)
    ug = interp2(oracle.spots, oracle.taus, oracle.ug, g, tn)
    ugg = interp2(oracle.spots, oracle.taus, oracle.ugg, g, tn)
    boundary = np.interp(tn, oracle.taus, oracle.boundary)
    component_delta = ug[:,None] * (g[:,None] / (d*np.exp(xn)))
    cov = covariance_matrix(d, float(cfg["contract"]["sigma"]), float(cfg["contract"]["rho"]), dtype=torch.float64).numpy()
    common = (ugg*g**2 + ug*g) / d**2
    contraction = np.einsum("ij,nij->n", cov, np.broadcast_to(common[:,None,None], (len(g),d,d)))
    return {
        "price": price, "delta": component_delta, "contraction": contraction,
        "g": g, "boundary": boundary, "reduced_delta": ug, "reduced_gamma": ugg,
        "scaled_reduced_gamma": float(cfg["target_reduced_contract"]["K"])*ugg,
    }


def safe_peak_cpu_memory_bytes(probe: Callable[[], int] | None = None) -> int | None:
    memory_probe = peak_cpu_memory_bytes if probe is None else probe
    try:
        value = int(memory_probe())
    except OSError:
        return None
    return value if value >= 0 else None


def diagonal_value_and_greeks(
    model: torch.nn.Module, *, d: int, spots: np.ndarray, taus: np.ndarray,
    device: torch.device | None = None, chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s, t = np.asarray(spots,float).reshape(-1), np.asarray(taus,float).reshape(-1)
    if s.shape != t.shape or np.any(s <= 0):
        raise ValueError("invalid common-path arrays")
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    values, deltas, gammas = [], [], []
    for start in range(0, len(s), int(chunk_size)):
        stop = min(start+int(chunk_size), len(s))
        g = torch.tensor(s[start:stop], device=device, dtype=torch.float32, requires_grad=True)
        tau = torch.tensor(t[start:stop], device=device, dtype=torch.float32)
        value = model(torch.log(g).reshape(-1,1).repeat(1,int(d)), tau).reshape(-1)
        delta = torch.autograd.grad(value.sum(), g, create_graph=True, retain_graph=True)[0]
        gamma = torch.autograd.grad(delta.sum(), g)[0]
        values.append(value.detach().cpu().numpy()); deltas.append(delta.detach().cpu().numpy()); gammas.append(gamma.detach().cpu().numpy())
    return np.concatenate(values), np.concatenate(deltas), np.concatenate(gammas)


def model_surface_arrays(
    model: torch.nn.Module, *, d: int, spots: np.ndarray, taus: np.ndarray,
    cfg: dict[str, Any], device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    value, delta, gamma = diagonal_value_and_greeks(model,d=d,spots=spots,taus=taus,device=device)
    s, t = np.asarray(spots,float), np.asarray(taus,float)
    payoff = np.maximum(1.0-s,0.0); premium = value-payoff
    payload = {
        "S_over_K": s, "tau_over_T": t, "V_over_K": value,
        "payoff_over_K": payoff, "premium_over_K": premium,
        "reduced_delta": delta, "reduced_gamma": gamma,
        "scaled_reduced_gamma": float(cfg["target_reduced_contract"]["K"])*gamma,
        "exercise_like_primary": premium <= float(cfg["boundary"]["primary_tolerance"]),
    }
    for key,value in payload.items():
        if key != "exercise_like_primary" and not np.all(np.isfinite(value)):
            raise RuntimeError(f"nonfinite_surface_export:{key}")
    return payload


def oracle_surface_arrays(oracle: Oracle, *, spots: np.ndarray, taus: np.ndarray, cfg: dict[str, Any]) -> dict[str,np.ndarray]:
    s,t = np.asarray(spots,float), np.asarray(taus,float)
    assert_oracle_coverage(oracle,s,t)
    value = interp2(oracle.spots,oracle.taus,oracle.values,s,t)
    delta = interp2(oracle.spots,oracle.taus,oracle.ug,s,t)
    gamma = interp2(oracle.spots,oracle.taus,oracle.ugg,s,t)
    payoff=np.maximum(1.0-s,0.0); premium=value-payoff
    return {
        "S_over_K":s,"tau_over_T":t,"V_over_K":value,"payoff_over_K":payoff,
        "premium_over_K":premium,"reduced_delta":delta,"reduced_gamma":gamma,
        "scaled_reduced_gamma":float(cfg["target_reduced_contract"]["K"])*gamma,
        "exercise_like_primary":premium <= float(cfg["boundary"]["primary_tolerance"]),
    }


def binomial_final_steps(cfg: dict[str, Any]) -> int:
    return int(cfg["binomial_steps"][-1])


def binomial_check_spots(cfg: dict[str, Any]) -> list[float]:
    return [float(v) for v in cfg["binomial_check_spots"]]
