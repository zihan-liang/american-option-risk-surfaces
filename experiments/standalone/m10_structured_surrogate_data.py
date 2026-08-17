"""Data/oracle helpers for M10 structured surrogate."""
from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

from data.splits import three_way_regime_split
from solvers.black_scholes import black_scholes_price
from solvers.cn_psor import SolverConfig, solve_option

FEATURE_NAMES = ("log_moneyness", "tau_over_T", "r", "q", "sigma", "T", "is_call")


def _linspace(spec: Mapping[str, Any]) -> np.ndarray:
    return np.linspace(float(spec["min"]), float(spec["max"]), int(spec["count"]), dtype=float)


def build_regimes(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    grid = config["parameter_grid"]
    out: list[dict[str, Any]] = []
    for i, (sigma, rate, dividend) in enumerate(
        (s, r, q) for s in _linspace(grid["sigma"]) for r in _linspace(grid["r"]) for q in _linspace(grid["q"])
    ):
        rid = f"regime_{i:04d}"
        out.append({"id": rid, "case_id": rid, "sigma": float(sigma), "r": float(rate), "q": float(dividend)})
    return out


def split_regimes(config: Mapping[str, Any], regimes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    spec = config["split"]
    return three_way_regime_split(
        regimes,
        train_fraction=float(spec["train_fraction"]),
        validation_fraction=float(spec["validation_fraction"]),
        seed=int(spec["seed"]),
    )


def solver_config(base: Mapping[str, Any], regime: Mapping[str, Any]) -> SolverConfig:
    payload = dict(base)
    payload.update({"sigma": float(regime["sigma"]), "r": float(regime["r"]), "q": float(regime["q"])})
    return SolverConfig(**payload)


def surface_features(result: Any, cfg: SolverConfig) -> np.ndarray:
    s, tau = np.meshgrid(result.spots, result.taus)
    return np.column_stack([
        np.log(np.maximum(s.ravel(), 1e-12) / cfg.k),
        tau.ravel() / cfg.t if cfg.t > 0.0 else tau.ravel(),
        np.full(s.size, cfg.r), np.full(s.size, cfg.q), np.full(s.size, cfg.sigma),
        np.full(s.size, cfg.t), np.full(s.size, 1.0 if cfg.option_type == "call" else 0.0),
    ]).astype(np.float64)


def analytic_european_surface(result: Any, cfg: SolverConfig) -> np.ndarray:
    s, tau = np.meshgrid(result.spots, result.taus)
    value = black_scholes_price(s, cfg.k, tau, cfg.r, cfg.q, cfg.sigma, cfg.option_type)
    return np.asarray(value, dtype=np.float64) / cfg.k


def solve_surface(base: Mapping[str, Any], regime: Mapping[str, Any]) -> dict[str, Any]:
    cfg = solver_config(base, regime)
    started = time.perf_counter()
    result = solve_option(cfg)
    solve_seconds = float(time.perf_counter() - started)
    if not np.isfinite(result.values).all() or not all(result.lcp_converged):
        raise RuntimeError(f"CN/PSOR failed for M10 regime {regime['id']}")
    started = time.perf_counter()
    european = analytic_european_surface(result, cfg)
    anchor_seconds = float(time.perf_counter() - started)
    return {
        "regime_id": str(regime["id"]), "cfg": cfg, "result": result,
        "features": surface_features(result, cfg),
        "target": (result.values / cfg.k).ravel().astype(np.float64),
        "european": european.ravel().astype(np.float64),
        "solve_seconds": solve_seconds, "anchor_seconds": anchor_seconds,
    }


def build_surface_cache(solver_spec: Mapping[str, Any], partitions: Mapping[str, Sequence[Mapping[str, Any]]]):
    cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        for regime in partitions[split]:
            surface = solve_surface(solver_spec, regime)
            rid = str(regime["id"])
            cache[rid] = surface
            rows.append({"split": split, "regime_id": rid, "nodes": len(surface["target"]),
                         "solve_seconds": surface["solve_seconds"], "anchor_seconds": surface["anchor_seconds"]})
    return cache, rows


def concatenate_surfaces(cache: Mapping[str, dict[str, Any]], regime_ids: Sequence[str]):
    ids = tuple(str(x) for x in regime_ids)
    if not ids:
        raise ValueError("M10 requires nonempty regime ids")
    x = np.concatenate([cache[k]["features"] for k in ids])
    y = np.concatenate([cache[k]["target"] for k in ids])
    e = np.concatenate([cache[k]["european"] for k in ids])
    keys = np.concatenate([np.full(len(cache[k]["target"]), k, dtype=object) for k in ids])
    return x, y, e, keys


def assert_complete_regime_rows(keys: np.ndarray, expected_nodes: int, regime_ids: Sequence[str]) -> None:
    keys = np.asarray(keys).astype(str)
    for rid in regime_ids:
        count = int(np.count_nonzero(keys == str(rid)))
        if count != int(expected_nodes):
            raise ValueError(f"regime {rid} is not a complete surface: {count} != {expected_nodes}")


def fit_feature_normalizer(train_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(train_features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or not np.isfinite(x).all():
        raise ValueError("feature normalization requires finite nonempty train rows")
    mean, std = np.mean(x, axis=0), np.std(x, axis=0)
    return mean, np.where(std > 1e-12, std, 1.0)


def normalize_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(features, dtype=np.float64) - mean) / std
