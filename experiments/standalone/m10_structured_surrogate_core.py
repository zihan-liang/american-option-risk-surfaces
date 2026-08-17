"""Pure helpers for the frozen M10 structured-surrogate experiment."""
from __future__ import annotations

from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np

CAMPAIGN_SEEDS = (101, 203, 307, 409, 503)
ARMS = (
    "direct_price",
    "positive_premium",
    "terminal_scaled_premium",
    "european_anchor_correction",
)
STRUCTURED_ARMS = ARMS[1:]
ORDINARY_DIAGNOSTICS = ("delta_rmse", "gamma_rmse", "boundary_rmse")
RESIDUAL_DIAGNOSTICS = ("masked_residual",)


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("method_id") != "M10" or config.get("phase") != "screen":
        raise ValueError("M10 runner requires method_id=M10 and phase=screen")
    if tuple(config.get("training_seeds", ())) != CAMPAIGN_SEEDS:
        raise ValueError("M10 requires exactly the five campaign training seeds")
    if tuple(config.get("arms", ())) != ARMS:
        raise ValueError("M10 arms must exactly match the frozen packet order")
    if int(config.get("budget_seconds", 0)) != 3600:
        raise ValueError("M10 screen budget must be exactly 3600 seconds")

    model = config.get("model", {})
    if model.get("device") != "cpu" or model.get("dtype") != "float64":
        raise ValueError("M10 canonical model must use CPU float64")
    if model.get("optimizer") != "adam":
        raise ValueError("M10 optimizer must be Adam")
    if int(model.get("epochs", 0)) < 1 or int(model.get("batch_size", 0)) < 1:
        raise ValueError("M10 requires positive fixed epochs and batch size")

    timing = config.get("timing", {})
    if (int(timing.get("warmups", -1)), int(timing.get("repetitions", -1))) != (5, 30):
        raise ValueError("M10 canonical timing must use five warm-ups and 30 repetitions")

    gates = config.get("gates", {})
    expected = {
        "price_improvement_pct_min": 2.0,
        "ordinary_degradation_pct_max": 2.0,
        "residual_degradation_pct_max": 5.0,
        "hard_obstacle_violation_max": 1e-10,
        "family_alpha": 0.05,
    }
    for key, value in expected.items():
        if float(gates.get(key, np.nan)) != value:
            raise ValueError(f"M10 gate {key} must equal {value}")

    split = config.get("split", {})
    train_fraction = float(split.get("train_fraction", 0.0))
    val_fraction = float(split.get("validation_fraction", 0.0))
    if not (0.0 < train_fraction < 1.0 and 0.0 < val_fraction < 1.0 and train_fraction + val_fraction < 1.0):
        raise ValueError("M10 requires nonempty regime-level train/validation/test fractions")


def _rms(values: np.ndarray, floor: float) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("scale fitting requires finite nonempty training values")
    value = float(np.sqrt(np.mean(arr * arr)))
    return max(value, float(floor))


def fit_representation_scales(
    american_train: Sequence[float],
    european_train: Sequence[float],
    *,
    floor: float = 1e-8,
) -> dict[str, float]:
    """Fit deterministic output scales from training labels only."""
    american = np.asarray(american_train, dtype=float)
    european = np.asarray(european_train, dtype=float)
    if american.shape != european.shape or american.ndim != 1 or american.size == 0:
        raise ValueError("American and European training labels must be paired nonempty vectors")
    if not np.isfinite(american).all() or not np.isfinite(european).all():
        raise ValueError("training labels must be finite")
    premium = np.maximum(american - european, 0.0)
    return {
        "price_scale": _rms(american, floor),
        "premium_scale": _rms(premium, floor),
    }


def _softplus_numpy(raw: np.ndarray) -> np.ndarray:
    x = np.asarray(raw, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def reconstruct_numpy(
    raw: Sequence[float] | np.ndarray,
    european: Sequence[float] | np.ndarray,
    tau_over_t: Sequence[float] | np.ndarray,
    arm: str,
    scales: Mapping[str, float],
) -> np.ndarray:
    raw_arr, euro_arr, tau_arr = np.broadcast_arrays(
        np.asarray(raw, dtype=float),
        np.asarray(european, dtype=float),
        np.asarray(tau_over_t, dtype=float),
    )
    if arm not in ARMS:
        raise ValueError(f"unknown M10 arm: {arm}")
    if not np.isfinite(raw_arr).all() or not np.isfinite(euro_arr).all() or not np.isfinite(tau_arr).all():
        raise ValueError("reconstruction inputs must be finite")
    if np.any((tau_arr < -1e-12) | (tau_arr > 1.0 + 1e-12)):
        raise ValueError("tau_over_t must lie in [0,1]")
    price_scale = float(scales["price_scale"])
    premium_scale = float(scales["premium_scale"])
    if price_scale <= 0.0 or premium_scale <= 0.0:
        raise ValueError("representation scales must be positive")

    if arm == "direct_price":
        return price_scale * raw_arr
    if arm == "positive_premium":
        return euro_arr + premium_scale * _softplus_numpy(raw_arr)
    if arm == "terminal_scaled_premium":
        return euro_arr + premium_scale * np.clip(tau_arr, 0.0, 1.0) * _softplus_numpy(raw_arr)
    return euro_arr + premium_scale * raw_arr


def reconstruct_torch(raw, european, tau_over_t, arm: str, scales: Mapping[str, float]):
    """Differentiable reconstruction matching :func:`reconstruct_numpy`."""
    import torch
    import torch.nn.functional as F

    if arm not in ARMS:
        raise ValueError(f"unknown M10 arm: {arm}")
    price_scale = float(scales["price_scale"])
    premium_scale = float(scales["premium_scale"])
    if arm == "direct_price":
        return raw * price_scale
    if arm == "positive_premium":
        return european + premium_scale * F.softplus(raw)
    if arm == "terminal_scaled_premium":
        return european + premium_scale * torch.clamp(tau_over_t, 0.0, 1.0) * F.softplus(raw)
    return european + premium_scale * raw


def bootstrap_relative_change(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    repetitions: int,
    seed: int,
    improvement: bool,
    floor: float,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Paired complete-scenario bootstrap of aggregate relative change in percent."""
    base = np.asarray(baseline, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    if base.shape != cand.shape or base.ndim != 1 or base.size == 0:
        raise ValueError("baseline and candidate must be paired nonempty scenario vectors")
    if not np.isfinite(base).all() or not np.isfinite(cand).all():
        raise ValueError("bootstrap inputs must be finite")
    if repetitions < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap settings")

    sign = 1.0 if improvement else -1.0

    def statistic(b: np.ndarray, c: np.ndarray) -> float:
        denom = max(abs(float(np.mean(b))), float(floor))
        return 100.0 * sign * (float(np.mean(b)) - float(np.mean(c))) / denom

    point = statistic(base, cand)
    rng = np.random.default_rng(int(seed))
    boots = np.empty(int(repetitions), dtype=float)
    for i in range(int(repetitions)):
        idx = rng.integers(0, base.size, size=base.size)
        boots[i] = statistic(base[idx], cand[idx])
    alpha = 0.5 * (1.0 - confidence)
    return point, float(np.quantile(boots, alpha)), float(np.quantile(boots, 1.0 - alpha))


def paired_signflip_pvalue(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    alternative: str = "candidate_lower",
    monte_carlo_draws: int = 20000,
    seed: int = 1010,
) -> float:
    """One-sided paired randomization p-value over complete scenarios."""
    base = np.asarray(baseline, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    if base.shape != cand.shape or base.ndim != 1 or base.size == 0:
        raise ValueError("paired test requires equal nonempty scenario vectors")
    if alternative not in {"candidate_lower", "candidate_higher"}:
        raise ValueError("unsupported alternative")
    diff = base - cand if alternative == "candidate_lower" else cand - base
    if not np.isfinite(diff).all():
        raise ValueError("paired differences must be finite")
    observed = float(np.mean(diff))
    n = diff.size
    if n <= 16:
        means = np.fromiter(
            (float(np.mean(diff * np.asarray(signs, dtype=float))) for signs in product((-1.0, 1.0), repeat=n)),
            dtype=float,
            count=2**n,
        )
    else:
        rng = np.random.default_rng(int(seed))
        signs = rng.choice(np.array([-1.0, 1.0]), size=(int(monte_carlo_draws), n), replace=True)
        means = np.mean(signs * diff[None, :], axis=1)
    return float((np.count_nonzero(means >= observed - 1e-15) + 1.0) / (len(means) + 1.0))


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    if not p_values:
        return {}
    items = sorted((str(name), float(value)) for name, value in p_values.items())
    if any(not np.isfinite(value) or value < 0.0 or value > 1.0 for _, value in items):
        raise ValueError("p-values must lie in [0,1]")
    ordered = sorted(items, key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        scaled = min(1.0, (m - rank) * value)
        running = max(running, scaled)
        adjusted[name] = running
    return adjusted


def candidate_passes(
    *,
    price_improvement_pct: float,
    price_improvement_ci_low_pct: float,
    holm_p_value: float,
    degradation_ci_upper_pct: Mapping[str, float],
    obstacle_violation: float,
    gates: Mapping[str, Any],
) -> bool:
    if not np.isfinite(price_improvement_pct) or float(price_improvement_pct) < float(gates["price_improvement_pct_min"]):
        return False
    if not np.isfinite(price_improvement_ci_low_pct) or float(price_improvement_ci_low_pct) <= 0.0:
        return False
    if not np.isfinite(holm_p_value) or float(holm_p_value) > float(gates["family_alpha"]):
        return False
    for metric in ORDINARY_DIAGNOSTICS:
        value = float(degradation_ci_upper_pct[metric])
        if not np.isfinite(value) or value > float(gates["ordinary_degradation_pct_max"]):
            return False
    for metric in RESIDUAL_DIAGNOSTICS:
        value = float(degradation_ci_upper_pct[metric])
        if not np.isfinite(value) or value > float(gates["residual_degradation_pct_max"]):
            return False
    return np.isfinite(obstacle_violation) and float(obstacle_violation) <= float(gates["hard_obstacle_violation_max"])


def classify_status(evidence_mode: str, *, passed: bool, budget_exceeded: bool = False) -> str:
    if evidence_mode == "smoke":
        return "diagnostic_only"
    if evidence_mode != "canonical":
        raise ValueError("evidence_mode must be smoke or canonical")
    if budget_exceeded:
        return "resource_blocked"
    return "candidate_supported" if passed else "failed_metric_gate"
