"""Pure helpers for the frozen M09 multi-fidelity campaign design."""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import numpy as np

CAMPAIGN_SEEDS = (101, 203, 307, 409, 503)
FINE_FRACTIONS = (0.10, 0.25, 0.50, 1.00)
ORDINARY_METRICS = ("price_rmse", "delta_rmse", "gamma_rmse", "boundary_rmse", "obstacle_violation")
RESIDUAL_METRICS = ("dual_violation", "complementarity", "continuation_residual")


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("method_id") != "M09" or config.get("phase") != "screen":
        raise ValueError("M09 runner requires method_id=M09 and phase=screen")
    if tuple(config.get("training_seeds", ())) != CAMPAIGN_SEEDS:
        raise ValueError("M09 requires exactly the five campaign training seeds")
    fractions = tuple(float(x) for x in config.get("fine_fractions", ()))
    if fractions != FINE_FRACTIONS:
        raise ValueError("M09 fine fractions must be exactly 10/25/50/100 percent")
    if int(config.get("budget_seconds", 0)) != 3600:
        raise ValueError("M09 screen budget must be exactly 3600 seconds")
    if config.get("model", {}).get("dtype") != "float64" or config.get("model", {}).get("device") != "cpu":
        raise ValueError("M09 canonical model must use CPU float64")
    coarse = config.get("coarse_solver", {})
    fine = config.get("fine_solver", {})
    if (int(coarse.get("m", -1)), int(coarse.get("n", -1))) != (60, 60):
        raise ValueError("M09 coarse grid must be 60x60")
    if (int(fine.get("m", -1)), int(fine.get("n", -1))) != (120, 120):
        raise ValueError("M09 fine grid must be 120x120")
    if float(config.get("matching", {}).get("relative_price_tolerance", -1.0)) != 0.02:
        raise ValueError("M09 matched-accuracy tolerance must be 2 percent")
    gates = config.get("gates", {})
    expected = {
        "fine_label_cost_reduction_pct_min": 2.0,
        "ordinary_degradation_pct_max": 2.0,
        "residual_degradation_pct_max": 5.0,
        "hard_obstacle_violation_max": 1e-10,
    }
    if any(float(gates.get(k, np.nan)) != v for k, v in expected.items()):
        raise ValueError("M09 packet gates must be exactly 2% cost / 2% ordinary / 5% residual / 1e-10 obstacle")


def nested_regime_subsets(regime_keys: Sequence[str], fractions: Sequence[float], *, selection_seed: int) -> dict[float, tuple[str, ...]]:
    keys = tuple(str(k) for k in regime_keys)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("regime keys must be nonempty and unique")
    fractions = tuple(float(f) for f in fractions)
    if any(not 0.0 < f <= 1.0 for f in fractions) or any(b <= a for a, b in zip(fractions, fractions[1:])):
        raise ValueError("fractions must be strictly increasing in (0,1]")
    rng = np.random.default_rng(int(selection_seed))
    order = np.asarray(keys, dtype=object)[rng.permutation(len(keys))]
    out: dict[float, tuple[str, ...]] = {}
    previous = 0
    for fraction in fractions:
        count = len(keys) if fraction == 1.0 else max(1, int(round(len(keys) * fraction)))
        count = max(previous, min(len(keys), count))
        out[fraction] = tuple(sorted(str(x) for x in order[:count]))
        previous = count
    return out


def assert_complete_regime_rows(regime_key: np.ndarray, expected_nodes_per_regime: int, selected_keys: Sequence[str]) -> np.ndarray:
    keys = np.asarray(regime_key).astype(str)
    selected = tuple(str(x) for x in selected_keys)
    mask = np.isin(keys, selected)
    for key in selected:
        count = int(np.count_nonzero(keys == key))
        if count != int(expected_nodes_per_regime):
            raise ValueError(f"regime {key} is not a complete surface: {count} != {expected_nodes_per_regime}")
    return mask


def matched_accuracy_target(fine_only_100_validation_rmse: float, relative_tolerance: float = 0.02) -> float:
    value = float(fine_only_100_validation_rmse)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("validation RMSE must be finite and nonnegative")
    return value * (1.0 + float(relative_tolerance))


def choose_smallest_matching_fraction(validation_rmse_by_fraction: Mapping[float, float], target_rmse: float) -> float | None:
    target = float(target_rmse)
    for fraction in sorted(float(x) for x in validation_rmse_by_fraction):
        value = float(validation_rmse_by_fraction[fraction])
        if np.isfinite(value) and value <= target:
            return fraction
    return None


def paired_bootstrap_mean_ci(values: Sequence[float], *, repetitions: int, seed: int, confidence: float = 0.95) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("bootstrap values must be a finite nonempty vector")
    if repetitions < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap settings")
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(repetitions), dtype=float)
    for i in range(int(repetitions)):
        means[i] = float(np.mean(arr[rng.integers(0, arr.size, size=arr.size)]))
    alpha = (1.0 - confidence) / 2.0
    return float(np.mean(arr)), float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def percent_degradation(candidate: Sequence[float], baseline: Sequence[float], *, floor: float) -> np.ndarray:
    cand = np.asarray(candidate, dtype=float)
    base = np.asarray(baseline, dtype=float)
    if cand.shape != base.shape or cand.ndim != 1 or cand.size == 0:
        raise ValueError("candidate and baseline must be paired nonempty vectors")
    return 100.0 * (cand - base) / np.maximum(np.abs(base), float(floor))


def candidate_passes(
    *,
    cost_reduction_ci_low_pct: float,
    degradation_ci_upper_pct: Mapping[str, float],
    obstacle_violation: float,
    gates: Mapping[str, Any],
) -> bool:
    if not np.isfinite(cost_reduction_ci_low_pct) or float(cost_reduction_ci_low_pct) < float(gates["fine_label_cost_reduction_pct_min"]):
        return False
    for metric in ORDINARY_METRICS:
        value = float(degradation_ci_upper_pct[metric])
        if not np.isfinite(value) or value > float(gates["ordinary_degradation_pct_max"]):
            return False
    for metric in RESIDUAL_METRICS:
        value = float(degradation_ci_upper_pct[metric])
        if not np.isfinite(value) or value > float(gates["residual_degradation_pct_max"]):
            return False
    return np.isfinite(obstacle_violation) and float(obstacle_violation) <= float(gates["hard_obstacle_violation_max"])


def classify_status(evidence_mode: str, *, passed: bool) -> str:
    if evidence_mode == "smoke":
        return "diagnostic_only"
    if evidence_mode != "canonical":
        raise ValueError("evidence_mode must be smoke or canonical")
    return "candidate_supported" if passed else "failed_metric_gate"
