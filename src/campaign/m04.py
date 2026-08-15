"""Selective M04 free-boundary representation screen support.

This module is deliberately self-contained: it consumes the accepted CN/PSOR
solver, keeps complete-regime split isolation explicit, and never reads the
blind holdout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from models.free_boundary import RootExtraction, extract_put_zero_crossing
from solvers.cn_psor import SolverConfig, solve_option


M04_SPLITS = ("train", "validation", "legacy_test")
BOUNDARY_STATUSES = {"defined", "no_exercise_region", "no_continuation_region", "non_monotone_profile", "oracle_failed", "unavailable"}


@dataclass(frozen=True)
class SurfaceScenario:
    scenario_id: str
    split: str
    features: np.ndarray
    labels: np.ndarray
    spots: np.ndarray
    theta: np.ndarray
    strike: float
    maturity: float
    rate: float
    dividend: float
    volatility: float


@dataclass(frozen=True)
class CompleteSurfaceDataset:
    splits: Mapping[str, tuple[SurfaceScenario, ...]]
    counts: Mapping[str, int]
    regime_ids: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class BoundaryExtraction:
    value: float | None
    status: str
    reason: str | None
    bracket: tuple[float, float] | None
    construction: str


@dataclass(frozen=True)
class RefinedBoundaryOracle:
    scenario_id: str
    values: tuple[BoundaryExtraction, ...]
    status: str
    reason: str | None
    diagnostics: Mapping[str, Any]


@dataclass
class M04Budget:
    limit_seconds: float
    category_seconds: dict[str, float] = field(default_factory=dict)
    state: str = "available"
    clock: Callable[[], float] = perf_counter
    _started_at: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._started_at = float(self.clock())

    def record(self, category: str, seconds: float) -> None:
        if not np.isfinite(seconds) or seconds < 0.0:
            raise ValueError("budget durations must be finite and non-negative")
        self.category_seconds[category] = self.category_seconds.get(category, 0.0) + float(seconds)

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, float(self.clock()) - self._started_at)

    def checkpoint(self, label: str) -> str | None:
        if self.elapsed_seconds > self.limit_seconds:
            self.state = "resource_blocked"
            return self.state
        return None


_NO_BOUNDARY_STATUSES = {"no_exercise_region", "no_continuation_region"}


def validate_data_freeze(config: Mapping[str, Any]) -> Mapping[str, tuple[str, ...]]:
    """Require immutable dataset identity and explicit, disjoint regime IDs."""
    digest = config.get("dataset_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
        raise ValueError("M04 dataset_sha256 must be a 64-character SHA-256 before any run")
    raw_ids = config.get("split_ids")
    if not isinstance(raw_ids, Mapping) or set(raw_ids) != set(M04_SPLITS):
        raise ValueError("M04 split_ids must explicitly name train, validation, and legacy_test")
    ids = {split: tuple(map(str, raw_ids[split])) if isinstance(raw_ids[split], (list, tuple)) else () for split in M04_SPLITS}
    flattened = [item for values in ids.values() for item in values]
    if not flattened or any(not values for values in ids.values()) or len(flattened) != len(set(flattened)):
        raise ValueError("M04 split_ids must be nonempty, explicit, and disjoint")
    return ids


@dataclass(frozen=True)
class TrainingNormalizer:
    """Feature normalization fitted exactly once from the relevant training representation."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, training_features: np.ndarray) -> "TrainingNormalizer":
        values = np.asarray(training_features, dtype=np.float64)
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("normalization requires finite nonempty training features")
        scale = values.std(axis=0)
        return cls(values.mean(axis=0), np.where(scale > 0.0, scale, 1.0))

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.mean) or not np.isfinite(values).all():
            raise ValueError("features do not match fitted normalization")
        return (values - self.mean) / self.scale


def load_complete_surface_dataset(path: str | Path, selections: Mapping[str, Sequence[str]]) -> CompleteSurfaceDataset:
    """Load only explicit, disjoint complete put surfaces; source test is legacy_test."""
    if set(selections) != set(M04_SPLITS):
        raise ValueError("M04 selections must have train, validation, and legacy_test")
    ids = {name: tuple(map(str, selections[name])) for name in M04_SPLITS}
    all_ids = [item for values in ids.values() for item in values]
    if not all_ids or len(all_ids) != len(set(all_ids)):
        raise ValueError("M04 regime IDs must be nonempty and disjoint")
    with np.load(path) as archive:
        required = {"features", "labels", "regime_key", "split"}
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"M04 dataset missing arrays: {sorted(missing)}")
        features, labels = np.asarray(archive["features"], dtype=np.float64), np.asarray(archive["labels"], dtype=np.float64)
        keys, source_split = archive["regime_key"].astype(str), archive["split"].astype(str)
    if features.ndim != 2 or features.shape[1] != 7 or labels.ndim != 2 or labels.shape[1] < 3:
        raise ValueError("M04 requires seven features and price/premium/payoff labels")
    if not np.isfinite(features).all() or not np.isfinite(labels).all():
        raise ValueError("M04 input surfaces must be finite")
    loaded: dict[str, tuple[SurfaceScenario, ...]] = {}
    for split in M04_SPLITS:
        expected = "test" if split == "legacy_test" else split
        scenarios = []
        for scenario_id in ids[split]:
            index = np.flatnonzero(keys == scenario_id)
            if not len(index):
                raise ValueError(f"selected M04 regime is absent: {scenario_id}")
            if set(source_split[index]) != {expected}:
                raise ValueError(f"regime {scenario_id} is not in source split {expected}")
            rows, target = features[index], labels[index]
            if not np.all(rows[:, 6] == 0.0):
                raise ValueError("M04 permits American put surfaces only")
            order = np.lexsort((rows[:, 0], rows[:, 1]))
            rows, target = rows[order], target[order]
            theta, spots = np.unique(rows[:, 1]), np.exp(np.unique(rows[:, 0]))
            if len(rows) != len(theta) * len(spots) or len(np.unique(rows[:, :2], axis=0)) != len(rows):
                raise ValueError(f"selected regime {scenario_id} is not a complete rectangular surface")
            scenarios.append(SurfaceScenario(scenario_id, split, rows, target, spots, theta, 1.0, float(rows[0, 5]), float(rows[0, 2]), float(rows[0, 3]), float(rows[0, 4])))
        loaded[split] = tuple(scenarios)
    return CompleteSurfaceDataset(loaded, {name: sum(len(s.features) for s in loaded[name]) for name in M04_SPLITS}, ids)


def validate_refined_boundary_curves(coarse_curve: Sequence[BoundaryExtraction], fine_curve: Sequence[BoundaryExtraction], *, max_refinement_rmse: float) -> dict[str, Any]:
    """Validate all nested-grid boundary states; never score only a passing subset."""
    if len(coarse_curve) != len(fine_curve):
        return {"status": "oracle_failed", "reason": "missing_levels", "refinement_rmse": None, "levels": []}
    differences: list[float] = []
    levels: list[dict[str, Any]] = []
    failure: str | None = None
    for index, (coarse, fine) in enumerate(zip(coarse_curve, fine_curve)):
        level = {"level_index": index, "coarse_status": coarse.status, "coarse_value": coarse.value, "coarse_reason": coarse.reason, "fine_status": fine.status, "fine_value": fine.value, "fine_reason": fine.reason, "reason": None}
        if coarse.status not in {"defined", *_NO_BOUNDARY_STATUSES}:
            failure = failure or f"coarse_{coarse.status}"
            level["reason"] = failure
        elif fine.status not in {"defined", *_NO_BOUNDARY_STATUSES}:
            failure = failure or f"fine_{fine.status}"
            level["reason"] = failure
        elif coarse.status == fine.status and coarse.status in _NO_BOUNDARY_STATUSES:
            level["reason"] = f"matching_{coarse.status}"
        elif coarse.status == "defined" and fine.status == "defined" and coarse.value is not None and fine.value is not None:
            differences.append(float(coarse.value) - float(fine.value))
            level["reason"] = "defined_pair"
        else:
            failure = failure or "mismatched_boundary_state"
            level["reason"] = "mismatched_boundary_state"
        levels.append(level)
    refinement = float(np.sqrt(np.mean(np.square(differences)))) if differences else None
    if failure:
        return {"status": "oracle_failed", "reason": failure, "refinement_rmse": refinement, "levels": levels}
    if refinement is not None and refinement > max_refinement_rmse:
        return {"status": "oracle_failed", "reason": "refinement_exceeds_tolerance", "refinement_rmse": refinement, "levels": levels}
    return {"status": "defined", "reason": None, "refinement_rmse": refinement, "levels": levels}


def extract_premium_boundary(spots: Sequence[float], premium: Sequence[float], *, strike: float, theta: float, tolerance: float = 1e-10) -> BoundaryExtraction:
    """Conservative premium-only boundary extraction with the B(0)=K convention."""
    x, p = np.asarray(spots, dtype=float), np.asarray(premium, dtype=float)
    if x.ndim != 1 or p.ndim != 1 or len(x) != len(p) or len(x) < 2:
        raise ValueError("premium extraction requires equally sized one-dimensional arrays")
    if not np.isfinite(x).all() or not np.isfinite(p).all() or not np.all(np.diff(x) > 0.0):
        return BoundaryExtraction(None, "unavailable", "nonfinite or unordered premium profile", None, "premium_only")
    if theta == 0.0:
        return BoundaryExtraction(float(strike), "defined", None, (float(strike), float(strike)), "american_put_maturity_convention")
    if np.min(p) < -tolerance:
        return BoundaryExtraction(None, "unavailable", "predicted premium is negative", None, "premium_only")
    exercise = p <= tolerance
    if not np.any(exercise):
        return BoundaryExtraction(None, "no_exercise_region", "premium is positive at every spot", None, "premium_only")
    if np.all(exercise):
        return BoundaryExtraction(None, "no_continuation_region", "premium is zero at every spot", None, "premium_only")
    first_continuation = int(np.flatnonzero(~exercise)[0])
    if np.any(~exercise[first_continuation:]) is False or np.any(exercise[first_continuation:]):
        return BoundaryExtraction(None, "non_monotone_profile", "premium contact is not low-to-high contiguous", None, "premium_only")
    left, right = first_continuation - 1, first_continuation
    if right + 1 >= len(x):
        return BoundaryExtraction(None, "unavailable", "one continuation node cannot support sub-grid extraction", None, "premium_only")
    root_one, root_two = np.sqrt(max(p[right], 0.0)), np.sqrt(max(p[right + 1], 0.0))
    if root_two <= root_one:
        return BoundaryExtraction(None, "non_monotone_profile", "continuation premium is not locally increasing", None, "premium_only")
    value = float(x[right] - root_one * (x[right + 1] - x[right]) / (root_two - root_one))
    return BoundaryExtraction(float(np.clip(value, x[left], x[right])), "defined", None, (float(x[left]), float(x[right])), "sqrt_premium_subgrid")


def select_premium_contact_tolerance(training_profiles: Sequence[tuple[np.ndarray, float | None, float]], *, spots: np.ndarray, strike: float, candidates: Sequence[float]) -> dict[str, Any]:
    """Choose one predeclared contact tolerance using training profiles only."""
    options = tuple(float(value) for value in candidates)
    if not options or len(options) > 5 or any(value < 0.0 or not np.isfinite(value) for value in options):
        raise ValueError("M04 requires one to five finite non-negative premium contact tolerances")
    if not training_profiles:
        raise ValueError("premium-contact tolerance selection requires training profiles")
    scores: list[dict[str, Any]] = []
    for tolerance in options:
        errors: list[float] = []
        mismatches = 0
        for premium, target, theta in training_profiles:
            extracted = extract_premium_boundary(spots, premium, strike=strike, theta=float(theta), tolerance=tolerance)
            if target is None:
                mismatches += int(extracted.value is not None)
            elif extracted.value is None:
                mismatches += 1
            else:
                errors.append(abs(float(extracted.value) - float(target)))
        score = float(np.mean(errors)) if errors and not mismatches else float("inf")
        scores.append({"threshold": tolerance, "training_boundary_mae": score if np.isfinite(score) else None, "training_state_mismatch_count": mismatches})
    best = min(enumerate(scores), key=lambda item: (float("inf") if item[1]["training_boundary_mae"] is None else item[1]["training_boundary_mae"], item[1]["training_state_mismatch_count"], item[0]))[1]
    return {"selection_split": "train", "candidate_thresholds": list(options), "candidate_metrics": scores, "selected_threshold": best["threshold"], "training_only_selection_metric": {"boundary_mae": best["training_boundary_mae"], "state_mismatch_count": best["training_state_mismatch_count"]}}


def extract_level_set_boundary(spots: Sequence[float], levels: Sequence[float]) -> BoundaryExtraction:
    result: RootExtraction = extract_put_zero_crossing(spots, levels)
    return BoundaryExtraction(result.value, result.status, result.reason, result.bracket, "implicit_level_set_zero")


def build_refined_oracle(scenario: SurfaceScenario, *, coarse_grid: int, fine_grid: int, tolerance: float, max_refinement_rmse: float) -> RefinedBoundaryOracle:
    """Generate only converged nested CN/PSOR boundaries, with explicit failures."""
    if fine_grid <= coarse_grid or fine_grid % coarse_grid:
        raise ValueError("M04 nested oracle requires an integral fine/coarse grid ratio")
    def solve(grid: int):
        cfg = SolverConfig(option_type="put", k=scenario.strike, t=scenario.maturity, r=scenario.rate, q=scenario.dividend, sigma=scenario.volatility, s_max=float(scenario.spots[-1]), m=grid, n=grid, tol=tolerance, time_scheme="cn", lcp_solver="psor")
        started = perf_counter()
        result = solve_option(cfg)
        return result, perf_counter() - started
    coarse, coarse_seconds = solve(coarse_grid)
    fine, fine_seconds = solve(fine_grid)
    if not all(coarse.lcp_converged) or not all(fine.lcp_converged):
        return RefinedBoundaryOracle(scenario.scenario_id, (), "oracle_failed", "CN/PSOR convergence failed", {"coarse_seconds": coarse_seconds, "fine_seconds": fine_seconds, "coarse_converged": bool(all(coarse.lcp_converged)), "fine_converged": bool(all(fine.lcp_converged))})
    def curve(result):
        values = []
        for theta in scenario.theta:
            if theta == 0.0:
                values.append(extract_premium_boundary(result.spots, result.values[0] - result.payoff, strike=scenario.strike, theta=0.0, tolerance=tolerance))
            else:
                index = int(round(theta * result.config.n))
                values.append(extract_premium_boundary(result.spots, result.values[index] - result.payoff, strike=scenario.strike, theta=float(theta), tolerance=tolerance))
        return tuple(values)
    coarse_curve, fine_curve = curve(coarse), curve(fine)
    validation = validate_refined_boundary_curves(coarse_curve, fine_curve, max_refinement_rmse=max_refinement_rmse)
    levels = [{**level, "theta": float(scenario.theta[level["level_index"]])} for level in validation["levels"]]
    diagnostics = {"coarse_seconds": coarse_seconds, "fine_seconds": fine_seconds, "refinement_rmse": validation["refinement_rmse"], "levels": levels, "coarse_grid": coarse_grid, "fine_grid": fine_grid, "provenance": "accepted_current_m01_cn_psor"}
    return RefinedBoundaryOracle(scenario.scenario_id, fine_curve, validation["status"], validation["reason"], diagnostics)


def predict_price_surrogate_boundaries(model: Any, *, feature_rows: np.ndarray, spots: np.ndarray, theta: np.ndarray, strike: float, tolerance: float = 1e-10) -> tuple[BoundaryExtraction, ...]:
    """Infer baseline boundaries from model predictions only; no held-out labels enter."""
    predicted = np.asarray(model(feature_rows), dtype=float).reshape(-1)
    if len(predicted) != len(feature_rows):
        raise ValueError("price surrogate prediction length does not match its features")
    if len(predicted) != len(spots) * len(theta):
        raise ValueError("baseline inference requires a complete surface")
    return tuple(extract_premium_boundary(spots, predicted.reshape(len(theta), len(spots))[row], strike=strike, theta=float(value), tolerance=tolerance) for row, value in enumerate(theta))


def aggregate_boundary_metrics(*, predictions: Sequence[float | None], targets: Sequence[float | None], statuses: Sequence[str]) -> dict[str, float | int | None]:
    if not (len(predictions) == len(targets) == len(statuses)):
        raise ValueError("M04 metric vectors must have equal length")
    target_defined = [target is not None for target in targets]
    defined = [(float(pred), float(target)) for pred, target in zip(predictions, targets) if pred is not None and target is not None]
    target_count = sum(target_defined)
    missing = sum(target is not None and pred is None for pred, target in zip(predictions, targets))
    false_finite_denominator = sum(target is None for target in targets)
    false_finite = sum(pred is not None and target is None for pred, target in zip(predictions, targets))
    errors = np.asarray([pred - target for pred, target in defined], dtype=float)
    return {"defined_count": len(defined), "target_defined_count": target_count, "missing_boundary_count": missing, "false_finite_boundary_count": false_finite, "boundary_rmse": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None, "boundary_mae": float(np.mean(np.abs(errors))) if len(errors) else None, "boundary_p95": float(np.percentile(np.abs(errors), 95)) if len(errors) else None, "missing_boundary_rate": float(missing / target_count) if target_count else None, "false_finite_boundary_rate": float(false_finite / false_finite_denominator) if false_finite_denominator else None}


def put_boundary_monotonicity(boundaries: Sequence[float | None]) -> int:
    return int(put_boundary_monotonicity_diagnostics(boundaries)["violation_count"])


def put_boundary_monotonicity_diagnostics(boundaries: Sequence[float | None]) -> dict[str, int | float | None]:
    """Count only physically adjacent defined transitions in a theta profile."""
    denominator = 0
    violations = 0
    missing = 0
    for left, right in zip(boundaries, boundaries[1:]):
        if left is None or right is None:
            missing += 1
            continue
        denominator += 1
        violations += int(float(right) > float(left) + 1e-12)
    return {"adjacent_transition_denominator": denominator, "violation_count": violations, "violation_rate": float(violations / denominator) if denominator else None, "missing_transition_count": missing}


def decide_m04_screen(rows: Sequence[Mapping[str, Any]], *, bootstrap_iterations: int, bootstrap_seed: int) -> dict[str, Any]:
    """Apply M04's paired complete-regime and Holm-corrected promotion rules."""
    from campaign.statistics import holm_adjust, summarize_paired_scenarios

    baseline_name = "price_surrogate_boundary_extraction"
    candidates = ("scalar_boundary_head", "implicit_level_set")
    split_rows = [row for row in rows if row.get("split", "validation") == "validation"]
    by_method: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for row in split_rows:
        method, scenario = str(row.get("method")), str(row.get("scenario_id"))
        by_method.setdefault(method, {}).setdefault(scenario, []).append(row)
    baseline = by_method.get(baseline_name, {})
    comparisons: dict[str, dict[str, Any]] = {}
    raw_p_values: list[float] = []
    ordered_candidates: list[str] = []
    for method in candidates:
        candidate = by_method.get(method, {})
        scenario_ids = sorted(set(baseline).intersection(candidate))
        def aggregate(values: Sequence[Mapping[str, Any]], metric: str) -> float | None:
            payload = [float(item[metric]) for item in values if item.get(metric) is not None and np.isfinite(float(item[metric]))]
            return float(np.mean(payload)) if payload else None
        paired = [(scenario, aggregate(baseline[scenario], "boundary_rmse"), aggregate(candidate[scenario], "boundary_rmse")) for scenario in scenario_ids]
        paired = [(scenario, base, other) for scenario, base, other in paired if base is not None and other is not None]
        diagnostics_available = bool(paired) and all(all(item.get("oracle_status") == "defined" and item.get("monotonicity_violation_rate") is not None and item.get("missing_boundary_rate") is not None for item in baseline[s]) and all(item.get("oracle_status") == "defined" and item.get("monotonicity_violation_rate") is not None and item.get("missing_boundary_rate") is not None for item in candidate[s]) for s, _, _ in paired)
        if len(paired) < 2 or not diagnostics_available:
            comparisons[method] = {"status": "insufficient_diagnostics", "scenario_count": len(paired), "reason": "complete paired oracle metrics unavailable", "holm_adjusted_p_value": None}
            continue
        summary = summarize_paired_scenarios({s: [base] for s, base, _ in paired}, {s: [other] for s, _, other in paired}, lower_is_better=True, bootstrap_iterations=bootstrap_iterations, random_seed=bootstrap_seed)
        baseline_missing = np.mean([aggregate(baseline[s], "missing_boundary_rate") or 0.0 for s, _, _ in paired])
        candidate_missing = np.mean([aggregate(candidate[s], "missing_boundary_rate") or 0.0 for s, _, _ in paired])
        baseline_monotonicity = np.mean([aggregate(baseline[s], "monotonicity_violation_rate") for s, _, _ in paired])
        candidate_monotonicity = np.mean([aggregate(candidate[s], "monotonicity_violation_rate") for s, _, _ in paired])
        comparisons[method] = {"status": "pending_holm", "scenario_count": summary.scenario_count, "relative_percent_improvement": summary.relative_percent_change, "confidence_interval_difference": list(summary.confidence_interval), "paired_p_value": summary.paired_p_value, "holm_adjusted_p_value": None, "missing_boundary_worsening": float(candidate_missing - baseline_missing), "monotonicity_worsening": float(candidate_monotonicity - baseline_monotonicity)}
        raw_p_values.append(summary.paired_p_value)
        ordered_candidates.append(method)
    if raw_p_values:
        adjusted = holm_adjust(raw_p_values)
        for method, adjusted_p, reject in zip(ordered_candidates, adjusted.adjusted_p_values, adjusted.reject):
            comparison = comparisons[method]
            comparison["holm_adjusted_p_value"] = adjusted_p
            comparison["holm_reject"] = reject
            interval = comparison["confidence_interval_difference"]
            supported = comparison["relative_percent_improvement"] >= 2.0 and interval[1] < 0.0 and reject and comparison["missing_boundary_worsening"] <= 0.02 and comparison["monotonicity_worsening"] <= 0.02
            comparison["status"] = "candidate_supported" if supported else "failed_metric_gate"
    return {"selection_split": "validation", "legacy_test_used_for_selection": False, "comparisons": comparisons}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fallback_fields: Sequence[str]) -> None:
    fields = tuple(dict.fromkeys(key for row in rows for key in row)) or tuple(fallback_fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None or (isinstance(value, float) and not np.isfinite(value)) else value for key, value in row.items()})


def write_m04_outputs(output_dir: str | Path, *, config: Mapping[str, Any], run_rows: Sequence[Mapping[str, Any]], scenario_rows: Sequence[Mapping[str, Any]], profiles: Sequence[Mapping[str, Any]], timing_rows: Sequence[Mapping[str, Any]], history: Sequence[Mapping[str, Any]], decision: Mapping[str, Any] | None = None) -> None:
    """Write M04's compact, finite standard evidence schema and hash it."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    methods = sorted({str(row.get("method", "")) for row in run_rows if row.get("method")})
    seeds = sorted({int(row["seed"]) for row in run_rows if row.get("seed") is not None})
    config_copy = dict(config)
    split_ids = config_copy.get("split_ids")
    manifest = {"branch": config_copy.get("branch"), "commit": config_copy.get("commit"), "method_spec": config_copy.get("method_spec", "M04 selective free-boundary screen"), "config": config_copy, "config_sha256": hashlib.sha256(json.dumps(_json_safe(config_copy), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(), "dataset_sha256": config_copy.get("dataset_sha256"), "split_ids": split_ids, "split_counts": config_copy.get("split_counts_verified", config_copy.get("split_counts")), "regime_counts": {str(split): len(ids) for split, ids in split_ids.items()} if isinstance(split_ids, Mapping) else None, "normalizations": config_copy.get("normalizations"), "parameter_counts": {str(row.get("method")): int(row["parameter_count"]) for row in run_rows if row.get("method") and row.get("parameter_count") is not None}, "seeds": config_copy.get("seeds", seeds), "blind_holdout_accessed": False}
    _write_json(root / "config_manifest.json", manifest)
    _write_csv(root / "run_metrics.csv", run_rows, ("method", "seed", "split", "status", "boundary_rmse"))
    _write_csv(root / "scenario_metrics.csv", scenario_rows, ("method", "seed", "scenario_id", "split", "status"))
    _write_csv(root / "boundary_profiles.csv", profiles, ("method", "seed", "scenario_id", "split", "theta", "prediction_boundary", "prediction_status", "prediction_reason", "oracle_boundary", "oracle_status", "oracle_reason"))
    _write_csv(root / "timing_samples.csv", timing_rows, ("method", "seed", "category", "sample", "seconds"))
    _write_csv(root / "training_history.csv", history, ("method", "seed", "epoch", "training_loss"))
    resolved_decision = dict(decision or {"status": "diagnostic_only", "reason": "screen_not_run"})
    resolved_decision["blind_holdout_accessed"] = False
    _write_json(root / "decision.json", resolved_decision)
    _write_json(root / "screen_summary.json", {"methods": methods, "seeds": seeds, "run_rows": len(run_rows), "scenario_rows": len(scenario_rows), "blind_holdout_accessed": False, "decision": resolved_decision})
    files = [path for path in root.iterdir() if path.is_file() and path.name != "artifact_hashes.json"]
    _write_json(root / "artifact_hashes.json", {"algorithm": "sha256", "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}})


def _torch_predict(model: Any, features: np.ndarray) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(features, dtype=torch.float64)).detach().cpu().numpy().reshape(-1)


def _fit_fixed_epochs(model: Any, features: np.ndarray, targets: np.ndarray, *, epochs: int, batch_size: int, learning_rate: float, seed: int) -> tuple[Any, list[dict[str, Any]]]:
    """Train fixed epochs only. There is intentionally no validation input or checkpoint selection."""
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(seed)
    model = model.to(dtype=torch.float64, device="cpu")
    x, y = torch.as_tensor(features, dtype=torch.float64), torch.as_tensor(targets[:, None], dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(x), generator=generator)
        loss_sum = 0.0
        for start in range(0, len(x), batch_size):
            batch = order[start : start + batch_size]
            prediction = model(x[batch])
            loss = functional.mse_loss(prediction, y[batch])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("M04 training loss is non-finite")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
        history.append({"epoch": epoch, "training_loss": loss_sum / len(x)})
    return model, history


def _surface_training_rows(scenarios: Sequence[SurfaceScenario], method: str) -> tuple[np.ndarray, np.ndarray]:
    inputs, targets = [], []
    for scenario in scenarios:
        if method == "price_surrogate_boundary_extraction":
            inputs.append(scenario.features[:, :6])
            targets.append(scenario.labels[:, 1])
        elif method == "implicit_level_set":
            inputs.append(scenario.features[:, :5])
            targets.append(np.where(scenario.labels[:, 1] > 1e-10, 1.0, -1.0))
        elif method == "scalar_boundary_head":
            premium = scenario.labels[:, 1].reshape(len(scenario.theta), len(scenario.spots))
            for index, theta in enumerate(scenario.theta):
                boundary = extract_premium_boundary(scenario.spots, premium[index], strike=scenario.strike, theta=float(theta))
                if boundary.value is not None:
                    first = scenario.features[index * len(scenario.spots)]
                    inputs.append(np.array([[theta, first[2], first[3], first[4]]], dtype=np.float64))
                    targets.append(np.array([boundary.value / scenario.strike], dtype=np.float64))
        else:
            raise ValueError(f"unknown M04 method: {method}")
    if not inputs:
        raise ValueError(f"M04 {method} has no complete training labels")
    return np.concatenate(inputs, axis=0), np.concatenate(targets, axis=0)


def _predict_raw_surface(model: Any, normalizer: TrainingNormalizer, scenario: SurfaceScenario, method: str) -> np.ndarray:
    if method == "price_surrogate_boundary_extraction":
        return _torch_predict(model, normalizer.transform(scenario.features[:, :6])).reshape(len(scenario.theta), len(scenario.spots))
    if method == "implicit_level_set":
        return _torch_predict(model, normalizer.transform(scenario.features[:, :5])).reshape(len(scenario.theta), len(scenario.spots))
    rows = np.asarray([[theta, scenario.rate, scenario.dividend, scenario.volatility] for theta in scenario.theta], dtype=np.float64)
    return _torch_predict(model, normalizer.transform(rows)) * scenario.strike


def _extract_raw_surface(raw: np.ndarray, scenario: SurfaceScenario, method: str, *, premium_tolerance: float = 1e-10) -> tuple[BoundaryExtraction, ...]:
    if method == "price_surrogate_boundary_extraction":
        return tuple(extract_premium_boundary(scenario.spots, raw[index], strike=scenario.strike, theta=float(theta), tolerance=premium_tolerance) for index, theta in enumerate(scenario.theta))
    if method == "implicit_level_set":
        return tuple(BoundaryExtraction(float(scenario.strike), "defined", None, (float(scenario.strike), float(scenario.strike)), "american_put_maturity_convention") if theta == 0.0 else extract_level_set_boundary(scenario.spots, raw[index]) for index, theta in enumerate(scenario.theta))
    return tuple(BoundaryExtraction(float(scenario.strike), "defined", None, (float(scenario.strike), float(scenario.strike)), "american_put_maturity_convention") if theta == 0.0 else BoundaryExtraction(float(value), "defined", None, None, "scalar_boundary_head") for value, theta in zip(raw, scenario.theta))


def _predict_surface(model: Any, normalizer: TrainingNormalizer, scenario: SurfaceScenario, method: str, *, premium_tolerance: float = 1e-10) -> tuple[BoundaryExtraction, ...]:
    return _extract_raw_surface(_predict_raw_surface(model, normalizer, scenario, method), scenario, method, premium_tolerance=premium_tolerance)


def run_m04_screen(*, config_path: str | Path, output_root: str | Path | None = None, smoke: bool = False) -> dict[str, Any]:
    """Execute the frozen M04 screen; callers must obtain manager approval before canonical use."""
    from common import load_config
    from models.free_boundary import ImplicitPutLevelSet, PositivePremiumSurrogate, ScalarPutBoundaryHead
    from models.training import set_global_seed

    cfg = load_config(config_path)
    root = Path(output_root or cfg["output_root"])
    budget = M04Budget(float(cfg.get("screen_budget_seconds", 0)))
    if cfg.get("device") != "cpu" or cfg.get("training_dtype") != "float64":
        raise ValueError("M04 requires CPU float64")
    if list(cfg.get("seeds", ())) != [101, 203, 307, 409, 503] or int(cfg.get("screen_budget_seconds", 0)) != 3600:
        raise ValueError("M04 seeds and budget differ from the frozen protocol")
    timing = cfg.get("timing", {})
    if timing != {"warmups": 5, "repetitions": 30}:
        raise ValueError("M04 timing protocol must use five warmups and 30 repetitions")
    if smoke:
        raise ValueError("M04 smoke is intentionally disabled until the real dataset and oracle gates are available")
    selections = validate_data_freeze(cfg)
    tolerance_candidates = tuple(float(value) for value in cfg.get("premium_contact_tolerances", ()))
    if not tolerance_candidates or len(tolerance_candidates) > 5:
        raise ValueError("M04 permits at most five predeclared premium contact tolerances")
    dataset_path = Path(cfg["dataset"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"M04 real dataset unavailable: {dataset_path}")
    if hashlib.sha256(dataset_path.read_bytes()).hexdigest() != cfg["dataset_sha256"]:
        raise ValueError("M04 dataset SHA-256 does not match the frozen config")
    preprocessing_started = perf_counter()
    dataset = load_complete_surface_dataset(dataset_path, selections)
    methods = ("price_surrogate_boundary_extraction", "scalar_boundary_head", "implicit_level_set")
    run_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    premium_thresholds: dict[str, Any] = {}

    def resource_blocked(checkpoint: str) -> dict[str, Any]:
        decision = {"status": "resource_blocked", "reason": f"M04 budget exceeded at {checkpoint}", "budget_seconds": budget.limit_seconds, "elapsed_seconds": budget.elapsed_seconds, "budget_categories_seconds": budget.category_seconds, "blind_holdout_accessed": False}
        write_m04_outputs(root, config={**cfg, "blind_holdout_accessed": False}, run_rows=run_rows, scenario_rows=scenario_rows, profiles=profiles, timing_rows=timing_rows, history=history, decision=decision)
        return {"status": "resource_blocked", "output_dir": str(root), "budget": budget.category_seconds, "blind_holdout_accessed": False}

    budget.record("preprocessing", perf_counter() - preprocessing_started)
    if budget.checkpoint("before_oracle_generation"):
        return resource_blocked("before_oracle_generation")
    oracle_started = perf_counter()
    oracles = {scenario.scenario_id: build_refined_oracle(scenario, coarse_grid=int(cfg["oracle_coarse_grid"]), fine_grid=int(cfg["oracle_fine_grid"]), tolerance=float(cfg["oracle_solver_tolerance"]), max_refinement_rmse=float(cfg["oracle_max_refinement_rmse"])) for split in ("validation", "legacy_test") for scenario in dataset.splits[split]}
    budget.record("oracle_generation", perf_counter() - oracle_started)
    if budget.checkpoint("before_model"):
        return resource_blocked("before_model")
    normalizations: dict[str, Any] = {}
    timing_rows.append({"method": "oracle", "seed": "", "category": "oracle_generation", "sample": 0, "seconds": budget.category_seconds["oracle_generation"]})
    timing_rows.append({"method": "preprocessing", "seed": "", "category": "preprocessing", "sample": 0, "seconds": budget.category_seconds["preprocessing"]})
    for seed in cfg["seeds"]:
        for method in methods:
            if budget.checkpoint("before_model_seed"):
                return resource_blocked("before_model_seed")
            set_global_seed(int(seed))
            train_x, train_y = _surface_training_rows(dataset.splits["train"], method)
            normalizer = TrainingNormalizer.fit(train_x)
            normalizations[method] = {"mean": normalizer.mean.tolist(), "scale": normalizer.scale.tolist(), "fitted_split": "train"}
            if method == "price_surrogate_boundary_extraction":
                model = PositivePremiumSurrogate.build(6, int(cfg["hidden"]), int(cfg["depth"]))
            elif method == "scalar_boundary_head":
                model = ScalarPutBoundaryHead.build(4, int(cfg["hidden"]), int(cfg["depth"]))
            else:
                model = ImplicitPutLevelSet.build(5, int(cfg["hidden"]), int(cfg["depth"]))
            started = perf_counter()
            model, local_history = _fit_fixed_epochs(model, normalizer.transform(train_x), train_y, epochs=int(cfg["epochs"]), batch_size=int(cfg["batch_size"]), learning_rate=float(cfg["learning_rate"]), seed=int(seed))
            train_seconds = perf_counter() - started
            budget.record("training", train_seconds)
            timing_rows.append({"method": method, "seed": seed, "category": "training", "sample": 0, "seconds": train_seconds})
            history.extend({"method": method, "seed": seed, **item} for item in local_history)
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            premium_tolerance = 1e-10
            if method == "price_surrogate_boundary_extraction":
                training_profiles: list[tuple[np.ndarray, float | None, float]] = []
                for scenario in dataset.splits["train"]:
                    raw_training = _predict_raw_surface(model, normalizer, scenario, method)
                    labels = scenario.labels[:, 1].reshape(len(scenario.theta), len(scenario.spots))
                    for index, theta in enumerate(scenario.theta):
                        target = extract_premium_boundary(scenario.spots, labels[index], strike=scenario.strike, theta=float(theta), tolerance=0.0)
                        training_profiles.append((raw_training[index], target.value, float(theta)))
                selected = select_premium_contact_tolerance(training_profiles, spots=dataset.splits["train"][0].spots, strike=dataset.splits["train"][0].strike, candidates=tolerance_candidates)
                premium_tolerance = float(selected["selected_threshold"])
                premium_thresholds[str(seed)] = selected
            for split in ("validation", "legacy_test"):
                for scenario in dataset.splits[split]:
                    oracle = oracles[scenario.scenario_id]
                    for sample in range(5 + 30):
                        if budget.checkpoint("before_timing_block"):
                            return resource_blocked("before_timing_block")
                        inference_started = perf_counter()
                        raw = _predict_raw_surface(model, normalizer, scenario, method)
                        inference_seconds = perf_counter() - inference_started
                        budget.record("inference", inference_seconds)
                        extraction_started = perf_counter()
                        predicted = _extract_raw_surface(raw, scenario, method, premium_tolerance=premium_tolerance)
                        extraction_seconds = perf_counter() - extraction_started
                        budget.record("extraction", extraction_seconds)
                        if sample >= 5:
                            timing_rows.append({"method": method, "seed": seed, "scenario_id": scenario.scenario_id, "category": "inference", "sample": sample - 5, "seconds": inference_seconds})
                            timing_rows.append({"method": method, "seed": seed, "scenario_id": scenario.scenario_id, "category": "extraction", "sample": sample - 5, "seconds": extraction_seconds})
                            timing_rows.append({"method": method, "seed": seed, "scenario_id": scenario.scenario_id, "category": "fallback", "sample": sample - 5, "seconds": 0.0})
                    targets = [item.value for item in oracle.values] if oracle.status == "defined" else [None] * len(predicted)
                    metrics = aggregate_boundary_metrics(predictions=[item.value for item in predicted], targets=targets, statuses=[item.status for item in predicted])
                    monotonicity = put_boundary_monotonicity_diagnostics([item.value for item in predicted])
                    import resource
                    memory_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
                    row = {"method": method, "seed": seed, "scenario_id": scenario.scenario_id, "split": split, "oracle_status": oracle.status, "oracle_reason": oracle.reason, "parameter_count": parameter_count, "training_seconds": train_seconds, "fallback_seconds": 0.0, "memory_mb": memory_mb, "limitation": "scalar_head_cannot_represent_no_finite_boundary" if method == "scalar_boundary_head" else None, **monotonicity, **{f"monotonicity_{key}": value for key, value in monotonicity.items()}, **metrics}
                    scenario_rows.append(row)
                    oracle_profile_values = oracle.values if len(oracle.values) == len(predicted) else tuple(BoundaryExtraction(None, "oracle_failed", oracle.reason, None, "oracle_unavailable") for _ in predicted)
                    profiles.extend({"method": method, "seed": seed, "scenario_id": scenario.scenario_id, "split": split, "theta": float(theta), "prediction_boundary": prediction.value, "prediction_status": prediction.status, "prediction_reason": prediction.reason, "oracle_boundary": target.value, "oracle_status": target.status, "oracle_reason": target.reason} for theta, prediction, target in zip(scenario.theta, predicted, oracle_profile_values))
            validation = [row for row in scenario_rows if row["method"] == method and row["seed"] == seed and row["split"] == "validation"]
            finite = [float(row["boundary_rmse"]) for row in validation if row["boundary_rmse"] is not None]
            boundary_only = method != "price_surrogate_boundary_extraction"
            run_rows.append({"method": method, "seed": seed, "split": "validation", "status": "diagnostic_only", "boundary_rmse": float(np.mean(finite)) if finite else None, "parameter_count": parameter_count, "training_seconds": train_seconds, "price_diagnostic": "not_applicable_by_design" if boundary_only else None, "delta_diagnostic": "not_applicable_by_design" if boundary_only else None, "gamma_diagnostic": "not_applicable_by_design" if boundary_only else None, "lcp_diagnostic": "not_applicable_by_design" if boundary_only else None})
    if budget.checkpoint("before_output_preparation"):
        return resource_blocked("before_output_preparation")
    decision = {**decide_m04_screen(scenario_rows, bootstrap_iterations=int(cfg["bootstrap_iterations"]), bootstrap_seed=int(cfg["bootstrap_seed"])), "status": "diagnostic_only", "reason": "screen evidence requires manager review; no promotion is asserted", "budget_state": budget.state, "budget_seconds": budget.limit_seconds, "elapsed_seconds": budget.elapsed_seconds, "budget_categories_seconds": budget.category_seconds, "blind_holdout_accessed": False}
    branch = subprocess.check_output(("git", "branch", "--show-current"), text=True).strip()
    commit = subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()
    output_started = perf_counter()
    write_m04_outputs(root, config={**cfg, "branch": branch, "commit": commit, "method_spec": "M04 selective free-boundary screen", "split_ids": selections, "split_counts_verified": dataset.counts, "normalizations": normalizations, "premium_contact_thresholds": premium_thresholds, "blind_holdout_accessed": False}, run_rows=run_rows, scenario_rows=scenario_rows, profiles=profiles, timing_rows=timing_rows, history=history, decision=decision)
    budget.record("output_preparation", perf_counter() - output_started)
    return {"status": decision["status"], "output_dir": str(root), "budget": budget.category_seconds, "blind_holdout_accessed": False}
