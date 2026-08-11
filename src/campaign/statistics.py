"""Scenario-level statistics for the independent-method campaign."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from time import perf_counter
from typing import Any, Callable, Union

import numpy as np


@dataclass(frozen=True)
class ScenarioObservation:
    """Grid-point observations belonging to one complete scenario ID."""

    scenario_id: str
    values: Sequence[float]


@dataclass(frozen=True)
class PairedScenarioSummary:
    """Paired scenario statistics with deterministic bootstrap inference."""

    scenario_ids: tuple[str, ...]
    scenario_count: int
    mean_difference: float
    sample_standard_deviation: float
    confidence_interval: tuple[float, float]
    paired_p_value: float
    relative_percent_change: float
    lower_is_better: bool


@dataclass(frozen=True)
class HolmResult:
    """Holm-adjusted p-values and step-down rejection decisions."""

    adjusted_p_values: tuple[float, ...]
    reject: tuple[bool, ...]


@dataclass(frozen=True)
class MetricGate:
    """One metric's direction-aware promotion evidence."""

    name: str
    relative_percent_change: float
    confidence_interval: tuple[float, float]
    lower_is_better: bool
    is_residual: bool = False
    is_primary: bool = False


@dataclass(frozen=True)
class PromotionDecision:
    """Campaign promotion state derived from all metric gates."""

    status: str
    failed_metrics: tuple[str, ...]
    improved_primary_metrics: tuple[str, ...]


@dataclass(frozen=True)
class TimingSummary:
    """Raw timing samples and stable summaries for campaign reporting."""

    samples: tuple[float, ...]
    median_seconds: float
    median_absolute_deviation_seconds: float
    p95_seconds: float


ObservationInput = Union[
    Mapping[str, Any],
    Iterable[Union[ScenarioObservation, tuple[str, Any]]],
]


def summarize_paired_scenarios(
    baseline: ObservationInput,
    candidate: ObservationInput,
    *,
    lower_is_better: bool,
    bootstrap_iterations: int = 10_000,
    random_seed: int = 0,
) -> PairedScenarioSummary:
    """Aggregate scenarios before resampling for paired campaign statistics.

    The confidence interval resamples complete scenario differences. The
    two-sided p-value reuses those draws after centering the differences at
    their observed mean, which supplies the paired null distribution.
    """
    if not isinstance(lower_is_better, bool):
        raise ValueError("lower_is_better must be a boolean")
    baseline_means = _scenario_means(baseline, "baseline")
    candidate_means = _scenario_means(candidate, "candidate")
    if set(baseline_means) != set(candidate_means):
        missing_from_candidate = sorted(set(baseline_means) - set(candidate_means))
        missing_from_baseline = sorted(set(candidate_means) - set(baseline_means))
        raise ValueError(
            "baseline and candidate must have identical scenario IDs; "
            f"missing_from_candidate={missing_from_candidate}, "
            f"missing_from_baseline={missing_from_baseline}"
        )
    if not isinstance(bootstrap_iterations, Integral) or isinstance(bootstrap_iterations, bool):
        raise ValueError("bootstrap_iterations must be a positive integer")
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be a positive integer")
    if not isinstance(random_seed, Integral) or isinstance(random_seed, bool):
        raise ValueError("random_seed must be an integer")
    if random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer")

    scenario_ids = tuple(baseline_means)
    if len(scenario_ids) < 2:
        raise ValueError("at least two scenarios are required for sample standard deviation")
    baseline_values = np.asarray([baseline_means[key] for key in scenario_ids], dtype=float)
    candidate_values = np.asarray([candidate_means[key] for key in scenario_ids], dtype=float)
    differences = candidate_values - baseline_values
    mean_difference = float(np.mean(differences))
    standard_deviation = float(np.std(differences, ddof=1))
    bootstrap_draws = _scenario_bootstrap_draws(
        differences.size, int(bootstrap_iterations), int(random_seed)
    )
    confidence_interval = _bootstrap_confidence_interval(differences, bootstrap_draws)
    paired_p_value = _paired_bootstrap_p_value(differences, bootstrap_draws)
    baseline_mean = float(np.mean(baseline_values))
    if baseline_mean == 0.0:
        raise ValueError("relative percentage change is undefined for a zero baseline mean")
    direction = -1.0 if lower_is_better else 1.0
    relative_percent_change = direction * mean_difference / abs(baseline_mean) * 100.0
    _require_finite(
        [
            mean_difference,
            standard_deviation,
            *confidence_interval,
            paired_p_value,
            relative_percent_change,
        ],
        "summary values",
    )
    return PairedScenarioSummary(
        scenario_ids=scenario_ids,
        scenario_count=differences.size,
        mean_difference=mean_difference,
        sample_standard_deviation=standard_deviation,
        confidence_interval=confidence_interval,
        paired_p_value=paired_p_value,
        relative_percent_change=float(relative_percent_change),
        lower_is_better=lower_is_better,
    )


def holm_adjust(p_values: Sequence[float], *, alpha: float = 0.05) -> HolmResult:
    """Apply the Holm step-down correction while retaining input order."""
    values = np.asarray(p_values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("p_values must not be empty")
    _require_finite(values, "p_values")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("p_values must lie in [0, 1]")
    _require_finite([alpha], "alpha")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    multiplier = np.arange(values.size, 0, -1, dtype=float)
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(sorted_values * multiplier))
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    _require_finite(adjusted, "adjusted p_values")
    return HolmResult(
        adjusted_p_values=tuple(float(value) for value in adjusted),
        reject=tuple(bool(value <= alpha) for value in adjusted),
    )


def decide_promotion(metrics: Iterable[MetricGate]) -> PromotionDecision:
    """Apply campaign no-regression and practical-improvement promotion gates."""
    metric_list = tuple(metrics)
    if not metric_list:
        raise ValueError("at least one metric gate is required")
    primary_metrics: list[MetricGate] = []
    failed_metrics: list[str] = []
    improved_primary_metrics: list[str] = []
    seen_names: set[str] = set()

    for metric in metric_list:
        _validate_metric_gate(metric, seen_names)
        if metric.is_primary:
            primary_metrics.append(metric)
        regression_limit = 5.0 if metric.is_residual else 2.0
        if metric.relative_percent_change < -regression_limit:
            failed_metrics.append(metric.name)
        if metric.is_primary and _is_claimed_improvement(metric):
            improved_primary_metrics.append(metric.name)

    if not primary_metrics:
        raise ValueError("at least one primary metric gate is required")
    if failed_metrics:
        status = "failed_metric_gate"
    elif len(improved_primary_metrics) == len(primary_metrics):
        status = "promote"
    else:
        status = "inconclusive_expand_to_ten_seeds"
    return PromotionDecision(
        status=status,
        failed_metrics=tuple(failed_metrics),
        improved_primary_metrics=tuple(improved_primary_metrics),
    )


def measure_timing(
    operation: Callable[[], Any],
    *,
    warmups: int = 5,
    repetitions: int = 30,
    clock: Callable[[], float] = perf_counter,
) -> TimingSummary:
    """Measure an operation with deterministic warm-up and repetition counts."""
    _validate_count(warmups, "warmups", allow_zero=True)
    _validate_count(repetitions, "repetitions", allow_zero=False)
    for _ in range(warmups):
        operation()

    samples: list[float] = []
    for _ in range(repetitions):
        started_at = float(clock())
        operation()
        elapsed = float(clock()) - started_at
        _require_finite([elapsed], "timing samples")
        if elapsed < 0.0:
            raise ValueError("timing samples must not be negative")
        samples.append(elapsed)

    sample_values = np.asarray(samples, dtype=float)
    median = float(np.median(sample_values))
    median_absolute_deviation = float(np.median(np.abs(sample_values - median)))
    p95 = float(np.quantile(sample_values, 0.95))
    _require_finite([median, median_absolute_deviation, p95], "timing summary")
    return TimingSummary(
        samples=tuple(samples),
        median_seconds=median,
        median_absolute_deviation_seconds=median_absolute_deviation,
        p95_seconds=p95,
    )


def _scenario_means(observations: ObservationInput, label: str) -> dict[str, float]:
    if isinstance(observations, Mapping):
        entries = observations.items()
    else:
        entries = observations

    means: dict[str, float] = {}
    for entry in entries:
        if isinstance(entry, ScenarioObservation):
            scenario_id, values = entry.scenario_id, entry.values
        else:
            try:
                scenario_id, values = entry
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label} observations must contain scenario ID/value pairs") from exc
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError(f"{label} scenario IDs must be non-empty strings")
        if scenario_id in means:
            raise ValueError(f"duplicate {label} scenario ID: {scenario_id}")
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size == 0:
            raise ValueError(f"{label} scenario {scenario_id!r} has no values")
        _require_finite(array, f"{label} scenario {scenario_id!r}")
        means[scenario_id] = float(np.mean(array))
    if not means:
        raise ValueError(f"{label} observations must not be empty")
    return means


def _validate_metric_gate(metric: MetricGate, seen_names: set[str]) -> None:
    if not isinstance(metric, MetricGate):
        raise ValueError("metrics must be MetricGate instances")
    if not isinstance(metric.name, str) or not metric.name:
        raise ValueError("metric names must be non-empty strings")
    if metric.name in seen_names:
        raise ValueError(f"duplicate metric name: {metric.name}")
    seen_names.add(metric.name)
    if not isinstance(metric.lower_is_better, bool):
        raise ValueError("lower_is_better must be a boolean")
    if not isinstance(metric.is_residual, bool) or not isinstance(metric.is_primary, bool):
        raise ValueError("metric flags must be booleans")
    confidence_interval = np.asarray(metric.confidence_interval, dtype=float).reshape(-1)
    if confidence_interval.size != 2:
        raise ValueError("confidence_interval must contain exactly two values")
    _require_finite(
        [metric.relative_percent_change, *confidence_interval],
        f"metric {metric.name!r}",
    )
    if confidence_interval[0] > confidence_interval[1]:
        raise ValueError("confidence_interval lower bound must not exceed upper bound")


def _is_claimed_improvement(metric: MetricGate) -> bool:
    interval_low, interval_high = metric.confidence_interval
    interval_is_beneficial = interval_high < 0.0 if metric.lower_is_better else interval_low > 0.0
    return metric.relative_percent_change >= 2.0 and interval_is_beneficial


def _validate_count(value: int, label: str, *, allow_zero: bool) -> None:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    lower_bound = 0 if allow_zero else 1
    if value < lower_bound:
        raise ValueError(f"{label} must be at least {lower_bound}")


def _scenario_bootstrap_draws(
    scenario_count: int, iterations: int, random_seed: int
) -> np.ndarray:
    generator = np.random.default_rng(random_seed)
    return generator.integers(0, scenario_count, size=(iterations, scenario_count))


def _bootstrap_confidence_interval(
    differences: np.ndarray, draws: np.ndarray
) -> tuple[float, float]:
    bootstrap_means = np.mean(differences[draws], axis=1)
    interval = np.quantile(bootstrap_means, [0.025, 0.975])
    _require_finite(interval, "bootstrap confidence interval")
    return float(interval[0]), float(interval[1])


def _paired_bootstrap_p_value(differences: np.ndarray, draws: np.ndarray) -> float:
    """Compute a two-sided scenario-bootstrap p-value under a centered null."""
    observed_mean = float(np.mean(differences))
    centered_differences = differences - observed_mean
    null_means = np.mean(centered_differences[draws], axis=1)
    extreme_count = int(np.count_nonzero(np.abs(null_means) >= abs(observed_mean)))
    p_value = (extreme_count + 1.0) / (draws.shape[0] + 1.0)
    _require_finite([p_value], "paired bootstrap p_value")
    if not 0.0 <= p_value <= 1.0:
        raise ValueError("paired bootstrap p_value must lie in [0, 1]")
    return float(p_value)


def _require_finite(values: Any, label: str) -> None:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
