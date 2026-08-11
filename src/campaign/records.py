"""Validated, serializable records for individual campaign runs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional


class RunPhase(str, Enum):
    SCREEN = "screen"
    CONFIRM = "confirm"
    FINAL = "final"


class RunStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED_METRIC_GATE = "failed_metric_gate"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    RESOURCE_BLOCKED = "resource_blocked"
    SOFTWARE_FAILED = "software_failed"
    NOT_IMPLEMENTED = "not_implemented"


def _is_finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


@dataclass(frozen=True)
class MethodRunRecord:
    """Canonical decision and provenance state for a single method phase."""

    method_id: str
    phase: RunPhase
    status: RunStatus
    config_id: str = ""
    primary_metric_value: Optional[float] = None
    baseline_metric_value: Optional[float] = None
    optional_metrics: Mapping[str, Optional[float]] = field(default_factory=dict)
    scenario_count: int = 0
    seed: Optional[int] = None
    split: str = ""
    branch: str = ""
    commit: str = ""
    timings_seconds: Mapping[str, Optional[float]] = field(default_factory=dict)
    memory_mb: Optional[float] = None
    claim_status: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not re.fullmatch(r"M(0[1-9]|1[0-9]|2[0-4])", self.method_id):
            raise ValueError("method_id must be exactly M01 through M24")
        if not isinstance(self.phase, RunPhase):
            object.__setattr__(self, "phase", RunPhase(self.phase))
        if not isinstance(self.status, RunStatus):
            object.__setattr__(self, "status", RunStatus(self.status))
        if self.scenario_count < 0:
            raise ValueError("scenario_count must be non-negative")
        if self.status is RunStatus.COMPLETED:
            if not _is_finite(self.primary_metric_value):
                raise ValueError("completed records require a finite primary_metric_value")
            if not _is_finite(self.baseline_metric_value):
                raise ValueError("completed records require a finite baseline_metric_value")
        for name, value in self.optional_metrics.items():
            if value not in (None, "") and not _is_finite(value):
                raise ValueError(f"optional_metrics[{name}] must be finite when supplied")
        for name, value in self.timings_seconds.items():
            if value not in (None, "") and (not _is_finite(value) or float(value) < 0):
                raise ValueError(f"timings_seconds[{name}] must be finite and non-negative when supplied")
        if self.memory_mb not in (None, "") and (not _is_finite(self.memory_mb) or float(self.memory_mb) < 0):
            raise ValueError("memory_mb must be finite and non-negative when supplied")

    def to_dict(self) -> dict[str, Any]:
        """Produce a stable, JSON-ready representation with enum values flattened."""
        return {
            "baseline_metric_value": self.baseline_metric_value,
            "branch": self.branch,
            "claim_status": self.claim_status,
            "commit": self.commit,
            "config_id": self.config_id,
            "memory_mb": self.memory_mb,
            "method_id": self.method_id,
            "notes": self.notes,
            "optional_metrics": dict(sorted(self.optional_metrics.items())),
            "phase": self.phase.value,
            "primary_metric_value": self.primary_metric_value,
            "scenario_count": self.scenario_count,
            "seed": self.seed,
            "split": self.split,
            "status": self.status.value,
            "timings_seconds": dict(sorted(self.timings_seconds.items())),
        }


METHOD_RUN_CSV_FIELDS = (
    "method_id",
    "phase",
    "status",
    "config_id",
    "primary_metric_value",
    "baseline_metric_value",
    "optional_metrics_json",
    "scenario_count",
    "seed",
    "split",
    "branch",
    "commit",
    "timings_seconds_json",
    "memory_mb",
    "claim_status",
    "notes",
)
