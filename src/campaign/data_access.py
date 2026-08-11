"""Access boundaries that keep blind-holdout data out of development APIs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from campaign.holdout import generate_holdout_regimes, verify_public_commitment


DEVELOPMENT_SPLITS = frozenset({"train", "validation", "legacy_test"})
ALL_SPLITS = DEVELOPMENT_SPLITS | {"blind_holdout"}


class BlindHoldoutAccessError(PermissionError):
    """Raised when an API call would violate the blind-holdout policy."""


@dataclass(frozen=True, init=False)
class CampaignSplits:
    """Scenario partitions with phase-specific access methods."""

    train: Sequence[Mapping[str, Any]]
    validation: Sequence[Mapping[str, Any]]
    legacy_test: Sequence[Mapping[str, Any]]
    public_commitment_sha256: str

    def __init__(
        self,
        *,
        train: Sequence[Mapping[str, Any]],
        validation: Sequence[Mapping[str, Any]],
        legacy_test: Sequence[Mapping[str, Any]],
        public_commitment_sha256: str,
    ) -> None:
        object.__setattr__(self, "train", tuple(train))
        object.__setattr__(self, "validation", tuple(validation))
        object.__setattr__(self, "legacy_test", tuple(legacy_test))
        object.__setattr__(self, "public_commitment_sha256", public_commitment_sha256)
        scenario_ids = [
            scenario["scenario_id"]
            for scenarios in (self.train, self.validation, self.legacy_test)
            for scenario in scenarios
        ]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("scenario IDs must be disjoint across campaign splits")

    def development(self, split: str) -> tuple[Mapping[str, Any], ...]:
        """Return a development-safe split, rejecting blind holdout access."""
        return self._non_final_access(split)

    def model_selection(self, split: str) -> tuple[Mapping[str, Any], ...]:
        """Return a selection-safe split, rejecting blind holdout access."""
        return self._non_final_access(split)

    def final_evaluation(
        self, *, configs_frozen: bool, evaluator_seed: int | None
    ) -> tuple[Mapping[str, Any], ...]:
        """Release holdout scenarios only after freeze and commitment verification."""
        if not configs_frozen:
            raise BlindHoldoutAccessError("blind holdout is unavailable before configurations are frozen")
        if evaluator_seed is None:
            raise BlindHoldoutAccessError("blind holdout requires an evaluator seed")
        if not verify_public_commitment(evaluator_seed, self.public_commitment_sha256):
            raise BlindHoldoutAccessError("holdout commitment verification failed")
        return generate_holdout_regimes(evaluator_seed)

    def _non_final_access(self, split: str) -> tuple[Mapping[str, Any], ...]:
        if split == "blind_holdout":
            raise BlindHoldoutAccessError("blind holdout is unavailable outside final evaluation")
        if split not in DEVELOPMENT_SPLITS:
            raise ValueError(f"unknown campaign split: {split}")
        return getattr(self, split)
