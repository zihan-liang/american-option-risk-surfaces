"""Public commitment verification and deterministic blind-holdout regimes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import qmc


GENERATOR_VERSION = "campaign-holdout-regimes-v1"
COMMITMENT_PREFIX = "american-option-risk-surfaces:independent-methods:v1:"
IN_DOMAIN_COUNT = 96
STRESS_COUNT = 32
PARAMETER_BOUNDS = {
    "T": (0.05, 2.0),
    "r": (-0.01, 0.10),
    "q": (0.0, 0.08),
    "sigma": (0.05, 1.0),
}
_PARAMETER_NAMES = tuple(PARAMETER_BOUNDS)
_STRESS_VALUES = {
    "T": (0.01, 0.05, 2.0, 4.0),
    "r": (-0.02, 0.12),
    "q": (0.0, 0.10),
    "sigma": (0.03, 1.25),
}


def canonical_commitment_bytes(seed: int) -> bytes:
    """Return the committed UTF-8 representation for an evaluator seed."""
    _validate_seed(seed)
    return f"{COMMITMENT_PREFIX}{seed}\n".encode("utf-8")


def verify_public_commitment(seed: int, public_commitment_sha256: str) -> bool:
    """Return whether a supplied evaluator seed matches the public commitment."""
    expected = _normalize_sha256(public_commitment_sha256)
    actual = hashlib.sha256(canonical_commitment_bytes(seed)).hexdigest()
    return actual == expected


def generate_holdout_regimes(seed: int) -> tuple[dict[str, Any], ...]:
    """Generate 96 scrambled Sobol regimes and 32 fixed stress regimes."""
    _validate_seed(seed)
    sobol = qmc.Sobol(d=len(_PARAMETER_NAMES), scramble=True, seed=seed)
    samples = sobol.random_base2(m=7)[:IN_DOMAIN_COUNT]
    regimes = [_sobol_regime(index, sample) for index, sample in enumerate(samples, start=1)]
    regimes.extend(_stress_regimes())
    _validate_generated_regimes(regimes)
    return tuple(regimes)


def canonical_regime_bytes(regimes: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize generated regimes with a stable JSON representation for hashing."""
    return json.dumps(
        list(regimes),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def build_holdout_manifest(
    regimes: Sequence[Mapping[str, Any]], *, public_commitment_sha256: str
) -> dict[str, Any]:
    """Build the public provenance manifest for a generated holdout set."""
    _validate_generated_regimes(regimes)
    source_counts = {
        "sobol_in_domain": sum(regime["source"] == "sobol_in_domain" for regime in regimes),
        "stress": sum(regime["source"] == "stress" for regime in regimes),
    }
    if source_counts != {"sobol_in_domain": IN_DOMAIN_COUNT, "stress": STRESS_COUNT}:
        raise ValueError("holdout regimes do not match the required source counts")
    return {
        "generator_version": GENERATOR_VERSION,
        "counts": {"total": len(regimes), **source_counts},
        "parameter_bounds": {
            name: {"min": bounds[0], "max": bounds[1]}
            for name, bounds in PARAMETER_BOUNDS.items()
        },
        "public_commitment_sha256": _normalize_sha256(public_commitment_sha256),
        "canonical_regimes_sha256": hashlib.sha256(canonical_regime_bytes(regimes)).hexdigest(),
    }


def _sobol_regime(index: int, sample: np.ndarray) -> dict[str, Any]:
    parameters = {
        name: round(lower + float(sample[position]) * (upper - lower), 12)
        for position, (name, (lower, upper)) in enumerate(PARAMETER_BOUNDS.items())
    }
    return _regime(f"blind-holdout-sobol-{index:03d}", "sobol_in_domain", parameters)


def _stress_regimes() -> list[dict[str, Any]]:
    regimes = []
    index = 1
    for maturity in _STRESS_VALUES["T"]:
        for rate in _STRESS_VALUES["r"]:
            for dividend in _STRESS_VALUES["q"]:
                for volatility in _STRESS_VALUES["sigma"]:
                    regimes.append(
                        _regime(
                            f"blind-holdout-stress-{index:03d}",
                            "stress",
                            {"T": maturity, "r": rate, "q": dividend, "sigma": volatility},
                        )
                    )
                    index += 1
    return regimes


def _regime(scenario_id: str, source: str, parameters: Mapping[str, float]) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "source": source,
        "option_type": "put",
        "strike": 1.0,
        **parameters,
    }


def _validate_generated_regimes(regimes: Sequence[Mapping[str, Any]]) -> None:
    if len(regimes) != IN_DOMAIN_COUNT + STRESS_COUNT:
        raise ValueError("holdout must contain exactly 128 regimes")
    scenario_ids = [regime.get("scenario_id") for regime in regimes]
    parameter_tuples = [
        tuple(regime.get(name) for name in (*_PARAMETER_NAMES, "strike", "option_type"))
        for regime in regimes
    ]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("holdout scenario IDs must be unique")
    if len(parameter_tuples) != len(set(parameter_tuples)):
        raise ValueError("holdout parameter tuples must be unique")
    if any(regime.get("option_type") != "put" or regime.get("strike", 0) <= 0 for regime in regimes):
        raise ValueError("holdout regimes must be normalized positive-strike puts")


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("evaluator seed must be a non-negative integer")


def _normalize_sha256(value: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError("public commitment must be a SHA-256 hex digest")
    return normalized
