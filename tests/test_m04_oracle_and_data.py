from __future__ import annotations

import numpy as np
import pytest


def test_complete_surface_loader_rejects_partial_rectangles_and_renames_legacy_test(tmp_path) -> None:
    from campaign.m04 import load_complete_surface_dataset

    path = tmp_path / "partial.npz"
    _write_dataset(path, partial_validation=True)
    with pytest.raises(ValueError, match="complete rectangular surface"):
        load_complete_surface_dataset(path, {"train": ["train"], "validation": ["validation"], "legacy_test": ["test"]})

    complete = tmp_path / "complete.npz"
    _write_dataset(complete)
    loaded = load_complete_surface_dataset(complete, {"train": ["train"], "validation": ["validation"], "legacy_test": ["test"]})
    assert loaded.splits["legacy_test"][0].split == "legacy_test"
    assert loaded.counts == {"train": 6, "validation": 6, "legacy_test": 6}


def test_boundary_extractor_handles_maturity_and_structured_nonfinite_cases() -> None:
    from campaign.m04 import extract_premium_boundary

    maturity = extract_premium_boundary([0.5, 0.8, 1.0], [0.0, 0.0, 0.0], strike=1.0, theta=0.0)
    no_exercise = extract_premium_boundary([0.5, 0.8, 1.0], [0.1, 0.2, 0.3], strike=1.0, theta=0.5)

    assert (maturity.status, maturity.value) == ("defined", 1.0)
    assert (no_exercise.status, no_exercise.value) == ("no_exercise_region", None)


def _write_dataset(path, *, partial_validation: bool = False) -> None:
    rows = []
    for regime, split in (("train", "train"), ("validation", "validation"), ("test", "test")):
        for theta in (0.0, 0.5):
            for spot in (0.5, 0.8, 1.1):
                if partial_validation and regime == "validation" and theta == 0.5 and spot == 1.1:
                    continue
                payoff = max(1.0 - spot, 0.0)
                premium = max(spot - 0.75, 0.0) ** 2
                rows.append(([np.log(spot), theta, 0.05, 0.0, 0.2, 1.0, 0.0], [payoff + premium, premium, payoff, 0.0, 0.0], regime, split))
    np.savez_compressed(
        path,
        features=np.asarray([row[0] for row in rows]),
        labels=np.asarray([row[1] for row in rows]),
        regime_key=np.asarray([row[2] for row in rows]),
        split=np.asarray([row[3] for row in rows]),
    )
