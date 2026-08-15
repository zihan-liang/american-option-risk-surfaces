from __future__ import annotations

import pytest
import torch


def test_scalar_head_is_bounded_and_level_set_is_unconstrained() -> None:
    from models.free_boundary import ImplicitPutLevelSet, ScalarPutBoundaryHead

    scalar = ScalarPutBoundaryHead.build(input_dim=4, hidden=8, depth=1).to(dtype=torch.float64)
    implicit = ImplicitPutLevelSet.build(input_dim=5, hidden=8, depth=1).to(dtype=torch.float64)

    scalar_prediction = scalar(torch.randn((3, 4), dtype=torch.float64))
    implicit_prediction = implicit(torch.randn((3, 5), dtype=torch.float64))

    assert scalar_prediction.shape == implicit_prediction.shape == (3, 1)
    assert bool(torch.all((scalar_prediction >= 0.0) & (scalar_prediction <= 1.0)))
    assert implicit_prediction.dtype is torch.float64


def test_root_extraction_requires_one_supported_exercise_to_continuation_transition() -> None:
    from models.free_boundary import extract_put_zero_crossing

    defined = extract_put_zero_crossing([0.5, 0.7, 0.9], [-0.2, 0.0, 0.2])
    reversed_crossing = extract_put_zero_crossing([0.5, 0.7, 0.9], [0.2, -0.1, -0.2])
    multiple_crossings = extract_put_zero_crossing([0.5, 0.7, 0.9, 1.1], [-0.2, 0.2, -0.1, 0.1])

    assert (defined.status, defined.value, defined.reason) == ("defined", pytest.approx(0.7), None)
    assert reversed_crossing.status == "non_monotone_profile"
    assert reversed_crossing.value is None
    assert multiple_crossings.status == "non_monotone_profile"
    assert multiple_crossings.value is None


@pytest.mark.parametrize(
    ("levels", "status"),
    [([0.1, 0.2], "no_exercise_region"), ([-0.2, -0.1], "no_continuation_region")],
)
def test_root_extraction_never_invents_a_missing_boundary(levels: list[float], status: str) -> None:
    from models.free_boundary import extract_put_zero_crossing

    result = extract_put_zero_crossing([0.5, 0.7], levels)

    assert result.status == status
    assert result.value is None
    assert result.reason


def test_scalar_and_implicit_extraction_enforce_american_put_maturity_boundary() -> None:
    from campaign.m04 import SurfaceScenario, _extract_raw_surface
    import numpy as np

    scenario = SurfaceScenario("s", "validation", np.zeros((6, 7)), np.zeros((6, 5)), np.array([0.5, 0.8, 1.1]), np.array([0.0, 0.5]), 1.0, 1.0, 0.05, 0.0, 0.2)

    scalar = _extract_raw_surface(np.array([0.2, 0.7]), scenario, "scalar_boundary_head")
    implicit = _extract_raw_surface(np.array([[1.0, 1.0, 1.0], [-1.0, 0.0, 1.0]]), scenario, "implicit_level_set")

    assert scalar[0].value == pytest.approx(1.0)
    assert implicit[0].value == pytest.approx(1.0)
