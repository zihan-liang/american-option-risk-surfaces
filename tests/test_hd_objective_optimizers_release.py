from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from hd_pipeline.objective_optimizers import ReLoBRaLoState, cagrad_combine


def test_cagrad_identical_tasks_matches_official_stabilized_formula():
    g = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    combined, meta = cagrad_combine(g, g, alpha=0.5, rescale=1)
    # The official implementation uses lambda=c/(||gw||+1e-4), so this is
    # intentionally close to, but not algebraically identical to, 1.2*g.
    assert torch.allclose(combined, 1.2 * g, atol=5e-5, rtol=5e-5)
    assert meta["solver_success"] is True


def test_cagrad_alpha_zero_is_average_and_conflict_case_is_finite():
    g1 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    g2 = torch.tensor([-0.25, 2.0], dtype=torch.float64)
    combined, _ = cagrad_combine(g1, g2, alpha=0.0, rescale=1)
    assert torch.allclose(combined, 0.5 * (g1 + g2), atol=1e-10, rtol=1e-10)
    conflict, meta = cagrad_combine(
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        torch.tensor([-0.8, 0.1], dtype=torch.float64),
        alpha=0.5,
        rescale=1,
    )
    assert torch.isfinite(conflict).all()
    assert math.isfinite(meta["lambda"])


def test_relobralo_weights_remain_positive_normalized_and_resume_safe():
    state = ReLoBRaLoState(task_count=2, temperature=1.0, alpha=0.999, rho=1.0)
    for losses in ([1.0, 1.0], [2.0, 1.0], [1.5, 1.2]):
        weights = state.update(losses)
        assert sum(weights) == pytest.approx(2.0, abs=1e-10)
        assert all(math.isfinite(value) and value > 0.0 for value in weights)
    restored = ReLoBRaLoState.from_state_dict(state.state_dict())
    assert np.allclose(
        state.update([1.2, 1.0]),
        restored.update([1.2, 1.0]),
        atol=1e-12,
        rtol=0,
    )
