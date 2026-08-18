from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from hd_pipeline.objective_optimizers import (
    ReLoBRaLoState,
    cagrad_combine,
)


def test_cagrad_identical_tasks_returns_the_same_gradient():
    g = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    combined, meta = cagrad_combine(g, g, alpha=0.5, rescale=1)
    assert torch.allclose(combined, g, atol=1e-10, rtol=1e-10)
    assert meta["solver_success"] is True
    assert 0.0 <= meta["task1_weight"] <= 1.0


def test_cagrad_alpha_zero_reduces_to_average_gradient():
    g1 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    g2 = torch.tensor([-0.25, 2.0], dtype=torch.float64)
    combined, _ = cagrad_combine(g1, g2, alpha=0.0, rescale=1)
    assert torch.allclose(combined, 0.5 * (g1 + g2), atol=1e-10, rtol=1e-10)


def test_cagrad_conflicting_tasks_returns_finite_direction():
    g1 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    g2 = torch.tensor([-0.8, 0.1], dtype=torch.float64)
    combined, meta = cagrad_combine(g1, g2, alpha=0.5, rescale=1)
    assert torch.isfinite(combined).all()
    assert math.isfinite(meta["lambda"])
    assert meta["solver_success"] is True


def test_relobralo_reference_update_preserves_two_task_weight_sum():
    state = ReLoBRaLoState(task_count=2, temperature=1.0, alpha=0.999, rho=1.0)
    w0 = state.update([1.0, 1.0])
    assert np.allclose(w0, [1.0, 1.0])
    w1 = state.update([2.0, 1.0])
    assert sum(w1) == pytest.approx(2.0, abs=1e-12)
    assert w1[0] > w1[1]
    w2 = state.update([1.5, 1.2])
    assert sum(w2) == pytest.approx(2.0, abs=1e-10)
    assert all(math.isfinite(value) and value > 0.0 for value in w2)


def test_relobralo_state_roundtrip_is_resume_safe():
    state = ReLoBRaLoState(task_count=2, temperature=1.0, alpha=0.999, rho=1.0)
    for losses in ([1.0, 2.0], [0.9, 1.5], [0.8, 1.2]):
        state.update(losses)
    payload = state.state_dict()
    restored = ReLoBRaLoState.from_state_dict(payload)
    expected = state.update([0.7, 1.0])
    actual = restored.update([0.7, 1.0])
    assert np.allclose(actual, expected, rtol=0, atol=1e-12)
