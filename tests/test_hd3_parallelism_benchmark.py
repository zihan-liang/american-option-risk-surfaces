from __future__ import annotations

import pytest

from hd3_parallelism_benchmark import (
    parse_compute_memory_gib,
    project_canonical_seconds,
)


def test_parse_compute_memory_sums_only_requested_child_pids():
    text = "123, 1024\n456, 2048\n999, 4096\n"
    assert parse_compute_memory_gib(text, {123, 456}) == pytest.approx(3.0)
    assert parse_compute_memory_gib(text, {999}) == pytest.approx(4.0)
    assert parse_compute_memory_gib("No running processes found", {123}) == 0.0


def test_projection_interpolates_d20_and_uses_measured_parallel_speedup():
    benchmark = {
        "sequential": {
            "d10_FC_HVI": {"wall_seconds": 40.0},
            "d10_DEEPSETS_HVI": {"wall_seconds": 50.0},
            "d50_FC_HVI": {"wall_seconds": 100.0},
            "d50_DEEPSETS_HVI": {"wall_seconds": 120.0},
        },
        "concurrent_d10": {"wall_seconds": 60.0},
        "smoke_steps": 200,
    }
    sequential = project_canonical_seconds(benchmark, canonical_steps=6000, parallel_enabled=False)
    parallel = project_canonical_seconds(benchmark, canonical_steps=6000, parallel_enabled=True)
    # d20 linear interpolation: FC=55, DS=67.5 seconds for 200 steps.
    assert sequential == pytest.approx((90.0 + 122.5 + 220.0) * 30.0)
    # Apply measured d10 concurrent/sum ratio to the interpolated d20 pair.
    expected_pair20 = 122.5 * (60.0 / 90.0)
    assert parallel == pytest.approx((60.0 + expected_pair20 + 220.0) * 30.0)
    assert parallel < sequential
