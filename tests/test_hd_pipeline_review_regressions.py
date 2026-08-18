from __future__ import annotations

import json
from pathlib import Path

from hd3_classical_vs_structured_high_dimensional_pinn import paired_batch_seed

ROOT = Path(__file__).resolve().parents[1]


def test_hd3_fc_and_deepsets_share_identical_minibatch_seed():
    cfg = json.loads(
        (ROOT / "configs" / "hd3_classical_vs_structured_high_dimensional_pinn.json").read_text()
    )
    for d in (10, 20, 50, 100):
        expected = int(cfg["seed"]) + 1000 * d
        assert paired_batch_seed(d, cfg, "FC_HVI") == expected
        assert paired_batch_seed(d, cfg, "DEEPSETS_HVI") == expected


def test_hd2_training_loop_updates_completed_step_after_each_optimizer_step():
    source = (
        ROOT / "experiments" / "hd2_hvi_objective_stabilization.py"
    ).read_text(encoding="utf-8")
    assert "completed_step = step + 1" in source
