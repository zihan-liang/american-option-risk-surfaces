from __future__ import annotations

import json
from pathlib import Path

from hd_pipeline.runtime import reported_completed_step
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


def test_interrupted_status_reports_the_latest_durable_checkpoint_step():
    # A KeyboardInterrupt may happen after the loop variable advanced but before
    # the next 250-step checkpoint. Resume can only guarantee the durable
    # checkpoint, so status must report that recovery point rather than a stale
    # in-memory loop variable.
    assert reported_completed_step(status_step=0, checkpoint_step=750) == 750
    assert reported_completed_step(status_step=900, checkpoint_step=750) == 750
    assert reported_completed_step(status_step=1000, checkpoint_step=None) == 1000
