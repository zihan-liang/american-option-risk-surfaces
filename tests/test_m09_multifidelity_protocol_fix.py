import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STANDALONE = REPO_ROOT / "experiments" / "standalone"
sys.path.insert(0, str(STANDALONE))


def test_seeded_builder_repeats_identical_initialization():
    torch = pytest.importorskip("torch")
    from m09_multifidelity_protocol import build_seeded_mlp
    import torch.nn as nn

    def build(input_dim, output_dim, hidden, depth):
        return nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, output_dim))

    a = build_seeded_mlp(build, seed=203, input_dim=7, output_dim=1, hidden=8, depth=2)
    b = build_seeded_mlp(build, seed=203, input_dim=7, output_dim=1, hidden=8, depth=2)
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))


def test_selected_test_keys_contains_only_validation_frozen_fractions():
    from m09_multifidelity_protocol import selected_test_keys

    selected = {
        101: {"fine_only": 0.25, "coarse_plus_few_fine": 0.10},
        203: {"fine_only": 0.50, "coarse_plus_few_fine": 0.25},
    }
    keys = selected_test_keys(selected)
    assert keys == [
        (101, "fine_only", 0.25),
        (101, "coarse_plus_few_fine", 0.10),
        (203, "fine_only", 0.50),
        (203, "coarse_plus_few_fine", 0.25),
    ]


def test_selected_test_keys_rejects_incomplete_matching():
    from m09_multifidelity_protocol import selected_test_keys

    with pytest.raises(ValueError, match="matched-accuracy selection incomplete"):
        selected_test_keys({101: {"fine_only": 0.25, "coarse_plus_few_fine": None}})
