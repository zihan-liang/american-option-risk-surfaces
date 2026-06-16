import unittest
from pathlib import Path

import numpy as np

from models.training import (
    capped_indices,
    load_dataset_arrays,
    save_torch_model,
    set_global_seed,
    split_mask,
)


class BaselineTrainingUtilityTests(unittest.TestCase):
    def test_capped_indices_are_deterministic_and_split_filtered(self):
        splits = np.array([b"train"] * 10 + [b"test"] * 5)
        mask = split_mask(splits, "train")

        first = capped_indices(mask, cap=4, seed=123)
        second = capped_indices(mask, cap=4, seed=123)

        self.assertEqual(first.tolist(), second.tolist())
        self.assertEqual(len(first), 4)
        self.assertTrue(np.all(first < 10))

    def test_load_dataset_arrays_reads_required_keys(self):
        root = Path("results/test_stage07_training_utils")
        root.mkdir(parents=True, exist_ok=True)
        path = root / "tiny.npz"
        np.savez_compressed(
            path,
            features=np.zeros((3, 2), dtype=np.float32),
            labels=np.zeros((3, 5), dtype=np.float32),
            split=np.array([b"train", b"validation", b"test"]),
            regime_key=np.array([b"a", b"b", b"c"]),
            near_boundary=np.array([False, True, False]),
        )

        data = load_dataset_arrays(path)

        self.assertEqual(data["features"].shape, (3, 2))
        self.assertIn("near_boundary", data)

    def test_save_torch_model_writes_checkpoint(self):
        import torch.nn as nn

        root = Path("results/test_stage07_training_utils")
        root.mkdir(parents=True, exist_ok=True)
        path = root / "model.pt"
        set_global_seed(5)
        model = nn.Linear(2, 1)
        save_torch_model(model, path, {"model": "tiny"})

        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
