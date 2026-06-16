import unittest

import numpy as np

from data.dataset_builder import build_surface_samples, solver_config_hash
from solvers.cn_psor import SolverConfig


class DatasetBuilderSchemaTests(unittest.TestCase):
    def test_surface_samples_include_schema_masks_and_quality_metadata(self):
        cfg = SolverConfig(option_type="put", k=1.0, t=0.25, r=0.05, q=0.0, sigma=0.2, m=12, n=10, tol=1e-8)

        sample = build_surface_samples(
            cfg,
            regime_key_value="put_clean",
            split_name="train",
            diagnostic_status="clean",
            source_stage="stage05",
        )

        node_count = (cfg.m + 1) * (cfg.n + 1)
        self.assertEqual(sample["features"].shape, (node_count, 7))
        self.assertEqual(sample["labels"].shape, (node_count, 5))
        self.assertEqual(sample["boundary"].shape, (node_count,))
        self.assertEqual(sample["exercise_indicator"].shape, (node_count,))
        self.assertEqual(sample["strict_greek_mask"].shape, (node_count,))
        self.assertEqual(sample["near_kink"].shape, (node_count,))
        self.assertEqual(sample["near_boundary"].shape, (node_count,))
        self.assertEqual(sample["near_zero_spot"].shape, (node_count,))
        self.assertEqual(sample["regime_key"].shape, (node_count,))
        self.assertEqual(sample["split"].shape, (node_count,))
        self.assertEqual(sample["diagnostic_status"].shape, (node_count,))
        self.assertEqual(sample["source_stage"].shape, (node_count,))
        self.assertEqual(sample["solver_config_hash"].shape, (node_count,))

        self.assertTrue(np.all(np.isfinite(sample["features"])))
        self.assertTrue(np.all(np.isfinite(sample["labels"])))
        self.assertTrue(np.all(sample["labels"][:, 1] >= -1e-10))
        self.assertTrue(set(np.unique(sample["exercise_indicator"])).issubset({False, True}))
        self.assertTrue(set(np.unique(sample["strict_greek_mask"])).issubset({False, True}))
        self.assertEqual(sample["regime_key"][0].decode("utf-8"), "put_clean")
        self.assertEqual(sample["split"][0].decode("utf-8"), "train")
        self.assertEqual(sample["diagnostic_status"][0].decode("utf-8"), "clean")
        self.assertEqual(sample["source_stage"][0].decode("utf-8"), "stage05")
        self.assertEqual(sample["solver_config_hash"][0].decode("utf-8"), solver_config_hash(cfg))


if __name__ == "__main__":
    unittest.main()
