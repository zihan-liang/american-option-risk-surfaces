import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


_EXPERIMENT_PATH = Path("experiments/27_m02_obstacle_aware_pinn.py")
_SPEC = importlib.util.spec_from_file_location("m02_obstacle_aware_pinn", _EXPERIMENT_PATH)
experiment = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = experiment
assert _SPEC.loader is not None
_SPEC.loader.exec_module(experiment)


class M02ObstaclePinnOutputTests(unittest.TestCase):
    def test_real_smoke_config_admits_one_complete_surface_per_development_split(self):
        from common import load_config

        cfg = load_config("configs/independent_m02_obstacle_pinn.yaml")
        data = np.load(cfg["dataset"])
        split = data["split"].astype(str)
        regimes = data["regime_key"].astype(str)
        selections = experiment._predeclared_split_selections(
            split,
            np.ones(len(split), dtype=bool),
            regimes,
            {**cfg, **cfg["smoke_settings"]},
        )

        self.assertEqual(cfg["smoke_settings"]["epochs"], 1)
        self.assertEqual(cfg["smoke_settings"]["hidden"], 8)
        self.assertEqual(cfg["smoke_settings"]["depth"], 1)
        for indices, regime_ids in selections.values():
            self.assertEqual(len(regime_ids), 1)
            self.assertEqual(len(indices), 14_641)

    def test_smoke_run_writes_labeled_outputs_with_all_four_methods(self):
        root = Path("results/test_m02_obstacle_pinn")
        root.mkdir(parents=True, exist_ok=True)
        dataset = root / "tiny_dataset.npz"
        features, labels, split, regime_key = _tiny_dataset()
        np.savez_compressed(
            dataset,
            features=features,
            labels=labels,
            split=split,
            regime_key=regime_key,
            near_boundary=np.zeros(len(features), dtype=bool),
            near_kink=np.zeros(len(features), dtype=bool),
            strict_greek_mask=np.ones(len(features), dtype=bool),
            boundary=np.full(len(features), 0.9, dtype=np.float64),
        )

        output_root = root / "out"
        summary = experiment.run(
            "configs/independent_m02_obstacle_pinn.yaml",
            output_root=output_root,
            overrides={
                "dataset": str(dataset),
                "mode": "smoke",
                "seeds": [101],
                "epochs": 2,
                "patience": 2,
                "batch_size": 8,
                "train_cap": 24,
                "validation_cap": 12,
                "legacy_test_cap": 12,
                "selection_seed": 991,
                "collocation_cap": 8,
                "residual_eval_cap": 8,
                "hidden": 8,
                "depth": 1,
                "device": "cpu",
            },
        )

        self.assertEqual(summary["mode"], "smoke")
        self.assertFalse(summary["evidence_run"])
        self.assertEqual(summary["canonical_run_launched"], False)
        self.assertEqual(
            set(summary["methods"]),
            {"supervised", "masked_pde", "separate_complementarity", "fischer_burmeister"},
        )
        self.assertEqual(summary["split_names"], ["train", "validation", "legacy_test"])
        for name in [
            "config_manifest.json",
            "run_metrics.csv",
            "training_history.csv",
            "decision.json",
            "artifact_hashes.json",
            "m02_config_manifest.json",
            "m02_metrics_by_regime.csv",
            "m02_training_history.csv",
            "m02_promotion.json",
            "m02_summary.json",
        ]:
            path = output_root / name
            self.assertTrue(path.exists(), name)
            if path.suffix == ".csv":
                self.assertNotIn(b"\r\n", path.read_bytes(), name)
        with open(output_root / "m02_metrics_by_regime.csv", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual({row["method"] for row in rows}, set(summary["methods"]))
        self.assertIn("legacy_test", {row["split"] for row in rows})
        self.assertIn("vi_collocation_samples", rows[0])
        self.assertIn("masked_pde_residual_mean_abs", rows[0])
        self.assertIn("complementarity_product_mean_abs", rows[0])
        self.assertIn("dual_feasibility_violation_mean", rows[0])
        self.assertNotIn("continuation_violation_mean", rows[0])
        with open(output_root / "m02_summary.json", encoding="utf-8") as fh:
            loaded = json.load(fh)
        self.assertEqual(loaded["mode"], "smoke")
        self.assertEqual(loaded["selection"]["selection_seed"], 991)
        self.assertEqual(loaded["selection"]["train"]["samples"], 24)
        self.assertEqual(loaded["promotion"]["comparisons"]["masked_pde"]["status"], "diagnostic_only")
        decision = json.loads((output_root / "decision.json").read_text(encoding="utf-8"))
        self.assertEqual(decision["status"], "diagnostic_only")
        self.assertEqual(decision["budget_seconds"], 3600)
        self.assertIn("elapsed_screen_seconds", decision)
        self.assertEqual(decision["blind_holdout_accessed"], False)
        config_manifest = json.loads((output_root / "config_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(config_manifest["config"]["blind_holdout_accessed"], False)
        hashes = json.loads((output_root / "artifact_hashes.json").read_text(encoding="utf-8"))
        self.assertIn("m02_metrics_by_regime.csv", hashes["files"])
        for name, expected_hash in hashes["files"].items():
            self.assertEqual(hashlib.sha256((output_root / name).read_bytes()).hexdigest(), expected_hash)
        self.assertEqual(loaded["blind_holdout_accessed"], False)
        timing = loaded["costs"]["methods"]["supervised"]["per_seed"][0]["inference"]
        self.assertEqual(timing["warmups"], 0)
        self.assertEqual(timing["repetitions"], 1)
        self.assertIn("preprocessing_seconds", loaded["costs"])
        self.assertEqual(loaded["costs"]["reused_psor_label_generation_seconds"], None)
        self.assertEqual(loaded["costs"]["fallback_seconds"], 0.0)
        self.assertIn("fallback_reason", loaded["costs"])
        self.assertIn("training_seconds", loaded["costs"]["methods"]["supervised"]["per_seed"][0])
        self.assertIn("residual_diagnostic_seconds", loaded["costs"]["methods"]["supervised"]["per_seed"][0])

    def test_holm_correction_downgrades_an_otherwise_promoted_primary_comparison(self):
        comparisons = {
            method: {
                "status": "promote",
                "raw_status": "promote",
                "claim_status": "screen_candidate",
                "metrics": {"price_rmse": {"status": "defined", "paired_p_value": p_value}},
            }
            for method, p_value in zip(
                ("masked_pde", "separate_complementarity", "fischer_burmeister"),
                (0.02, 0.03, 0.04),
            )
        }

        experiment._apply_price_holm(comparisons, {"mode": "canonical"})

        self.assertFalse(comparisons["masked_pde"]["primary_price_holm"]["reject"])
        self.assertEqual(comparisons["masked_pde"]["status"], "inconclusive_holm")

    def test_missing_primary_family_member_downgrades_every_raw_promotion(self):
        comparisons = {
            "masked_pde": _promoted_price_comparison(0.01),
            "separate_complementarity": _promoted_price_comparison(0.02),
            "fischer_burmeister": {
                "status": "insufficient_diagnostics",
                "raw_status": "insufficient_diagnostics",
                "claim_status": "diagnostic_only",
                "metrics": {"price_rmse": {"status": "unavailable"}},
            },
        }

        experiment._apply_price_holm(comparisons, {"mode": "canonical"})

        for method in ("masked_pde", "separate_complementarity"):
            self.assertEqual(comparisons[method]["raw_status"], "inconclusive_holm_unavailable")
            self.assertEqual(comparisons[method]["status"], "inconclusive_holm_unavailable")
            self.assertEqual(comparisons[method]["claim_status"], "diagnostic_only")

    def test_predeclared_selections_and_residual_indices_ignore_training_seed(self):
        _, _, split, regime_keys = _tiny_dataset()
        features, _, _, _ = _tiny_dataset()
        selections = experiment._predeclared_split_selections(
            split,
            np.ones(len(split), dtype=bool),
            regime_keys,
            {
                "selection_seed": 13,
                "train_cap": 24,
                "validation_cap": 12,
                "legacy_test_cap": 12,
            },
        )
        np.random.seed(101)
        first_selection = experiment._predeclared_split_selections(
            split,
            np.ones(len(split), dtype=bool),
            regime_keys,
            {
                "selection_seed": 13,
                "train_cap": 24,
                "validation_cap": 12,
                "legacy_test_cap": 12,
            },
        )
        np.random.seed(203)
        second_selection = experiment._predeclared_split_selections(
            split,
            np.ones(len(split), dtype=bool),
            regime_keys,
            {
                "selection_seed": 13,
                "train_cap": 24,
                "validation_cap": 12,
                "legacy_test_cap": 12,
            },
        )
        first = experiment._predeclared_residual_indices(
            selections["validation"][0],
            regime_keys,
            np.ones(len(split), dtype=bool),
            cap=2,
            selection_seed=13,
            split_name="validation",
        )
        second = experiment._predeclared_residual_indices(
            selections["validation"][0],
            regime_keys,
            np.ones(len(split), dtype=bool),
            cap=2,
            selection_seed=13,
            split_name="validation",
        )

        self.assertEqual(first_selection["train"][0].tolist(), second_selection["train"][0].tolist())
        self.assertEqual(first_selection["validation"][1], second_selection["validation"][1])
        self.assertEqual({key: value.tolist() for key, value in first.items()}, {key: value.tolist() for key, value in second.items()})
        self.assertTrue(all(len(indices) == 2 for indices in first.values()))
        self.assertEqual(features.shape[0], len(split))

    def test_promotion_blocks_missing_boundary_diagnostics(self):
        baseline, candidate = _promotion_rows()

        promotion = experiment._paired_promotion(baseline, candidate, _promotion_cfg())

        self.assertEqual(promotion["status"], "insufficient_diagnostics")
        self.assertEqual(promotion["metrics"]["boundary_rmse"]["status"], "unavailable")

    def test_shared_cli_dispatches_m02_smoke(self):
        root = Path("results/test_m02_obstacle_pinn")
        dataset = root / "cli_dataset.npz"
        features, labels, split, regime_key = _tiny_dataset()
        np.savez_compressed(
            dataset,
            features=features,
            labels=labels,
            split=split,
            regime_key=regime_key,
            near_boundary=np.zeros(len(features), dtype=bool),
            near_kink=np.zeros(len(features), dtype=bool),
            boundary=np.full(len(features), 0.9, dtype=np.float64),
        )
        from common import load_config

        config = load_config("configs/independent_m02_obstacle_pinn.yaml")
        config["dataset"] = str(dataset)
        config["smoke_settings"].update({"epochs": 1, "batch_size": 8, "collocation_cap": 8, "train_cap": 24, "validation_cap": 12, "legacy_test_cap": 12, "residual_eval_cap": 2})
        config_path = root / "cli_config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        output_root = root / "cli_out"

        result = subprocess.run(
            [
                "/Users/liangzihan/Documents/SURF_0159/SURF/.worktrees/american-option-risk-surfaces/.venv-campaign/bin/python",
                "experiments/26_independent_method_campaign.py",
                "--method", "M02", "--phase", "screen", "--smoke", "--config", str(config_path), "--output-root", str(output_root),
            ],
            text=True,
            capture_output=True,
            check=True,
            env={**__import__("os").environ, "PYTHONPATH": "src:experiments", "MPLBACKEND": "Agg"},
        )

        self.assertEqual(json.loads(result.stdout)["mode"], "smoke")
        self.assertTrue((output_root / "M02" / "decision.json").exists())


def _tiny_dataset():
    rows_per_regime = 4
    regimes = [
        ("train_a", "train"),
        ("train_b", "train"),
        ("train_c", "train"),
        ("train_d", "train"),
        ("train_e", "train"),
        ("train_f", "train"),
        ("validation_a", "validation"),
        ("validation_b", "validation"),
        ("validation_c", "validation"),
        ("legacy_a", "test"),
        ("legacy_b", "test"),
        ("legacy_c", "test"),
    ]
    n = len(regimes) * rows_per_regime
    features = np.zeros((n, 7), dtype=np.float64)
    split = []
    regime_key = []
    for regime_index, (name, split_name) in enumerate(regimes):
        start = regime_index * rows_per_regime
        end = start + rows_per_regime
        features[start:end, 0] = np.linspace(-0.2, 0.2, rows_per_regime)
        features[start:end, 1] = np.linspace(0.2, 0.8, rows_per_regime)
        features[start:end, 2] = 0.05
        features[start:end, 3] = 0.01
        features[start:end, 4] = 0.25
        features[start:end, 5] = 1.0
        split.extend([split_name.encode("utf-8")] * rows_per_regime)
        regime_key.extend([name.encode("utf-8")] * rows_per_regime)
    payoff = np.maximum(1.0 - np.exp(features[:, 0]), 0.0)
    premium = 0.03 + 0.02 * features[:, 1]
    price = payoff + premium
    labels = np.column_stack([price, premium, payoff, np.zeros(n), np.zeros(n)])
    return features, labels, np.asarray(split), np.asarray(regime_key)


def _promotion_cfg():
    return {"bootstrap_iterations": 20, "bootstrap_seed": 3, "obstacle_tolerance": 1.0e-10, "mode": "canonical"}


def _promoted_price_comparison(p_value):
    return {
        "status": "promote",
        "raw_status": "promote",
        "claim_status": "screen_candidate",
        "metrics": {"price_rmse": {"status": "defined", "paired_p_value": p_value}},
    }


def _promotion_rows():
    rows = []
    for method in ("supervised", "masked_pde"):
        for regime_index, regime_id in enumerate(("a", "b", "c")):
            rows.append(
                {
                    "method": method,
                    "regime_id": regime_id,
                    "price_rmse": 1.0,
                    "delta_rmse": 1.0,
                    "k_gamma_rmse": 1.0,
                    "boundary_rmse": None,
                    "dual_feasibility_violation_mean": 0.1,
                    "complementarity_product_mean_abs": 0.1,
                    "masked_pde_residual_mean_abs": 0.1,
                    "max_obstacle_violation": 0.0,
                }
            )
    return [row for row in rows if row["method"] == "supervised"], [row for row in rows if row["method"] == "masked_pde"]


if __name__ == "__main__":
    unittest.main()
