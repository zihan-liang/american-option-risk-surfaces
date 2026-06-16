import csv
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

from common import ensure_dir, write_csv

_EXPERIMENT_PATH = Path("experiments/05_surrogate_dataset.py")
_SPEC = importlib.util.spec_from_file_location("stage06_dataset_outputs", _EXPERIMENT_PATH)
experiment = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = experiment
assert _SPEC.loader is not None
_SPEC.loader.exec_module(experiment)


class SurrogateDatasetExperimentOutputTests(unittest.TestCase):
    def test_small_run_writes_dataset_outputs_and_exclusions(self):
        source_root = Path("results/test_stage06_source")
        ensure_dir(source_root)
        risk_rows = [
            {"case_id": "put_clean", "regime_class": "put", "option_type": "put", "T": 0.25, "sigma": 0.2, "r": 0.05, "q": 0.0},
            {"case_id": "call_clean", "regime_class": "dividend_call", "option_type": "call", "T": 0.25, "sigma": 0.2, "r": 0.05, "q": 0.06},
            {"case_id": "control", "regime_class": "no_dividend_call_control", "option_type": "call", "T": 0.25, "sigma": 0.2, "r": 0.05, "q": 0.0},
            {"case_id": "put_bad", "regime_class": "put", "option_type": "put", "T": 2.0, "sigma": 0.6, "r": 0.05, "q": 0.1},
        ]
        flag_rows = [
            {"case_id": "put_clean", "diagnostic_failure": False, "delicate_regime": False},
            {"case_id": "call_clean", "diagnostic_failure": False, "delicate_regime": False},
            {"case_id": "control", "diagnostic_failure": False, "delicate_regime": False},
            {"case_id": "put_bad", "diagnostic_failure": True, "delicate_regime": True},
        ]
        write_csv(source_root / "risk_metrics.csv", risk_rows)
        write_csv(source_root / "regime_flags.csv", flag_rows)
        write_csv(source_root / "boundary_summary.csv", [{"case_id": row["case_id"]} for row in risk_rows])

        out_dir = Path("results/test_stage06_dataset")
        dataset_path = Path("data/processed/test_surrogate_dataset.npz")
        summary = experiment.run(
            "configs/surrogate_dataset.yaml",
            output_root=out_dir,
            dataset_path=dataset_path,
            figure_root=Path("results/figures/test_stage06_dataset"),
            overrides={
                "source_experiment_root": str(source_root),
                "m": 10,
                "n": 8,
                "train_fraction": 0.5,
                "validation_fraction": 0.25,
                "seed": 3,
            },
        )

        self.assertEqual(summary["clean_regimes"], 2)
        self.assertEqual(summary["excluded_regimes"], 2)
        for name in [
            "dataset_manifest.json",
            "split_table.csv",
            "excluded_regimes.csv",
            "dataset_schema.json",
            "label_summary.csv",
            "mask_summary.csv",
            "dataset_audit.json",
        ]:
            self.assertTrue((out_dir / name).exists(), name)
        self.assertTrue(dataset_path.exists())

        with np.load(dataset_path) as data:
            for key in [
                "features",
                "labels",
                "boundary",
                "exercise_indicator",
                "strict_greek_mask",
                "near_kink",
                "near_boundary",
                "near_zero_spot",
                "regime_key",
                "split",
                "diagnostic_status",
                "source_stage",
                "solver_config_hash",
            ]:
                self.assertIn(key, data.files)
            self.assertEqual(set(np.unique(data["regime_key"]).astype(str)), {"call_clean", "put_clean"})

        with open(out_dir / "excluded_regimes.csv", newline="", encoding="utf-8") as fh:
            excluded = list(csv.DictReader(fh))
        self.assertEqual({row["case_id"] for row in excluded}, {"control", "put_bad"})

        with open(out_dir / "dataset_manifest.json", encoding="utf-8") as fh:
            manifest = json.load(fh)
        self.assertEqual(manifest["source_stage"], "stage05_application_risk_surfaces")


if __name__ == "__main__":
    unittest.main()
