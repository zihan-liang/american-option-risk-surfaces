import unittest

from data.splits import three_way_regime_split


class DatasetSplitTests(unittest.TestCase):
    def test_three_way_split_has_no_regime_leakage(self):
        regimes = [{"case_id": f"regime_{idx}"} for idx in range(20)]

        splits = three_way_regime_split(regimes, train_fraction=0.7, validation_fraction=0.15, seed=11)

        train = {row["case_id"] for row in splits["train"]}
        validation = {row["case_id"] for row in splits["validation"]}
        test = {row["case_id"] for row in splits["test"]}

        self.assertEqual(len(train | validation | test), 20)
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)
        self.assertEqual(len(train), 14)
        self.assertEqual(len(validation), 3)
        self.assertEqual(len(test), 3)


if __name__ == "__main__":
    unittest.main()
