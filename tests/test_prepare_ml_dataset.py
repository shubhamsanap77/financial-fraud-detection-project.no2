"""Automated tests for Week 2 Day 3 ML dataset preparation."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

try:
    from prepare_ml_dataset import NUMERIC_CANDIDATES, prepare_dataset
except ModuleNotFoundError:
    from src.prepare_ml_dataset import NUMERIC_CANDIDATES, prepare_dataset
from prepare_ml_dataset import NUMERIC_CANDIDATES, prepare_dataset


class PrepareDatasetTests(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input_file = self.root / "sample.csv"

        payment_types = ["PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER", "DEBIT"]

        df = pd.DataFrame({
            "transaction_id": [f"txn-{i}" for i in range(100)],
            "is_fraud": [1] * 10 + [0] * 90,
            "payment_type": [payment_types[i % 5] for i in range(100)],
        })

        for i, column in enumerate(NUMERIC_CANDIDATES):
            if column == "transaction_month":
                df[column] = 10
            else:
                df[column] = [float(j + i + 1) for j in range(100)]

        df.to_csv(self.input_file, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_stratified_split_and_outputs(self):
        out = self.root / "output"
        summary = prepare_dataset(self.input_file, out)

        self.assertEqual(summary["train_rows"], 80)
        self.assertEqual(summary["test_rows"], 20)
        self.assertEqual(summary["train_class_distribution"], {"0": 72, "1": 8})
        self.assertEqual(summary["test_class_distribution"], {"0": 18, "1": 2})
        self.assertIn("transaction_month", summary["constant_features_removed"])

        expected_files = [
            "X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv",
            "preprocessor.joblib", "split_summary.json",
        ]
        for name in expected_files:
            self.assertTrue((out / name).is_file(), f"Missing output: {name}")

        X_train = pd.read_csv(out / "X_train.csv")
        X_test = pd.read_csv(out / "X_test.csv")
        y_train = pd.read_csv(out / "y_train.csv")
        y_test = pd.read_csv(out / "y_test.csv")

        self.assertEqual(X_train.shape, (80, 20))
        self.assertEqual(X_test.shape, (20, 20))
        self.assertEqual(y_train.shape, (80, 1))
        self.assertEqual(y_test.shape, (20, 1))
        self.assertFalse(X_train.isna().any().any())
        self.assertFalse(X_test.isna().any().any())
        self.assertNotIn("is_fraud", X_train.columns)

    def test_split_is_reproducible(self):
        out1 = self.root / "output1"
        out2 = self.root / "output2"

        prepare_dataset(self.input_file, out1)
        prepare_dataset(self.input_file, out2)

        for name in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
            pd.testing.assert_frame_equal(
                pd.read_csv(out1 / name),
                pd.read_csv(out2 / name),
            )

    def test_invalid_target_is_rejected(self):
        df = pd.read_csv(self.input_file)
        df.loc[0, "is_fraud"] = 2
        bad_file = self.root / "invalid.csv"
        df.to_csv(bad_file, index=False)

        with self.assertRaisesRegex(ValueError, "only 0 and 1"):
            prepare_dataset(bad_file, self.root / "invalid_output")


if __name__ == "__main__":
    unittest.main()
