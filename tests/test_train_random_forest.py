"""Unit tests for Week 2 Day 5 fraud model training."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from train_random_forest import (
    evaluate_model,
    get_feature_importance,
    load_dataset,
    train_logistic_regression,
    train_random_forest,
)


class RandomForestTests(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        rng = np.random.default_rng(42)

        X_train = pd.DataFrame({
            "signal": np.r_[
                rng.normal(0, 1, 30),
                rng.normal(3, 1, 10),
            ],
            "noise": rng.normal(0, 1, 40),
        })
        y_train = pd.Series([0] * 30 + [1] * 10, name="is_fraud")

        X_test = pd.DataFrame({
            "signal": np.r_[
                rng.normal(0, 1, 9),
                rng.normal(3, 1, 3),
            ],
            "noise": rng.normal(0, 1, 12),
        })
        y_test = pd.Series([0] * 9 + [1] * 3, name="is_fraud")

        X_train.to_csv(self.data_dir / "X_train.csv", index=False)
        y_train.to_csv(self.data_dir / "y_train.csv", index=False)
        X_test.to_csv(self.data_dir / "X_test.csv", index=False)
        y_test.to_csv(self.data_dir / "y_test.csv", index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_dataset(self):
        X_train, y_train, X_test, y_test = load_dataset(self.data_dir)

        self.assertEqual(X_train.shape, (40, 2))
        self.assertEqual(X_test.shape, (12, 2))
        self.assertEqual(int(y_train.sum()), 10)
        self.assertEqual(int(y_test.sum()), 3)

    def test_random_forest_training_and_evaluation(self):
        X_train, y_train, X_test, y_test = load_dataset(self.data_dir)
        model = train_random_forest(
            X_train, y_train, n_estimators=20, random_state=42
        )
        metrics = evaluate_model(model, X_test, y_test, "Random Forest")

        self.assertEqual(metrics["fraud_class_metrics"]["support"], 3)
        self.assertIn("accuracy", metrics["aggregate_metrics"])
        self.assertIn("roc_auc", metrics["aggregate_metrics"])
        self.assertEqual(
            sum(metrics["confusion_matrix"]["matrix_array"][0])
            + sum(metrics["confusion_matrix"]["matrix_array"][1]),
            12,
        )

    def test_logistic_regression_training(self):
        X_train, y_train, X_test, y_test = load_dataset(self.data_dir)
        model = train_logistic_regression(X_train, y_train)
        predictions = model.predict(X_test)

        self.assertEqual(len(predictions), len(y_test))
        self.assertTrue(set(predictions).issubset({0, 1}))

    def test_feature_importance_is_ranked(self):
        X_train, y_train, _, _ = load_dataset(self.data_dir)
        model = train_random_forest(
            X_train, y_train, n_estimators=20, random_state=42
        )
        importance = get_feature_importance(model)

        self.assertEqual(len(importance), 2)
        self.assertEqual(importance["rank"].tolist(), [1, 2])
        self.assertTrue(
            importance["importance"].is_monotonic_decreasing
        )
        self.assertAlmostEqual(importance["importance"].sum(), 1.0)

    def test_invalid_estimator_count_is_rejected(self):
        X_train, y_train, _, _ = load_dataset(self.data_dir)

        with self.assertRaisesRegex(ValueError, "at least 1"):
            train_random_forest(X_train, y_train, n_estimators=0)

    def test_missing_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as empty_dir:
            with self.assertRaisesRegex(FileNotFoundError, "Missing dataset"):
                load_dataset(Path(empty_dir))


if __name__ == "__main__":
    unittest.main()
