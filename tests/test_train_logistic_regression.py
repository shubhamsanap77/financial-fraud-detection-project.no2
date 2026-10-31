"""Unit tests for Logistic Regression training, evaluation, and serialization."""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.train_logistic_regression import (
    load_dataset,
    train_logistic_regression,
    evaluate_model,
    generate_evaluation_plots,
    save_model_artifacts,
)


class TestTrainLogisticRegression(unittest.TestCase):

    def setUp(self):
        """Create mock synthetic datasets and temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        np.random.seed(42)
        n_train = 100
        n_test = 30
        n_features = 5

        # Synthetic feature matrices
        self.X_train = pd.DataFrame(
            np.random.randn(n_train, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        # Imbalanced target (10% fraud)
        self.y_train = pd.Series(
            np.random.choice([0, 1], size=n_train, p=[0.9, 0.1]),
            name="is_fraud",
        )

        self.X_test = pd.DataFrame(
            np.random.randn(n_test, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        self.y_test = pd.Series(
            np.random.choice([0, 1], size=n_test, p=[0.9, 0.1]),
            name="is_fraud",
        )
        # Ensure at least one positive in test set for testing metrics
        self.y_test.iloc[0] = 1
        self.y_test.iloc[1] = 0

        # Save synthetic files
        self.X_train.to_csv(self.data_dir / "X_train.csv", index=False)
        self.y_train.to_csv(self.data_dir / "y_train.csv", index=False)
        self.X_test.to_csv(self.data_dir / "X_test.csv", index=False)
        self.y_test.to_csv(self.data_dir / "y_test.csv", index=False)

    def tearDown(self):
        """Clean up temporary test artifacts."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_dataset(self):
        """Verify load_dataset reads and parses all four dataset files."""
        X_tr, y_tr, X_te, y_te = load_dataset(self.data_dir)
        self.assertEqual(len(X_tr), len(self.X_train))
        self.assertEqual(len(y_tr), len(self.y_train))
        self.assertEqual(len(X_te), len(self.X_test))
        self.assertEqual(len(y_te), len(self.y_test))

    def test_load_dataset_missing_file_raises(self):
        """Verify load_dataset raises FileNotFoundError on missing files."""
        (self.data_dir / "X_test.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            load_dataset(self.data_dir)

    def test_train_logistic_regression(self):
        """Verify model training returns a fitted Pipeline and generates predictions."""
        model = train_logistic_regression(self.X_train, self.y_train, max_iter=200)
        self.assertIsInstance(model, Pipeline)
        preds = model.predict(self.X_test)
        self.assertEqual(len(preds), len(self.X_test))
        self.assertTrue(set(preds).issubset({0, 1}))

    def test_evaluate_model_metrics(self):
        """Verify evaluation returns precision, recall, F1, and confusion matrix."""
        model = train_logistic_regression(self.X_train, self.y_train, max_iter=200)
        metrics = evaluate_model(model, self.X_test, self.y_test)

        self.assertIn("confusion_matrix", metrics)
        self.assertIn("fraud_class_metrics", metrics)
        self.assertIn("aggregate_metrics", metrics)

        cm = metrics["confusion_matrix"]
        self.assertIn("true_positives", cm)
        self.assertIn("false_positives", cm)
        self.assertIn("true_negatives", cm)
        self.assertIn("false_negatives", cm)

        f_metrics = metrics["fraud_class_metrics"]
        self.assertIn("precision", f_metrics)
        self.assertIn("recall", f_metrics)
        self.assertIn("f1_score", f_metrics)

        agg = metrics["aggregate_metrics"]
        self.assertIn("accuracy", agg)
        self.assertIn("roc_auc", agg)

    def test_save_and_reload_artifacts(self):
        """Verify model pipeline and metrics serialization to disk."""
        model = train_logistic_regression(self.X_train, self.y_train, max_iter=200)
        metrics = evaluate_model(model, self.X_test, self.y_test)

        model_path = Path(self.temp_dir) / "test_model.joblib"
        json_path = Path(self.temp_dir) / "test_metrics.json"
        txt_path = Path(self.temp_dir) / "test_report.txt"

        save_model_artifacts(model, metrics, model_path, json_path, txt_path)

        self.assertTrue(model_path.is_file())
        self.assertTrue(json_path.is_file())
        self.assertTrue(txt_path.is_file())

        reloaded_model = joblib.load(model_path)
        orig_preds = model.predict(self.X_test)
        reloaded_preds = reloaded_model.predict(self.X_test)
        np.testing.assert_array_equal(orig_preds, reloaded_preds)

    def test_generate_evaluation_plots(self):
        """Verify generated plot image files."""
        model = train_logistic_regression(self.X_train, self.y_train, max_iter=200)
        fig_dir = Path(self.temp_dir) / "figures"
        plot_paths = generate_evaluation_plots(model, self.X_test, self.y_test, fig_dir)

        self.assertIn("confusion_matrix", plot_paths)
        self.assertIn("roc_curve", plot_paths)
        self.assertIn("precision_recall_curve", plot_paths)
        self.assertIn("feature_coefficients", plot_paths)

        for path in plot_paths.values():
            self.assertTrue(Path(path).is_file())


if __name__ == "__main__":
    unittest.main()
