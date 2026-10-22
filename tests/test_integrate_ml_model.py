"""Automated unit tests for ML model integration and prediction service.

Week 3 Day 4 — Rajarsh
"""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

from src.integrate_ml_model import (
    MLFraudPredictor,
    execute_model_integration_pipeline,
    generate_prediction_visualizations,
    REQUIRED_RAW_FEATURE_COLUMNS,
)


class TestIntegrateMLModel(unittest.TestCase):

    def setUp(self):
        """Prepare test environment with mock transactions."""
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)

        self.model_path = Path("models/logistic_regression_model.joblib")
        self.preprocessor_path = Path("data/processed/week2_day3/preprocessor.joblib")

        # Create a small valid sample DataFrame matching REQUIRED_RAW_FEATURE_COLUMNS
        sample_data = {
            "transaction_id": ["tx-001", "tx-002", "tx-003", "tx-004"],
            "amount": [150.0, 500000.0, 75.0, 250000.0],
            "old_balance_orig": [500.0, 500000.0, 200.0, 250000.0],
            "new_balance_orig": [350.0, 0.0, 125.0, 0.0],
            "old_balance_dest": [0.0, 0.0, 50.0, 1000.0],
            "new_balance_dest": [0.0, 500000.0, 125.0, 251000.0],
            "error_balance_orig": [0.0, 0.0, 0.0, 0.0],
            "error_balance_dest": [0.0, 0.0, 0.0, 0.0],
            "orig_balance_change": [-150.0, -500000.0, -75.0, -250000.0],
            "dest_balance_change": [0.0, 500000.0, 75.0, 250000.0],
            "orig_balance_error": [0.0, 0.0, 0.0, 0.0],
            "dest_balance_error": [0.0, 0.0, 0.0, 0.0],
            "amount_to_orig_balance_ratio": [0.3, 1.0, 0.375, 1.0],
            "log_amount": [5.01, 13.12, 4.31, 12.42],
            "transaction_hour": [10, 14, 9, 2],
            "transaction_day_of_week": [1, 2, 3, 4],
            "payment_type": ["PAYMENT", "TRANSFER", "CASH_IN", "CASH_OUT"],
        }
        self.sample_df = pd.DataFrame(sample_data)
        self.sample_csv = self.root / "sample_transactions.csv"
        self.sample_df.to_csv(self.sample_csv, index=False)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_predictor_initialization(self):
        """Verify predictor loads model and preprocessor successfully."""
        predictor = MLFraudPredictor(
            model_path=self.model_path,
            preprocessor_path=self.preprocessor_path,
            threshold=0.5,
        )
        self.assertIsNotNone(predictor.model)
        self.assertIsNotNone(predictor.preprocessor)
        self.assertEqual(predictor.threshold, 0.5)

    def test_missing_model_raises(self):
        """Verify FileNotFoundError is raised if model path is invalid."""
        with self.assertRaises(FileNotFoundError):
            MLFraudPredictor(model_path="models/non_existent_model.joblib")

    def test_risk_level_categorization(self):
        """Verify probability to risk tier mapping."""
        self.assertEqual(MLFraudPredictor.get_risk_level(0.95), "CRITICAL")
        self.assertEqual(MLFraudPredictor.get_risk_level(0.65), "HIGH")
        self.assertEqual(MLFraudPredictor.get_risk_level(0.35), "MEDIUM")
        self.assertEqual(MLFraudPredictor.get_risk_level(0.05), "LOW")

    def test_predict_transactions_batch(self):
        """Verify batch prediction produces correct output fields and types."""
        predictor = MLFraudPredictor(
            model_path=self.model_path,
            preprocessor_path=self.preprocessor_path,
        )
        results = predictor.predict_transactions(self.sample_df)

        expected_cols = [
            "fraud_probability",
            "predicted_label",
            "prediction_class",
            "risk_level",
            "fraud_alert",
            "confidence_score_pct",
        ]
        for col in expected_cols:
            self.assertIn(col, results.columns)

        self.assertEqual(len(results), len(self.sample_df))
        self.assertTrue(results["fraud_probability"].between(0.0, 1.0).all())
        self.assertTrue(set(results["predicted_label"]).issubset({0, 1}))
        self.assertTrue(set(results["prediction_class"]).issubset({"LEGITIMATE", "FRAUD"}))

    def test_predict_single_transaction(self):
        """Verify real-time scoring of a single event dictionary."""
        predictor = MLFraudPredictor(
            model_path=self.model_path,
            preprocessor_path=self.preprocessor_path,
        )
        single_record = self.sample_df.iloc[0].to_dict()
        scored = predictor.predict_single_transaction(single_record)

        self.assertIn("fraud_probability", scored)
        self.assertIn("predicted_label", scored)
        self.assertIn("risk_level", scored)
        self.assertIn("fraud_alert", scored)

    def test_threshold_sensitivity(self):
        """Verify threshold affects predictions."""
        pred_strict = MLFraudPredictor(self.model_path, self.preprocessor_path, threshold=0.99)
        pred_lenient = MLFraudPredictor(self.model_path, self.preprocessor_path, threshold=0.01)

        res_strict = pred_strict.predict_transactions(self.sample_df)
        res_lenient = pred_lenient.predict_transactions(self.sample_df)

        self.assertLessEqual(res_strict["predicted_label"].sum(), res_lenient["predicted_label"].sum())

    def test_execute_pipeline_and_artifacts(self):
        """Verify full execution pipeline outputs CSV, JSON alerts, summary, and figures."""
        out_dir = self.root / "outputs"
        fig_dir = self.root / "figures"

        summary = execute_model_integration_pipeline(
            input_path=self.sample_csv,
            model_path=self.model_path,
            preprocessor_path=self.preprocessor_path,
            output_dir=out_dir,
            figures_dir=fig_dir,
        )

        self.assertEqual(summary["execution_metadata"]["total_transactions_scored"], 4)
        self.assertTrue((out_dir / "prediction_results.csv").is_file())
        self.assertTrue((out_dir / "fraud_alerts.json").is_file())
        self.assertTrue((out_dir / "prediction_summary.json").is_file())

        self.assertTrue((fig_dir / "prediction_probability_distribution.png").is_file())
        self.assertTrue((fig_dir / "predictions_by_payment_type.png").is_file())
        self.assertTrue((fig_dir / "risk_level_breakdown.png").is_file())


if __name__ == "__main__":
    unittest.main()
