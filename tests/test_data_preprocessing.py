"""Unit tests for PaySim data preprocessing and feature engineering."""
import os
import unittest
import pandas as pd
import numpy as np

from src.data_preprocessing import (
    validate_schema,
    clean_paysim_data,
    engineer_features,
    export_pipeline_events,
    prepare_dataset,
    REQUIRED_PAYSIM_COLUMNS,
)


class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        """Create sample raw records for testing."""
        self.sample_raw_data = pd.DataFrame({
            "step": [1, 2, 3, 0, 4],  # Row 3 has invalid step 0
            "type": ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "TRANSFER"],
            "amount": [50.0, 1000.0, 500.0, -10.0, 250000.0],  # Row 3 has invalid negative amount
            "nameOrig": ["C10001", "C10002", "C10003", "C10004", "C10005"],
            "oldbalanceOrg": [100.0, 1000.0, 600.0, 100.0, 250000.0],
            "newbalanceOrig": [50.0, 0.0, 100.0, 90.0, 0.0],
            "nameDest": ["M20001", "C30002", "C30003", "C30004", "C30005"],
            "oldbalanceDest": [0.0, 500.0, 100.0, 0.0, 0.0],
            "newbalanceDest": [0.0, 1500.0, 600.0, 0.0, 250000.0],
            "isFraud": [0, 1, 0, 0, 1],
            "isFlaggedFraud": [0, 0, 0, 0, 1],
        })

    def test_validate_schema_success(self):
        """Verify schema validation passes when all required columns exist."""
        is_valid, missing = validate_schema(self.sample_raw_data)
        self.assertTrue(is_valid)
        self.assertEqual(len(missing), 0)

    def test_validate_schema_failure(self):
        """Verify schema validation fails and reports missing columns."""
        incomplete_df = self.sample_raw_data.drop(columns=["isFraud", "step"])
        is_valid, missing = validate_schema(incomplete_df)
        self.assertFalse(is_valid)
        self.assertIn("isFraud", missing)
        self.assertIn("step", missing)

    def test_clean_paysim_data(self):
        """Verify data cleaning eliminates invalid steps, negative amounts, and computes audit."""
        cleaned_df, stats = clean_paysim_data(self.sample_raw_data)
        self.assertEqual(stats["initial_rows"], 5)
        # Row 3 (negative amount and step 0) should be dropped
        self.assertEqual(len(cleaned_df), 4)
        self.assertEqual(stats["dropped_rows"], 1)
        self.assertTrue((cleaned_df["amount"] > 0).all())
        self.assertTrue((cleaned_df["step"] >= 1).all())

    def test_engineer_features(self):
        """Verify feature engineering calculations and pipeline field alignments."""
        cleaned_df, _ = clean_paysim_data(self.sample_raw_data)
        enriched_df = engineer_features(cleaned_df)

        # Check required pipeline fields
        for col in ["transaction_id", "timestamp", "user_id", "payment_type", "currency", "is_synthetic_test_event"]:
            self.assertIn(col, enriched_df.columns)

        # Check currency and test flag defaults
        self.assertTrue((enriched_df["currency"] == "USD").all())
        self.assertFalse(enriched_df["is_synthetic_test_event"].any())

        # Check balance discrepancy feature:
        # For Row 1: TRANSFER amount 1000, old 1000, new 0 => errorBalanceOrig = (1000 - 1000) - 0 = 0.0
        # dest: old 500, amount 1000, new 1500 => errorBalanceDest = (500 + 1000) - 1500 = 0.0
        row1 = enriched_df[enriched_df["nameOrig"] == "C10002"].iloc[0]
        self.assertEqual(row1["errorBalanceOrig"], 0.0)
        self.assertEqual(row1["errorBalanceDest"], 0.0)

    def test_export_pipeline_events(self):
        """Verify JSON export creates properly structured Kafka event records."""
        cleaned_df, _ = clean_paysim_data(self.sample_raw_data)
        enriched_df = engineer_features(cleaned_df)

        test_json = "data/processed/test_stream_events.json"
        exported = export_pipeline_events(enriched_df, test_json, max_records=2)
        self.assertEqual(exported, 2)
        self.assertTrue(os.path.exists(test_json))

        # Cleanup test artifact
        if os.path.exists(test_json):
            os.remove(test_json)


if __name__ == "__main__":
    unittest.main()
