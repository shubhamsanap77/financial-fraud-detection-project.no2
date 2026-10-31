"""Tests for transaction validation before fraud prediction."""

import unittest

import numpy as np
import pandas as pd

from validate_prediction_data import (
    MODEL_INPUT_COLUMNS,
    validate_transactions,
)


class ValidatePredictionDataTests(unittest.TestCase):
    def setUp(self):
        self.payment_types = {
            "PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"
        }
        self.sample = pd.DataFrame({
            "transaction_id": ["tx-001", "tx-002"],
            "amount": [100.0, 250.0],
            "old_balance_orig": [200.0, 400.0],
            "new_balance_orig": [100.0, 150.0],
            "old_balance_dest": [0.0, 10.0],
            "new_balance_dest": [100.0, 260.0],
            "error_balance_orig": [0.0, 0.0],
            "error_balance_dest": [0.0, 0.0],
            "orig_balance_change": [100.0, 250.0],
            "dest_balance_change": [100.0, 250.0],
            "orig_balance_error": [0.0, 0.0],
            "dest_balance_error": [0.0, 0.0],
            "amount_to_orig_balance_ratio": [0.5, 0.625],
            "log_amount": [np.log1p(100), np.log1p(250)],
            "transaction_hour": [12, 14],
            "transaction_day_of_week": [2, 3],
            "payment_type": ["PAYMENT", "TRANSFER"],
        })

    def test_valid_transactions_are_accepted(self):
        accepted, rejected, audit = validate_transactions(
            self.sample, self.payment_types
        )
        self.assertEqual(len(accepted), 2)
        self.assertTrue(rejected.empty)
        self.assertEqual(audit["rows_rejected"], 0)
        self.assertEqual(accepted.columns.tolist(), MODEL_INPUT_COLUMNS)

    def test_missing_balance_can_be_imputed(self):
        sample = self.sample.copy()
        sample.loc[0, "old_balance_orig"] = np.nan

        accepted, rejected, audit = validate_transactions(
            sample, self.payment_types
        )
        self.assertEqual(len(accepted), 2)
        self.assertTrue(rejected.empty)
        self.assertTrue(pd.isna(accepted.loc[0, "old_balance_orig"]))
        self.assertEqual(
            audit["accepted_missing_numeric_values_for_imputation"][
                "old_balance_orig"
            ],
            1,
        )

    def test_non_positive_amount_is_rejected(self):
        sample = self.sample.copy()
        sample.loc[0, "amount"] = -5.0

        accepted, rejected, _ = validate_transactions(
            sample, self.payment_types
        )
        self.assertEqual(len(accepted), 1)
        self.assertIn("non_positive_amount", rejected.iloc[0]["validation_errors"])

    def test_negative_balance_is_rejected(self):
        sample = self.sample.copy()
        sample.loc[0, "old_balance_orig"] = -1.0

        accepted, rejected, _ = validate_transactions(
            sample, self.payment_types
        )
        self.assertEqual(len(accepted), 1)
        self.assertIn("negative_old_balance_orig", rejected.iloc[0]["validation_errors"])

    def test_malformed_numeric_value_is_rejected(self):
        sample = self.sample.copy()
        sample["orig_balance_error"] = sample["orig_balance_error"].astype(object)
        sample.loc[0, "orig_balance_error"] = "not-a-number"

        accepted, rejected, _ = validate_transactions(
            sample, self.payment_types
        )
        self.assertEqual(len(accepted), 1)
        self.assertIn(
            "invalid_numeric_orig_balance_error",
            rejected.iloc[0]["validation_errors"],
        )

    def test_unsupported_payment_type_is_rejected(self):
        sample = self.sample.copy()
        sample.loc[0, "payment_type"] = "CRYPTO"

        accepted, rejected, _ = validate_transactions(
            sample, self.payment_types
        )
        self.assertEqual(len(accepted), 1)
        self.assertIn(
            "unsupported_payment_type",
            rejected.iloc[0]["validation_errors"],
        )

    def test_duplicate_transaction_ids_are_rejected(self):
        sample = self.sample.copy()
        sample["transaction_id"] = ["duplicate-id", "duplicate-id"]

        accepted, rejected, audit = validate_transactions(
            sample, self.payment_types
        )
        self.assertTrue(accepted.empty)
        self.assertEqual(len(rejected), 2)
        self.assertEqual(
            audit["rejection_reason_counts"]["duplicate_transaction_id"], 2
        )

    def test_invalid_transaction_hour_is_rejected(self):
        sample = self.sample.copy()
        sample.loc[0, "transaction_hour"] = 25

        accepted, rejected, _ = validate_transactions(
            sample, self.payment_types
        )
        self.assertEqual(len(accepted), 1)
        self.assertIn(
            "transaction_hour_out_of_range",
            rejected.iloc[0]["validation_errors"],
        )

    def test_required_column_is_enforced(self):
        sample = self.sample.drop(columns=["amount"])

        with self.assertRaisesRegex(ValueError, "Missing required prediction columns"):
            validate_transactions(sample, self.payment_types)


if __name__ == "__main__":
    unittest.main()
