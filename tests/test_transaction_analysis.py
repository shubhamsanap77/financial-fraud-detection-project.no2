"""Unit tests for transaction data analysis and reporting."""
import os
import unittest
import pandas as pd

from src.transaction_analysis import (
    analyze_transaction_data,
    generate_visualizations,
    export_markdown_report,
)


class TestTransactionAnalysis(unittest.TestCase):

    def setUp(self):
        """Prepare sample transaction DataFrame for analysis tests."""
        self.sample_df = pd.DataFrame({
            "step": [1, 2, 3, 4, 5, 6],
            "type": ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "TRANSFER", "DEBIT"],
            "amount": [100.0, 50000.0, 15000.0, 2000.0, 250000.0, 50.0],
            "nameOrig": ["C1", "C2", "C3", "C4", "C5", "C6"],
            "oldbalanceOrg": [200.0, 50000.0, 15000.0, 0.0, 250000.0, 100.0],
            "newbalanceOrig": [100.0, 0.0, 0.0, 2000.0, 0.0, 50.0],
            "nameDest": ["M1", "C20", "C30", "C40", "C50", "C60"],
            "oldbalanceDest": [0.0, 0.0, 500.0, 10000.0, 1000.0, 200.0],
            "newbalanceDest": [0.0, 50000.0, 15500.0, 8000.0, 251000.0, 250.0],
            "isFraud": [0, 1, 1, 0, 1, 0],
            "isFlaggedFraud": [0, 0, 0, 0, 1, 0],
        })

    def test_analyze_transaction_data_metrics(self):
        """Verify summary metrics and fraud percentages."""
        analysis = analyze_transaction_data(self.sample_df)

        summary = analysis["summary"]
        self.assertEqual(summary["total_transactions"], 6)
        self.assertEqual(summary["fraudulent_transactions"], 3)
        self.assertEqual(summary["legitimate_transactions"], 3)
        self.assertEqual(summary["fraud_percentage"], 50.0)

    def test_fraud_type_exclusivity(self):
        """Verify that fraud patterns isolate to TRANSFER and CASH_OUT."""
        analysis = analyze_transaction_data(self.sample_df)
        fraud_types = analysis["fraud_behavior_patterns"]["fraud_types_present"]

        self.assertIn("TRANSFER", fraud_types)
        self.assertIn("CASH_OUT", fraud_types)
        self.assertNotIn("PAYMENT", fraud_types)
        self.assertNotIn("CASH_IN", fraud_types)
        self.assertNotIn("DEBIT", fraud_types)

    def test_account_draining_metric(self):
        """Verify account draining percentage computation."""
        analysis = analyze_transaction_data(self.sample_df)
        drained_pct = analysis["fraud_behavior_patterns"]["drained_origin_balance_pct"]
        # In our sample data, all 3 fraud transactions drained origin to 0
        self.assertEqual(drained_pct, 100.0)

    def test_visualizations_generation(self):
        """Verify that analysis charts are rendered without errors."""
        test_fig_dir = "docs/figures/test_figs"
        fig_paths = generate_visualizations(self.sample_df, output_dir=test_fig_dir)

        self.assertTrue(os.path.exists(fig_paths["type_distribution"]))
        self.assertTrue(os.path.exists(fig_paths["fraud_by_type"]))
        self.assertTrue(os.path.exists(fig_paths["amount_distribution"]))

        # Cleanup
        for path in fig_paths.values():
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(test_fig_dir):
            os.rmdir(test_fig_dir)

    def test_export_markdown_report(self):
        """Verify markdown report creation."""
        test_report = "docs/test_report.md"
        analysis = analyze_transaction_data(self.sample_df)
        export_markdown_report(analysis, test_report)

        self.assertTrue(os.path.exists(test_report))
        with open(test_report, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("PaySim Transaction-Data Analysis Report", content)
            self.assertIn("Class Imbalance Ratio", content)

        # Cleanup
        if os.path.exists(test_report):
            os.remove(test_report)


if __name__ == "__main__":
    unittest.main()
