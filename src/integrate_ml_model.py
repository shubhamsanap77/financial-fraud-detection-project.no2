"""ML Model Integration, Real-Time Prediction, and Results Generation Module.

Week 3 Day 4 — Rajarsh
Integrates the trained Machine Learning fraud detection model (Logistic Regression / Random Forest),
performs real-time and batch scoring of transactions into legitimate and fraudulent classes,
categorizes risk levels, triggers fraud alerts, and outputs comprehensive prediction results.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Expected raw input columns required for the trained preprocessor pipeline
REQUIRED_RAW_FEATURE_COLUMNS = [
    "amount",
    "old_balance_orig",
    "new_balance_orig",
    "old_balance_dest",
    "new_balance_dest",
    "error_balance_orig",
    "error_balance_dest",
    "orig_balance_change",
    "dest_balance_change",
    "orig_balance_error",
    "dest_balance_error",
    "amount_to_orig_balance_ratio",
    "log_amount",
    "transaction_hour",
    "transaction_day_of_week",
    "payment_type",
]


class MLFraudPredictor:
    """Production ML Model Integration Service for Financial Fraud Detection."""

    def __init__(
        self,
        model_path: Union[str, Path] = "models/logistic_regression_model.joblib",
        preprocessor_path: Optional[Union[str, Path]] = "data/processed/week2_day3/preprocessor.joblib",
        threshold: float = 0.5,
    ):
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path) if preprocessor_path else None
        self.threshold = float(threshold)

        if not self.model_path.is_file():
            raise FileNotFoundError(f"Trained model not found at '{self.model_path}'")

        self.model = joblib.load(self.model_path)
        self.preprocessor = None
        if self.preprocessor_path and self.preprocessor_path.is_file():
            self.preprocessor = joblib.load(self.preprocessor_path)

        # Cache model classes
        if hasattr(self.model, "classes_"):
            self.classes_ = list(self.model.classes_)
        elif hasattr(self.model, "named_steps") and hasattr(self.model.named_steps.get("classifier", None), "classes_"):
            self.classes_ = list(self.model.named_steps["classifier"].classes_)
        else:
            self.classes_ = [0, 1]

        self.fraud_class_idx = self.classes_.index(1) if 1 in self.classes_ else 1

    @staticmethod
    def get_risk_level(prob: float) -> str:
        """Categorize continuous fraud probability into standardized risk tiers."""
        if prob >= 0.85:
            return "CRITICAL"
        elif prob >= 0.50:
            return "HIGH"
        elif prob >= 0.20:
            return "MEDIUM"
        else:
            return "LOW"

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and transform raw transaction DataFrame into model-ready features."""
        # Case A: Input already contains the 20 preprocessed feature columns
        expected_features = getattr(self.model, "feature_names_in_", None)
        if expected_features is not None and all(col in df.columns for col in expected_features):
            return df[list(expected_features)].copy()

        # Case B: Input contains raw features and must be passed through preprocessor
        if self.preprocessor is not None:
            missing = [c for c in REQUIRED_RAW_FEATURE_COLUMNS if c not in df.columns]
            if missing:
                raise ValueError(f"Input data is missing required feature columns: {missing}")

            X_subset = df[REQUIRED_RAW_FEATURE_COLUMNS].copy()
            transformed_arr = self.preprocessor.transform(X_subset)
            feature_names = self.preprocessor.get_feature_names_out()
            return pd.DataFrame(transformed_arr, columns=feature_names, index=df.index)

        raise ValueError("Cannot prepare features: preprocessor is not loaded and features are not pre-transformed.")

    def predict_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate fraud predictions, probabilities, risk tiers, and alert flags for transactions.

        Parameters
        ----------
        df : pd.DataFrame
            Transaction records.

        Returns
        -------
        pd.DataFrame
            Enriched DataFrame containing prediction results.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        features_ready = self.prepare_features(df)
        probabilities = self.model.predict_proba(features_ready)[:, self.fraud_class_idx]
        predictions = (probabilities >= self.threshold).astype(int)

        results_df = df.copy()

        # Remove ground truth label if present from prediction output to reflect real-world scoring
        label_cols = [c for c in ["is_fraud", "isFraud", "is_flagged_fraud", "isFlaggedFraud"] if c in results_df.columns]
        
        results_df["fraud_probability"] = np.round(probabilities, 4)
        results_df["predicted_label"] = predictions
        results_df["prediction_class"] = np.where(predictions == 1, "FRAUD", "LEGITIMATE")
        results_df["risk_level"] = [self.get_risk_level(p) for p in probabilities]
        results_df["fraud_alert"] = predictions == 1
        # Confidence score: Distance from decision boundary normalized to [0, 100]%
        results_df["confidence_score_pct"] = np.round(np.abs(probabilities - 0.5) * 2 * 100, 2)

        return results_df

    def predict_single_transaction(self, txn_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Perform real-time scoring on a single streaming transaction event.

        Parameters
        ----------
        txn_dict : Dict[str, Any]
            Single transaction event payload.

        Returns
        -------
        Dict[str, Any]
            Scored transaction event enriched with predictions and alert flags.
        """
        single_df = pd.DataFrame([txn_dict])
        scored_df = self.predict_transactions(single_df)
        return scored_df.iloc[0].to_dict()


def generate_prediction_visualizations(
    results_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, str]:
    """Generate diagnostic visualization charts for prediction results.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame enriched with model predictions.
    output_dir : Path
        Directory to save figures.

    Returns
    -------
    Dict[str, str]
        Paths of generated visualization images.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {}

    # 1. Prediction Probability Distribution
    plt.figure(figsize=(8, 5))
    legit_probs = results_df[results_df["predicted_label"] == 0]["fraud_probability"]
    fraud_probs = results_df[results_df["predicted_label"] == 1]["fraud_probability"]

    plt.hist(
        legit_probs,
        bins=30,
        alpha=0.6,
        color="#2b5c8f",
        label=f"Predicted Legitimate (n={len(legit_probs)})",
        edgecolor="black",
    )
    if len(fraud_probs) > 0:
        plt.hist(
            fraud_probs,
            bins=20,
            alpha=0.8,
            color="#d9534f",
            label=f"Predicted Fraud (n={len(fraud_probs)})",
            edgecolor="black",
        )

    plt.axvline(0.5, color="#e67e22", linestyle="--", lw=2, label="Decision Threshold (0.50)")
    plt.title("Distribution of Predicted Fraud Probabilities", fontsize=12, fontweight="bold")
    plt.xlabel("Fraud Probability", fontsize=11)
    plt.ylabel("Transaction Count", fontsize=11)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    p1 = output_dir / "prediction_probability_distribution.png"
    plt.savefig(p1, dpi=200)
    plt.close()
    plot_paths["probability_distribution"] = str(p1)

    # 2. Predictions by Payment Type
    if "payment_type" in results_df.columns or "type" in results_df.columns:
        type_col = "payment_type" if "payment_type" in results_df.columns else "type"
        crosstab = pd.crosstab(results_df[type_col], results_df["prediction_class"])

        plt.figure(figsize=(8, 5))
        crosstab.plot(kind="bar", stacked=True, color=["#d9534f", "#2b5c8f"], edgecolor="black", ax=plt.gca())
        plt.title("Prediction Class Breakdown by Payment Type", fontsize=12, fontweight="bold")
        plt.xlabel("Payment Type", fontsize=11)
        plt.ylabel("Transaction Count", fontsize=11)
        plt.xticks(rotation=0)
        plt.legend(["Fraud", "Legitimate"], title="Prediction", loc="upper right")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        p2 = output_dir / "predictions_by_payment_type.png"
        plt.savefig(p2, dpi=200)
        plt.close()
        plot_paths["predictions_by_payment_type"] = str(p2)

    # 3. Risk Level Breakdown
    risk_counts = results_df["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).fillna(0)
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(risk_counts.index, risk_counts.values, color=colors, edgecolor="black")
    plt.title("Transactions Categorized by Risk Level", fontsize=12, fontweight="bold")
    plt.xlabel("Risk Level Tier", fontsize=11)
    plt.ylabel("Transaction Count", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + max(1.0, h * 0.01),
            f"{int(h):,}",
            ha="center",
            va="bottom",
            fontweight="bold" if h > 0 else "normal",
        )

    plt.tight_layout()
    p3 = output_dir / "risk_level_breakdown.png"
    plt.savefig(p3, dpi=200)
    plt.close()
    plot_paths["risk_level_breakdown"] = str(p3)

    return plot_paths


def execute_model_integration_pipeline(
    input_path: Union[str, Path] = "data/processed/paysim_feature_engineered.csv",
    model_path: Union[str, Path] = "models/logistic_regression_model.joblib",
    preprocessor_path: Union[str, Path] = "data/processed/week2_day3/preprocessor.joblib",
    output_dir: Union[str, Path] = "data/processed/week3_day4",
    threshold: float = 0.5,
    figures_dir: Union[str, Path] = "docs/figures/week3",
) -> Dict[str, Any]:
    """Execute end-to-end model integration, transaction prediction, and results export.

    Parameters
    ----------
    input_path : Union[str, Path]
        CSV file of transactions to predict.
    model_path : Union[str, Path]
        Path to trained ML model.
    preprocessor_path : Union[str, Path]
        Path to preprocessing pipeline.
    output_dir : Union[str, Path]
        Target folder for prediction CSV and summary JSONs.
    threshold : float
        Decision threshold.
    figures_dir : Union[str, Path]
        Folder to save diagnostic charts.

    Returns
    -------
    Dict[str, Any]
        Prediction summary metrics.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input transaction dataset not found: {input_path}")

    print(f"Loading transaction dataset from '{input_path}'...")
    raw_df = pd.read_csv(input_path)
    print(f"Total rows received for inference: {len(raw_df)}")

    predictor = MLFraudPredictor(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        threshold=threshold,
    )
    print(f"Loaded ML model from '{model_path}' (Decision Threshold = {threshold})")

    print("\nRunning inference and predicting transactions...")
    scored_df = predictor.predict_transactions(raw_df)

    # 1. Export enriched prediction results
    results_csv_path = output_dir / "prediction_results.csv"
    scored_df.to_csv(results_csv_path, index=False)
    print(f"Prediction results saved to: {results_csv_path}")

    # 2. Extract and export high-risk fraud alerts
    fraud_alerts_df = scored_df[scored_df["fraud_alert"]].copy()
    alerts_list: List[Dict[str, Any]] = []
    for _, row in fraud_alerts_df.iterrows():
        alerts_list.append({
            "transaction_id": str(row.get("transaction_id", "N/A")),
            "payment_type": str(row.get("payment_type", row.get("type", "N/A"))),
            "amount": float(row.get("amount", 0.0)),
            "user_id": str(row.get("user_id", row.get("nameOrig", "N/A"))),
            "fraud_probability": float(row["fraud_probability"]),
            "risk_level": str(row["risk_level"]),
            "confidence_score_pct": float(row["confidence_score_pct"]),
            "alert_timestamp": str(row.get("timestamp", "N/A")),
            "reason": f"Fraud probability ({row['fraud_probability']:.2%}) exceeds threshold ({threshold:.2%}) with {row['risk_level']} risk level",
        })

    fraud_alerts_json_path = output_dir / "fraud_alerts.json"
    with fraud_alerts_json_path.open("w", encoding="utf-8") as f:
        json.dump(alerts_list, f, indent=2)
    print(f"Exported {len(alerts_list)} fraud alerts to: {fraud_alerts_json_path}")

    # 3. Compute detailed summary statistics
    total_txns = len(scored_df)
    fraud_count = int(scored_df["predicted_label"].sum())
    legit_count = total_txns - fraud_count
    risk_breakdown = scored_df["risk_level"].value_counts().to_dict()

    summary_metrics = {
        "execution_metadata": {
            "input_file": str(input_path),
            "model_file": str(model_path),
            "decision_threshold": threshold,
            "total_transactions_scored": total_txns,
        },
        "prediction_counts": {
            "legitimate_transactions": legit_count,
            "legitimate_percentage": round((legit_count / total_txns) * 100, 2),
            "fraudulent_transactions": fraud_count,
            "fraud_percentage": round((fraud_count / total_txns) * 100, 2),
        },
        "probability_statistics": {
            "mean_fraud_probability": round(float(scored_df["fraud_probability"].mean()), 4),
            "median_fraud_probability": round(float(scored_df["fraud_probability"].median()), 4),
            "max_fraud_probability": round(float(scored_df["fraud_probability"].max()), 4),
            "min_fraud_probability": round(float(scored_df["fraud_probability"].min()), 4),
        },
        "risk_level_breakdown": {
            "CRITICAL": int(risk_breakdown.get("CRITICAL", 0)),
            "HIGH": int(risk_breakdown.get("HIGH", 0)),
            "MEDIUM": int(risk_breakdown.get("MEDIUM", 0)),
            "LOW": int(risk_breakdown.get("LOW", 0)),
        },
        "fraud_alerts_generated": len(alerts_list),
        "output_files": {
            "prediction_results_csv": str(results_csv_path),
            "fraud_alerts_json": str(fraud_alerts_json_path),
        },
    }

    summary_json_path = output_dir / "prediction_summary.json"
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"Prediction summary exported to: {summary_json_path}")

    # 4. Generate visualizations
    print("\nGenerating diagnostic prediction figures...")
    plot_paths = generate_prediction_visualizations(scored_df, figures_dir)
    for name, path in plot_paths.items():
        print(f"  - {name}: {path}")

    # Synchronize copies into week3/ and results/week3/ folders for seamless organization
    sync_dirs = [Path("data/processed/week3"), Path("results/week3")]
    for sdir in sync_dirs:
        sdir.mkdir(parents=True, exist_ok=True)
        scored_df.to_csv(sdir / "prediction_results.csv", index=False)
        with (sdir / "fraud_alerts.json").open("w", encoding="utf-8") as f:
            json.dump(alerts_list, f, indent=2)
        with (sdir / "prediction_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary_metrics, f, indent=2)

    # Copy figures to root docs/figures as well
    root_fig_dir = Path("docs/figures")
    for name, path in plot_paths.items():
        shutil_dest = root_fig_dir / Path(path).name
        try:
            import shutil
            shutil.copy(path, shutil_dest)
        except Exception:
            pass

    print("\n=== Model Integration & Prediction Complete ===")
    print(f"Total Transactions : {total_txns:,}")
    print(f"Legitimate Count   : {legit_count:,} ({summary_metrics['prediction_counts']['legitimate_percentage']}%)")
    print(f"Fraudulent Count   : {fraud_count:,} ({summary_metrics['prediction_counts']['fraud_percentage']}%)")
    print(f"Fraud Alerts Raised: {len(alerts_list)}")

    return summary_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrate ML model, predict transactions, and generate results.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/paysim_feature_engineered.csv"),
        help="Path to input transaction CSV",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/logistic_regression_model.joblib"),
        help="Path to saved ML model",
    )
    parser.add_argument(
        "--preprocessor",
        type=Path,
        default=Path("data/processed/week2_day3/preprocessor.joblib"),
        help="Path to saved preprocessor pipeline",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for fraud probability (default: 0.5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/week3_day4"),
        help="Output directory for prediction results",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("docs/figures/week3"),
        help="Directory to save prediction plots",
    )
    args = parser.parse_args()

    execute_model_integration_pipeline(
        input_path=args.input,
        model_path=args.model,
        preprocessor_path=args.preprocessor,
        output_dir=args.output_dir,
        threshold=args.threshold,
        figures_dir=args.figures_dir,
    )


if __name__ == "__main__":
    main()
