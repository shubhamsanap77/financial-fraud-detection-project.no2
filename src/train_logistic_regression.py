"""Logistic Regression Model Training and Evaluation Module.

Trains a regularized Logistic Regression classifier with class-imbalance weighting
on the prepared PaySim transaction dataset, evaluates performance using precision,
recall, F1-score, and confusion matrix, and saves all model artifacts and metrics.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_dataset(data_dir: Path) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load train and test feature matrices and target vectors.

    Parameters
    ----------
    data_dir : Path
        Directory containing X_train.csv, y_train.csv, X_test.csv, y_test.csv.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
        X_train, y_train, X_test, y_test.
    """
    data_dir = Path(data_dir)
    x_train_path = data_dir / "X_train.csv"
    y_train_path = data_dir / "y_train.csv"
    x_test_path = data_dir / "X_test.csv"
    y_test_path = data_dir / "y_test.csv"

    for path in [x_train_path, y_train_path, x_test_path, y_test_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Required dataset file not found: {path}")

    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).squeeze("columns").astype(int)
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).squeeze("columns").astype(int)

    return X_train, y_train, X_test, y_test


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    c_param: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
    solver: str = "lbfgs",
    class_weight: str | dict = "balanced",
) -> Pipeline:
    """Train a regularized Logistic Regression model wrapped in a scaling pipeline.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.
    c_param : float
        Inverse of regularization strength.
    max_iter : int
        Maximum number of optimization iterations.
    random_state : int
        Random seed for reproducibility.
    solver : str
        Algorithm to use in the optimization problem.
    class_weight : str | dict
        Weights associated with classes (default 'balanced' for fraud imbalance).

    Returns
    -------
    Pipeline
        Fitted Scikit-learn Pipeline with StandardScaler and LogisticRegression.
    """
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_param,
                    max_iter=max_iter,
                    random_state=random_state,
                    solver=solver,
                    class_weight=class_weight,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Evaluate classifier performance using precision, recall, F1, confusion matrix, and AUC.

    Parameters
    ----------
    model : Pipeline
        Trained Scikit-learn Pipeline.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        True test labels.

    Returns
    -------
    Dict[str, Any]
        Structured evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Confusion matrix extraction: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel().tolist()

    # Per-class and aggregate metrics
    metrics = {
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "matrix_array": cm.tolist(),
        },
        "fraud_class_metrics": {
            "precision": float(round(precision_score(y_test, y_pred, pos_label=1, zero_division=0), 4)),
            "recall": float(round(recall_score(y_test, y_pred, pos_label=1, zero_division=0), 4)),
            "f1_score": float(round(f1_score(y_test, y_pred, pos_label=1, zero_division=0), 4)),
            "support": int(np.sum(y_test == 1)),
        },
        "non_fraud_class_metrics": {
            "precision": float(round(precision_score(y_test, y_pred, pos_label=0, zero_division=0), 4)),
            "recall": float(round(recall_score(y_test, y_pred, pos_label=0, zero_division=0), 4)),
            "f1_score": float(round(f1_score(y_test, y_pred, pos_label=0, zero_division=0), 4)),
            "support": int(np.sum(y_test == 0)),
        },
        "aggregate_metrics": {
            "accuracy": float(round(accuracy_score(y_test, y_pred), 4)),
            "macro_precision": float(round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4)),
            "macro_recall": float(round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4)),
            "macro_f1": float(round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)),
            "weighted_f1": float(round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4)),
            "roc_auc": float(round(roc_auc_score(y_test, y_prob), 4)),
            "average_precision_pr_auc": float(round(average_precision_score(y_test, y_prob), 4)),
        },
        "rates": {
            "false_positive_rate": float(round(fp / (fp + tn), 4)) if (fp + tn) > 0 else 0.0,
            "false_negative_rate": float(round(fn / (fn + tp), 4)) if (fn + tp) > 0 else 0.0,
            "specificity": float(round(tn / (tn + fp), 4)) if (tn + fp) > 0 else 0.0,
        },
        "classification_report_text": classification_report(
            y_test, y_pred, target_names=["Non-Fraud (0)", "Fraud (1)"], digits=4
        ),
    }

    return metrics


def generate_evaluation_plots(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> Dict[str, str]:
    """Generate and save evaluation plots for the trained model.

    Generates:
    - confusion_matrix.png
    - roc_curve.png
    - precision_recall_curve.png
    - logistic_regression_coefficients.png

    Parameters
    ----------
    model : Pipeline
        Fitted pipeline.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test labels.
    output_dir : Path
        Directory to save plot images.

    Returns
    -------
    Dict[str, str]
        Mapping of plot names to saved file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = {}

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 1. Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Logistic Regression — Confusion Matrix", fontsize=12, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Non-Fraud (0)", "Fraud (1)"], fontsize=10)
    plt.yticks(tick_marks, ["Non-Fraud (0)", "Fraud (1)"], fontsize=10)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                f"{cm[i, j]:,}",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=13,
                fontweight="bold",
            )

    plt.ylabel("True Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    plots["confusion_matrix"] = str(cm_path)

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#2b5c8f", lw=2, label=f"Logistic Regression (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="#999999", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.4)
    plt.tight_layout()
    roc_path = output_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=200)
    plt.close()
    plots["roc_curve"] = str(roc_path)

    # 3. Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    ap_score = average_precision_score(y_test, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(recall_vals, precision_vals, color="#d9534f", lw=2, label=f"PR Curve (AP = {ap_score:.4f})")
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall Curve (Fraud Detection)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.4)
    plt.tight_layout()
    pr_path = output_dir / "precision_recall_curve.png"
    plt.savefig(pr_path, dpi=200)
    plt.close()
    plots["precision_recall_curve"] = str(pr_path)

    # 4. Feature Coefficients (Feature Importance)
    classifier = model.named_steps["classifier"]
    feature_names = list(X_test.columns)
    coefficients = classifier.coef_[0]

    # Sort by absolute impact
    indices = np.argsort(np.abs(coefficients))[-10:]  # top 10 features
    top_features = [feature_names[i].replace("numeric__", "").replace("categorical__", "") for i in indices]
    top_coeffs = [coefficients[i] for i in indices]

    plt.figure(figsize=(8, 5))
    colors = ["#2b5c8f" if c > 0 else "#d9534f" for c in top_coeffs]
    plt.barh(top_features, top_coeffs, color=colors, edgecolor="black")
    plt.title("Logistic Regression — Top 10 Feature Coefficients", fontsize=12, fontweight="bold")
    plt.xlabel("Coefficient Weight (Standardized Impact)", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    feat_path = output_dir / "logistic_regression_coefficients.png"
    plt.savefig(feat_path, dpi=200)
    plt.close()
    plots["feature_coefficients"] = str(feat_path)

    return plots


def save_model_artifacts(
    model: Pipeline,
    metrics: Dict[str, Any],
    model_output_path: Path,
    results_json_path: Path,
    report_text_path: Path,
) -> None:
    """Serialize the trained model and save performance metrics.

    Parameters
    ----------
    model : Pipeline
        Fitted Scikit-learn Pipeline.
    metrics : Dict[str, Any]
        Evaluation metrics dictionary.
    model_output_path : Path
        Path to save .joblib model.
    results_json_path : Path
        Path to save JSON metrics.
    report_text_path : Path
        Path to save plain-text classification report.
    """
    model_output_path = Path(model_output_path)
    results_json_path = Path(results_json_path)
    report_text_path = Path(report_text_path)

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    results_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_text_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model pipeline
    joblib.dump(model, model_output_path)

    # Save metrics JSON (excluding long plain text report from JSON for clean serialization)
    json_metrics = {k: v for k, v in metrics.items() if k != "classification_report_text"}
    with results_json_path.open("w", encoding="utf-8") as f:
        json.dump(json_metrics, f, indent=2)

    # Save textual classification report
    with report_text_path.open("w", encoding="utf-8") as f:
        f.write("=== Logistic Regression Performance Report ===\n\n")
        f.write(metrics["classification_report_text"])
        f.write("\n\nConfusion Matrix:\n")
        cm = metrics["confusion_matrix"]
        f.write(f"  True Negatives (TN) : {cm['true_negatives']}\n")
        f.write(f"  False Positives (FP): {cm['false_positives']}\n")
        f.write(f"  False Negatives (FN): {cm['false_negatives']}\n")
        f.write(f"  True Positives (TP) : {cm['true_positives']}\n")
        f.write(f"\nROC AUC Score: {metrics['aggregate_metrics']['roc_auc']:.4f}\n")
        f.write(f"PR AUC Score : {metrics['aggregate_metrics']['average_precision_pr_auc']:.4f}\n")


def run_training_pipeline(
    data_dir: Path = Path("data/processed/week2_day3"),
    model_output_dir: Path = Path("models"),
    figures_dir: Path = Path("docs/figures"),
    results_dir: Path = Path("results"),
    c_param: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute complete end-to-end model training, evaluation, and serialization."""
    data_dir = Path(data_dir)
    model_output_dir = Path(model_output_dir)
    figures_dir = Path(figures_dir)
    results_dir = Path(results_dir)

    print(f"Loading prepared ML dataset from '{data_dir}'...")
    X_train, y_train, X_test, y_test = load_dataset(data_dir)
    print(f"Train set: {X_train.shape[0]} rows, {X_train.shape[1]} features (Fraud: {int(y_train.sum())})")
    print(f"Test set : {X_test.shape[0]} rows, {X_test.shape[1]} features (Fraud: {int(y_test.sum())})")

    print("\nTraining Logistic Regression model with balanced class weights...")
    model = train_logistic_regression(
        X_train=X_train,
        y_train=y_train,
        c_param=c_param,
        max_iter=max_iter,
        random_state=random_state,
        class_weight="balanced",
    )
    print("Model training completed successfully.")

    print("\nEvaluating model performance on test set...")
    metrics = evaluate_model(model, X_test, y_test)

    # Print summary to console
    print("\n--- Test Set Evaluation Results ---")
    print(metrics["classification_report_text"])
    cm = metrics["confusion_matrix"]
    print(f"Confusion Matrix: TN={cm['true_negatives']}, FP={cm['false_positives']}, FN={cm['false_negatives']}, TP={cm['true_positives']}")
    fraud_metrics = metrics["fraud_class_metrics"]
    print(f"Fraud Class (1) -> Precision: {fraud_metrics['precision']:.4f}, Recall: {fraud_metrics['recall']:.4f}, F1-Score: {fraud_metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['aggregate_metrics']['roc_auc']:.4f}, PR-AUC: {metrics['aggregate_metrics']['average_precision_pr_auc']:.4f}")

    print("\nGenerating evaluation figures...")
    plot_paths = generate_evaluation_plots(model, X_test, y_test, figures_dir)
    for name, path in plot_paths.items():
        print(f"  - {name}: {path}")

    print("\nSaving model artifacts and results...")
    model_path = model_output_dir / "logistic_regression_model.joblib"
    results_json = results_dir / "evaluation_results.json"
    report_text = results_dir / "classification_report.txt"

    save_model_artifacts(
        model=model,
        metrics=metrics,
        model_output_path=model_path,
        results_json_path=results_json,
        report_text_path=report_text,
    )

    # Also save copy in models/ for redundancy
    joblib.dump(metrics, model_output_dir / "evaluation_results.joblib")
    with (model_output_dir / "evaluation_results.json").open("w", encoding="utf-8") as f:
        json.dump({k: v for k, v in metrics.items() if k != "classification_report_text"}, f, indent=2)

    print(f"Trained model saved to : {model_path}")
    print(f"Evaluation metrics saved: {results_json}")
    print(f"Classification report   : {report_text}")

    return {
        "model_path": str(model_path),
        "metrics": metrics,
        "plots": plot_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Logistic Regression for Fraud Detection.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/week2_day3"),
        help="Directory containing X_train.csv, y_train.csv, etc.",
    )
    parser.add_argument(
        "--model-output-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save the trained model.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("docs/figures"),
        help="Directory to save evaluation plots.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to save evaluation metrics JSON and text reports.",
    )
    parser.add_argument(
        "--c-param",
        type=float,
        default=1.0,
        help="Regularization parameter C (default: 1.0)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum solver iterations (default: 1000)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    run_training_pipeline(
        data_dir=args.data_dir,
        model_output_dir=args.model_output_dir,
        figures_dir=args.figures_dir,
        results_dir=args.results_dir,
        c_param=args.c_param,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
