"""Train and evaluate Random Forest for financial fraud detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_dataset(data_dir: Path):
    """Load the existing prepared train/test datasets."""
    data_dir = Path(data_dir)
    names = ["X_train.csv", "y_train.csv", "X_test.csv", "y_test.csv"]

    for name in names:
        if not (data_dir / name).is_file():
            raise FileNotFoundError(f"Missing dataset file: {data_dir / name}")

    X_train = pd.read_csv(data_dir / "X_train.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").squeeze("columns").astype(int)
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze("columns").astype(int)

    if X_train.columns.tolist() != X_test.columns.tolist():
        raise ValueError("Training and test features do not match.")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature and target row counts do not match.")
    if X_train.empty or X_train.shape[1] == 0:
        raise ValueError("Training data is empty.")
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Features contain missing values.")
    if not set(y_train.unique()).issubset({0, 1}):
        raise ValueError("Training target must contain only 0 and 1.")
    if not set(y_test.unique()).issubset({0, 1}):
        raise ValueError("Test target must contain only 0 and 1.")
    if y_train.nunique() != 2:
        raise ValueError("Training target must contain both classes.")

    print(f"Training rows: {len(X_train)}, features: {X_train.shape[1]}")
    print(f"Training fraud cases: {int(y_train.sum())}")
    print(f"Test rows: {len(X_test)}, fraud cases: {int(y_test.sum())}")
    return X_train, y_train, X_test, y_test


def train_random_forest(X_train, y_train, n_estimators=300, random_state=42):
    """Train a class-balanced Random Forest."""
    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1.")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    return model.fit(X_train, y_train)


def train_logistic_regression(X_train, y_train, random_state=42):
    """Reproduce the existing balanced Logistic Regression baseline."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
            class_weight="balanced",
        )),
    ])
    return model.fit(X_train, y_train)


def evaluate_model(model, X_test, y_test, model_name):
    """Calculate classification metrics on the held-out test set."""
    predictions = model.predict(X_test)
    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X_test)[:, positive_index]

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    auc = roc_auc_score(y_test, probabilities) if y_test.nunique() == 2 else None
    ap = average_precision_score(y_test, probabilities) if y_test.nunique() == 2 else None

    return {
        "model_name": model_name,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "matrix_array": matrix.tolist(),
        },
        "fraud_class_metrics": {
            "precision": round(float(precision_score(
                y_test, predictions, pos_label=1, zero_division=0
            )), 4),
            "recall": round(float(recall_score(
                y_test, predictions, pos_label=1, zero_division=0
            )), 4),
            "f1_score": round(float(f1_score(
                y_test, predictions, pos_label=1, zero_division=0
            )), 4),
            "support": int((y_test == 1).sum()),
        },
        "aggregate_metrics": {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "macro_f1": round(float(f1_score(
                y_test, predictions, average="macro", zero_division=0
            )), 4),
            "weighted_f1": round(float(f1_score(
                y_test, predictions, average="weighted", zero_division=0
            )), 4),
            "roc_auc": round(float(auc), 4) if auc is not None else None,
            "average_precision_pr_auc": round(float(ap), 4) if ap is not None else None,
        },
        "rates": {
            "false_positive_rate": round(float(fp / (fp + tn)), 4) if fp + tn else 0.0,
            "false_negative_rate": round(float(fn / (fn + tp)), 4) if fn + tp else 0.0,
            "specificity": round(float(tn / (tn + fp)), 4) if tn + fp else 0.0,
        },
        "classification_report_text": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Non-Fraud (0)", "Fraud (1)"],
            digits=4,
            zero_division=0,
        ),
    }


def save_metrics(metrics, json_path, report_path):
    """Save structured metrics and a readable classification report."""
    json_path = Path(json_path)
    report_path = Path(report_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        key: value for key, value in metrics.items()
        if key != "classification_report_text"
    }
    json_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    cm = metrics["confusion_matrix"]
    report = (
        f"=== {metrics['model_name']} Performance Report ===\n\n"
        f"{metrics['classification_report_text']}\n"
        f"True Negatives: {cm['true_negatives']}\n"
        f"False Positives: {cm['false_positives']}\n"
        f"False Negatives: {cm['false_negatives']}\n"
        f"True Positives: {cm['true_positives']}\n"
    )
    report_path.write_text(report, encoding="utf-8")


def get_feature_importance(model):
    """Rank features by Random Forest impurity-based importance."""
    importance = pd.DataFrame({
        "feature": model.feature_names_in_,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False, ignore_index=True)

    importance.insert(0, "rank", np.arange(1, len(importance) + 1))
    return importance


def generate_plots(rf_model, lr_model, X_test, y_test, importance, figures_dir):
    """Save model evaluation and feature-importance plots."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Random Forest confusion matrix
    matrix = confusion_matrix(y_test, rf_model.predict(X_test), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Non-Fraud", "Fraud"])
    ax.set_yticks([0, 1], labels=["Non-Fraud", "Fraud"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Random Forest — Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(figures_dir / "random_forest_confusion_matrix.png", dpi=160)
    plt.close(fig)

    # Top 10 feature importances
    top = importance.head(10).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Impurity-based importance")
    ax.set_title("Random Forest — Top 10 Feature Importances")
    fig.tight_layout()
    fig.savefig(figures_dir / "random_forest_feature_importance.png", dpi=160)
    plt.close(fig)

    if y_test.nunique() == 2:
        # ROC curve comparison
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, model in [
            ("Random Forest", rf_model),
            ("Logistic Regression", lr_model),
        ]:
            probabilities = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
            fpr, tpr, _ = roc_curve(y_test, probabilities)
            auc = roc_auc_score(y_test, probabilities)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

        ax.plot([0, 1], [0, 1], "--", label="Random chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve Comparison")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_dir / "model_comparison_roc_curve.png", dpi=160)
        plt.close(fig)

        # Precision-recall curve comparison
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, model in [
            ("Random Forest", rf_model),
            ("Logistic Regression", lr_model),
        ]:
            probabilities = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
            precision, recall, _ = precision_recall_curve(y_test, probabilities)
            ap = average_precision_score(y_test, probabilities)
            ax.plot(recall, precision, label=f"{name} (AP={ap:.4f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve Comparison")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_dir / "model_comparison_precision_recall.png", dpi=160)
        plt.close(fig)


def run_training_pipeline(
    data_dir=Path("data/processed/week2_day3"),
    model_output_dir=Path("models"),
    figures_dir=Path("docs/figures"),
    results_dir=Path("results"),
    n_estimators=300,
    random_state=42,
):
    """Train both models, evaluate them, and save all results."""
    X_train, y_train, X_test, y_test = load_dataset(data_dir)

    print("\nTraining Random Forest...")
    rf_model = train_random_forest(
        X_train, y_train, n_estimators=n_estimators,
        random_state=random_state,
    )

    print("Training Logistic Regression...")
    lr_model = train_logistic_regression(
        X_train, y_train, random_state=random_state,
    )

    print("Evaluating both models on the same test set...")
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    lr_metrics = evaluate_model(
        lr_model, X_test, y_test, "Logistic Regression"
    )

    model_output_dir = Path(model_output_dir)
    results_dir = Path(results_dir)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_output_dir / "random_forest_model.joblib"
    joblib.dump(rf_model, model_path)

    importance = get_feature_importance(rf_model)
    importance.to_csv(results_dir / "feature_importance.csv", index=False)

    save_metrics(
        rf_metrics,
        results_dir / "random_forest_evaluation.json",
        results_dir / "random_forest_classification_report.txt",
    )
    save_metrics(
        lr_metrics,
        results_dir / "logistic_regression_comparison.json",
        results_dir / "logistic_regression_comparison_report.txt",
    )

    def comparison_row(name, metrics):
        fraud = metrics["fraud_class_metrics"]
        aggregate = metrics["aggregate_metrics"]
        cm = metrics["confusion_matrix"]
        return {
            "model": name,
            "accuracy": aggregate["accuracy"],
            "fraud_precision": fraud["precision"],
            "fraud_recall": fraud["recall"],
            "fraud_f1": fraud["f1_score"],
            "macro_f1": aggregate["macro_f1"],
            "roc_auc": aggregate["roc_auc"],
            "average_precision_pr_auc": aggregate["average_precision_pr_auc"],
            "true_negatives": cm["true_negatives"],
            "false_positives": cm["false_positives"],
            "false_negatives": cm["false_negatives"],
            "true_positives": cm["true_positives"],
        }

    comparison = pd.DataFrame([
        comparison_row("Logistic Regression", lr_metrics),
        comparison_row("Random Forest", rf_metrics),
    ])
    comparison.to_csv(results_dir / "model_comparison.csv", index=False)

    combined = {
        "logistic_regression": {
            key: value for key, value in lr_metrics.items()
            if key != "classification_report_text"
        },
        "random_forest": {
            key: value for key, value in rf_metrics.items()
            if key != "classification_report_text"
        },
    }
    (results_dir / "model_comparison.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )

    generate_plots(
        rf_model, lr_model, X_test, y_test, importance, figures_dir
    )

    print("\n=== Model comparison ===")
    print(comparison.to_string(index=False))
    print("\n=== Top 10 feature importances ===")
    print(importance.head(10).to_string(index=False))
    print(f"\nSaved model: {model_path}")
    print(f"Saved results: {results_dir}")
    print(f"Saved figures: {figures_dir}")
    print(
        "\nWARNING: Only three fraud cases are in the test set. "
        "Treat evaluation results as preliminary, not production validation."
    )

    return {
        "random_forest": rf_metrics,
        "logistic_regression": lr_metrics,
        "comparison": comparison,
        "feature_importance": importance,
        "model_path": str(model_path),
    }


def main():
    """Parse command-line arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train and compare Random Forest and Logistic Regression."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/week2_day3"),
    )
    parser.add_argument(
        "--model-output-dir",
        type=Path,
        default=Path("models"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("docs/figures"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    run_training_pipeline(
        data_dir=args.data_dir,
        model_output_dir=args.model_output_dir,
        figures_dir=args.figures_dir,
        results_dir=args.results_dir,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
