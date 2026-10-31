"""Train and evaluate an XGBoost fraud detection model."""

import json
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "paysim_feature_engineered.csv"
OUTPUT_DIR = ROOT / "outputs" / "week2_day6"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "is_fraud"

EXCLUDED_COLUMNS = [
    "transaction_id",
    "timestamp",
    "user_id",
    "name_dest",
    "currency",
    "is_synthetic_test_event",
    "is_flagged_fraud",
    TARGET,
]


def main():
    print("Loading feature-engineered dataset...")
    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")

    df = df.dropna(subset=[TARGET]).copy()
    y = df[TARGET].astype(int)

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError(
            "Both classes need at least two records for stratified splitting."
        )

    X = df.drop(
        columns=[c for c in EXCLUDED_COLUMNS if c in df.columns]
    ).copy()

    # Convert datetime-like and categorical columns into model-ready values.
    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            X[col] = X[col].astype("int64") / 1e9
        elif X[col].dtype == "object":
            X[col] = X[col].fillna("Missing").astype(str)

    X = pd.get_dummies(X, dummy_na=True)
    X = X.replace([float("inf"), float("-inf")], float("nan"))
    X = X.fillna(0)
    X = X.astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    fraud_count = int((y_train == 1).sum())
    non_fraud_count = int((y_train == 0).sum())
    scale_pos_weight = non_fraud_count / max(fraud_count, 1)

    print("Dataset shape:", df.shape)
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Training fraud cases:", fraud_count)
    print("Testing fraud cases:", int((y_test == 1).sum()))
    print("Number of features:", X_train.shape[1])

    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=2,
        tree_method="hist",
    )

    print("\nTraining XGBoost model...")
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "model": "XGBoost",
        "dataset_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "fraud_cases_total": int((y == 1).sum()),
        "fraud_cases_test": int((y_test == 1).sum()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "fraud_precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "fraud_recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "fraud_f1": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
        "classification_report": classification_report(
            y_test, predictions, zero_division=0, output_dict=True
        ),
    }

    model.save_model(str(OUTPUT_DIR / "xgboost_model.json"))
    joblib.dump(list(X.columns), OUTPUT_DIR / "feature_columns.joblib")

    pd.DataFrame({
        "actual_is_fraud": y_test.to_numpy(),
        "predicted_is_fraud": predictions,
        "fraud_probability": probabilities,
    }).to_csv(OUTPUT_DIR / "xgboost_predictions.csv", index=False)

    with open(OUTPUT_DIR / "xgboost_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        OUTPUT_DIR / "xgboost_feature_importance.csv", index=False
    )

    print("\nEvaluation results:")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Fraud Precision:   {metrics['fraud_precision']:.4f}")
    print(f"Fraud Recall:      {metrics['fraud_recall']:.4f}")
    print(f"Fraud F1-score:    {metrics['fraud_f1']:.4f}")
    print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])
    print("\nTop 10 important features:")
    print(importance.head(10).to_string(index=False))
    print("\nSaved model and evaluation outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()