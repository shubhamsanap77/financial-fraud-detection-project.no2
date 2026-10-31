"""Compare Logistic Regression, Random Forest, and XGBoost fairly."""

import json
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "paysim_feature_engineered.csv"
OUTPUT_DIR = ROOT / "outputs" / "week2_day6"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "is_fraud"
EXCLUDED = [
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
    df = pd.read_csv(DATA_PATH).dropna(subset=[TARGET]).copy()
    y = df[TARGET].astype(int)
    X = df.drop(columns=[c for c in EXCLUDED if c in df.columns])

    # Keep preprocessing identical for all three models.
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ])

    X_train_ready = preprocessor.fit_transform(X_train)
    X_test_ready = preprocessor.transform(X_test)

    if hasattr(X_train_ready, "toarray"):
        X_train_ready = X_train_ready.toarray()
        X_test_ready = X_test_ready.toarray()

    # XGBoost uses the same split and encoded feature matrix as other models.
    positive = int((y_train == 1).sum())
    negative = int((y_train == 0).sum())
    pos_weight = negative / max(positive, 1)

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=2,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=pos_weight,
            random_state=42,
            n_jobs=2,
            tree_method="hist",
        ),
    }

    results = []
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_ready, y_train)
        probabilities = model.predict_proba(X_test_ready)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "fraud_precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "fraud_recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "fraud_f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "average_precision": average_precision_score(
                y_test, probabilities
            ),
            "confusion_matrix": confusion_matrix(
                y_test, predictions
            ).tolist(),
        }
        results.append(row)
        print(row)

    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values(
        ["fraud_f1", "fraud_recall", "average_precision"],
        ascending=False,
    )

    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    with open(
        OUTPUT_DIR / "model_comparison.json", "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=4)

    print("\n=== Model Comparison ===")
    print(comparison.drop(columns=["confusion_matrix"]).to_string(index=False))
    print("\nSaved comparison to:", OUTPUT_DIR)
    print(
        "\nCAUTION: Only 13 fraud cases exist in the full dataset. "
        "The test set has very few fraud cases, so scores are preliminary."
    )


if __name__ == "__main__":
    main()