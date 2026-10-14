"""Prepare reproducible ML-ready datasets for fraud detection."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight


TARGET_COLUMN = "is_fraud"
TEST_SIZE = 0.20
RANDOM_STATE = 42

NUMERIC_CANDIDATES = [
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
    "transaction_month",
]

CATEGORICAL_CANDIDATES = ["payment_type"]

EXCLUDED_COLUMNS = [
    "transaction_id",
    "user_id",
    "name_dest",
    "timestamp",
    "currency",
    "is_synthetic_test_event",
    "is_flagged_fraud",
    TARGET_COLUMN,
]


def prepare_dataset(input_path: Path, output_dir: Path) -> dict:
    """Validate, split, preprocess, and save the dataset."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    df = pd.read_csv(input_path)

    required = {
        TARGET_COLUMN,
        *NUMERIC_CANDIDATES,
        *CATEGORICAL_CANDIDATES,
    }
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df.empty:
        raise ValueError("The input dataset contains no rows.")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("The target column contains missing values.")

    target_numeric = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    if target_numeric.isna().any():
        raise ValueError("The target column must contain only 0 and 1.")
    if not target_numeric.isin([0, 1]).all():
        raise ValueError("The target column must contain only 0 and 1.")

    y = target_numeric.astype(int)
    original_counts = {
        str(int(label)): int(count)
        for label, count in y.value_counts().sort_index().items()
    }

    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError(
            "Stratified splitting requires at least two examples of each class."
        )

    if "transaction_id" in df.columns:
        if df["transaction_id"].duplicated().any():
            raise ValueError("Duplicate transaction_id values were found.")

    X = df[NUMERIC_CANDIDATES + CATEGORICAL_CANDIDATES].copy()

    for column in NUMERIC_CANDIDATES:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    X[NUMERIC_CANDIDATES] = X[NUMERIC_CANDIDATES].replace(
        [np.inf, -np.inf], np.nan
    )

    numeric_columns = [
        column for column in NUMERIC_CANDIDATES if column in X.columns
    ]
    categorical_columns = [
        column for column in CATEGORICAL_CANDIDATES if column in X.columns
    ]

    if not numeric_columns and not categorical_columns:
        raise ValueError("No usable feature columns remain.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Detect constant features using training data only.
    constant_columns = [
        column
        for column in X_train.columns
        if X_train[column].nunique(dropna=True) <= 1
    ]
    X_train = X_train.drop(columns=constant_columns)
    X_test = X_test.drop(columns=constant_columns)

    numeric_columns = [
        column for column in NUMERIC_CANDIDATES if column in X_train.columns
    ]
    categorical_columns = [
        column for column in CATEGORICAL_CANDIDATES if column in X_train.columns
    ]

    if not numeric_columns and not categorical_columns:
        raise ValueError("No usable feature columns remain after constant-feature removal.")

    # Fit imputers and encoders on training data only.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            )),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    X_train_array = preprocessor.fit_transform(X_train)
    X_test_array = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out().tolist()

    X_train_ready = pd.DataFrame(X_train_array, columns=feature_names)
    X_test_ready = pd.DataFrame(X_test_array, columns=feature_names)

    # Calculate weights from training labels only; never rebalance the test set.
    classes = np.sort(y_train.unique())
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train.to_numpy(),
    )
    class_weights = {
        str(int(label)): float(weight)
        for label, weight in zip(classes, weights)
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    X_train_ready.to_csv(output_dir / "X_train.csv", index=False)
    X_test_ready.to_csv(output_dir / "X_test.csv", index=False)
    y_train.rename(TARGET_COLUMN).to_csv(
        output_dir / "y_train.csv", index=False
    )
    y_test.rename(TARGET_COLUMN).to_csv(
        output_dir / "y_test.csv", index=False
    )
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")

    def distribution(values: pd.Series) -> dict:
        return {
            str(int(label)): int(count)
            for label, count in values.value_counts().sort_index().items()
        }

    summary = {
        "source_file": str(input_path),
        "target_column": TARGET_COLUMN,
        "total_rows": int(len(df)),
        "original_class_distribution": original_counts,
        "excluded_columns": [
            column for column in EXCLUDED_COLUMNS if column in df.columns
        ],
        "constant_features_removed": constant_columns,
        "numeric_features_used": numeric_columns,
        "categorical_features_used": categorical_columns,
        "encoded_feature_names": feature_names,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_rows": int(len(X_train_ready)),
        "test_rows": int(len(X_test_ready)),
        "train_class_distribution": distribution(y_train),
        "test_class_distribution": distribution(y_test),
        "balanced_class_weights_from_training_data": class_weights,
        "imbalance_strategy": (
            "No resampling. Use the saved class weights when training a "
            "compatible classifier. The test set remains unchanged."
        ),
        "evaluation_warning": (
            "Only 13 fraud examples exist in the source dataset. Test metrics "
            "will be unstable and must not be treated as production evidence."
        ),
    }

    with (output_dir / "split_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print("ML dataset preparation completed.")
    print("Rows:", len(df))
    print("Original class distribution:", original_counts)
    print("Training rows:", len(X_train_ready))
    print("Test rows:", len(X_test_ready))
    print("Training class distribution:", distribution(y_train))
    print("Test class distribution:", distribution(y_test))
    print("Encoded feature count:", len(feature_names))
    print("Class weights:", class_weights)
    print("Output directory:", output_dir)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the fraud dataset for ML model training."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/paysim_feature_engineered.csv"),
        help="Path to the feature-engineered CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/week2_day3"),
        help="Directory where the prepared files will be saved.",
    )
    args = parser.parse_args()
    prepare_dataset(args.input, args.output_dir)


if __name__ == "__main__":
    main()
