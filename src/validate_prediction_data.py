"""Validate transactions and prepare them for the trained fraud model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import issparse


MODEL_INPUT_COLUMNS = [
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

NUMERIC_COLUMNS = [
    column for column in MODEL_INPUT_COLUMNS if column != "payment_type"
]

BALANCE_COLUMNS = [
    "old_balance_orig",
    "new_balance_orig",
    "old_balance_dest",
    "new_balance_dest",
]

OUTPUT_FEATURES_NAME = "prediction_ready_features.csv"
OUTPUT_PREDICTIONS_NAME = "predictions.csv"
OUTPUT_REJECTED_NAME = "rejected_transactions.csv"
OUTPUT_REPORT_NAME = "data_quality_report.json"


def mark_reason(reasons: pd.Series, mask, reason: str) -> None:
    """Append a validation reason to each matching row."""
    mask = pd.Series(mask, index=reasons.index).fillna(False).astype(bool)

    for index in reasons.index[mask]:
        previous = reasons.at[index]
        reasons.at[index] = (
            f"{previous};{reason}" if previous else reason
        )


def load_artifacts(preprocessor_path: Path, model_path: Path):
    """Load the saved training-fitted preprocessor and Random Forest."""
    preprocessor_path = Path(preprocessor_path)
    model_path = Path(model_path)

    for path in (preprocessor_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required model artifact not found: {path}")

    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)

    expected_inputs = list(preprocessor.feature_names_in_)
    if expected_inputs != MODEL_INPUT_COLUMNS:
        raise ValueError(
            "Saved preprocessor input schema does not match this script. "
            f"Expected: {MODEL_INPUT_COLUMNS}; found: {expected_inputs}"
        )

    transformed_names = list(preprocessor.get_feature_names_out())

    if hasattr(model, "feature_names_in_"):
        model_names = list(model.feature_names_in_)
        if model_names != transformed_names:
            raise ValueError(
                "Model feature names do not match the saved preprocessor output."
            )

    if not hasattr(model, "predict_proba"):
        raise ValueError("The saved model does not support probability predictions.")

    return preprocessor, model


def known_payment_types(preprocessor) -> set[str]:
    """Read the payment categories learned by the saved encoder."""
    try:
        encoder = (
            preprocessor.named_transformers_["categorical"]
            .named_steps["onehot"]
        )
        return {str(value).strip().upper() for value in encoder.categories_[0]}
    except (KeyError, AttributeError, IndexError):
        raise ValueError(
            "Could not retrieve payment types from the saved preprocessor."
        ) from None


def validate_transactions(
    df: pd.DataFrame,
    payment_types: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Validate transactions and separate accepted from rejected rows."""
    if df.empty:
        raise ValueError("The input dataset contains no rows.")

    df = df.copy().reset_index(drop=True)

    missing_columns = [
        column for column in MODEL_INPUT_COLUMNS
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required prediction columns: {missing_columns}"
        )

    features = df[MODEL_INPUT_COLUMNS].copy()
    reasons = pd.Series("", index=df.index, dtype="object")

    missing_counts = {
        column: int(features[column].isna().sum())
        for column in MODEL_INPUT_COLUMNS
    }
    numeric_parse_failures = {}

    # Validate transaction IDs when an ID column is supplied.
    if "transaction_id" in df.columns:
        identifiers = df["transaction_id"].astype("string").str.strip()
        missing_ids = identifiers.isna() | identifiers.eq("")
        mark_reason(reasons, missing_ids, "missing_transaction_id")

        duplicate_ids = (
            identifiers.notna()
            & identifiers.ne("")
            & identifiers.duplicated(keep=False)
        )
        mark_reason(reasons, duplicate_ids, "duplicate_transaction_id")

    # Convert numeric predictors safely; never allow infinity into the model.
    for column in NUMERIC_COLUMNS:
        raw = features[column]
        converted = pd.to_numeric(raw, errors="coerce")

        parse_failed = raw.notna() & converted.isna()
        non_finite = converted.isin([np.inf, -np.inf])
        numeric_parse_failures[column] = int(
            parse_failed.sum() + non_finite.sum()
        )

        # Reject non-empty malformed values and infinities. Genuine blanks
        # can be imputed by the saved training-fitted preprocessing pipeline.
        if column != "amount":
            mark_reason(
                reasons,
                parse_failed | non_finite,
                f"invalid_numeric_{column}",
            )

        # Balances must never be negative.
        if column in BALANCE_COLUMNS:
            mark_reason(
                reasons,
                converted.notna() & converted.lt(0),
                f"negative_{column}",
            )

        if column == "amount":
            mark_reason(
                reasons,
                converted.isna() | non_finite,
                "missing_or_invalid_amount",
            )
            mark_reason(
                reasons,
                converted.notna() & converted.le(0),
                "non_positive_amount",
            )

        if column == "transaction_hour":
            mark_reason(
                reasons,
                converted.notna() & ~converted.between(0, 23),
                "transaction_hour_out_of_range",
            )

        if column == "transaction_day_of_week":
            mark_reason(
                reasons,
                converted.notna() & ~converted.between(0, 6),
                "transaction_day_of_week_out_of_range",
            )

        features[column] = converted.replace(
            [np.inf, -np.inf], np.nan
        )

    # Normalize categories before checking whether they were seen in training.
    raw_payment_types = features["payment_type"].astype("string").str.strip()
    normalized_types = raw_payment_types.str.upper()
    missing_types = normalized_types.isna() | normalized_types.eq("")
    mark_reason(reasons, missing_types, "missing_payment_type")

    unsupported_types = (
        ~missing_types & ~normalized_types.isin(payment_types)
    )
    mark_reason(reasons, unsupported_types, "unsupported_payment_type")
    features["payment_type"] = normalized_types

    rejected_mask = reasons.ne("")
    accepted_features = features.loc[~rejected_mask].copy()

    rejected = df.loc[rejected_mask].copy()
    rejected["validation_errors"] = reasons.loc[rejected_mask]

    rejection_counts = {}
    for reason_string in reasons.loc[rejected_mask]:
        for reason in reason_string.split(";"):
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    audit = {
        "rows_received": int(len(df)),
        "rows_accepted": int((~rejected_mask).sum()),
        "rows_rejected": int(rejected_mask.sum()),
        "missing_values_before_preprocessing": missing_counts,
        "numeric_parse_failures": numeric_parse_failures,
        "rejection_reason_counts": rejection_counts,
        "accepted_missing_numeric_values_for_imputation": {
            column: int(accepted_features[column].isna().sum())
            for column in NUMERIC_COLUMNS
        },
    }

    return accepted_features, rejected, audit


def prepare_and_predict(
    input_path: Path,
    output_dir: Path,
    preprocessor_path: Path,
    model_path: Path,
    threshold: float = 0.5,
) -> dict:
    """Validate transactions, preprocess accepted rows, and predict fraud."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Prediction threshold must be between 0 and 1.")

    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input transaction file not found: {input_path}")

    df = pd.read_csv(input_path)
    preprocessor, model = load_artifacts(preprocessor_path, model_path)
    payment_types = known_payment_types(preprocessor)

    accepted, rejected, audit = validate_transactions(df, payment_types)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = list(preprocessor.get_feature_names_out())

    if not accepted.empty:
        transformed = preprocessor.transform(accepted)

        if issparse(transformed):
            transformed = transformed.toarray()

        ready = pd.DataFrame(
            np.asarray(transformed),
            columns=feature_names,
            index=accepted.index,
        )

        if ready.isna().any().any() or not np.isfinite(
            ready.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "Preprocessing produced missing or non-finite model features."
            )

        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError("The loaded model does not contain fraud class 1.")

        fraud_index = classes.index(1)
        probabilities = model.predict_proba(ready)[:, fraud_index]
        predictions = (probabilities >= threshold).astype(int)

        # Keep source information for traceability, but don't present any
        # existing fraud label as an input to the prediction process.
        source_rows = df.loc[accepted.index].copy()
        label_columns = [
            column for column in
            ("is_fraud", "isFraud", "is_flagged_fraud", "isFlaggedFraud")
            if column in source_rows.columns
        ]
        source_rows = source_rows.drop(columns=label_columns).reset_index(drop=True)

        predictions_df = source_rows.copy()
        predictions_df["fraud_probability"] = probabilities
        predictions_df["predicted_is_fraud"] = predictions
    else:
        ready = pd.DataFrame(columns=feature_names)
        source_columns = [
            column for column in df.columns
            if column not in (
                "is_fraud", "isFraud", "is_flagged_fraud", "isFlaggedFraud"
            )
        ]
        predictions_df = df.iloc[0:0][source_columns].copy()
        predictions_df["fraud_probability"] = pd.Series(dtype=float)
        predictions_df["predicted_is_fraud"] = pd.Series(dtype=int)

    feature_path = output_dir / OUTPUT_FEATURES_NAME
    predictions_path = output_dir / OUTPUT_PREDICTIONS_NAME
    rejected_path = output_dir / OUTPUT_REJECTED_NAME
    report_path = output_dir / OUTPUT_REPORT_NAME

    ready.to_csv(feature_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    rejected.to_csv(rejected_path, index=False)

    audit.update({
        "input_file": str(input_path),
        "prediction_threshold": float(threshold),
        "model_file": str(model_path),
        "preprocessor_file": str(preprocessor_path),
        "output_features": str(feature_path),
        "output_predictions": str(predictions_path),
        "output_rejected_transactions": str(rejected_path),
        "rows_prepared_for_prediction": int(len(ready)),
        "transformed_feature_count": int(len(feature_names)),
        "missing_values_after_preprocessing": int(ready.isna().sum().sum()),
        "prediction_positive_count": int(
            predictions_df["predicted_is_fraud"].sum()
        ),
    })

    report_path.write_text(
        json.dumps(audit, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\n=== Prediction Data Quality Report ===")
    print(f"Input rows: {audit['rows_received']}")
    print(f"Accepted rows: {audit['rows_accepted']}")
    print(f"Rejected rows: {audit['rows_rejected']}")
    print(f"Features prepared: {audit['transformed_feature_count']}")
    print(f"Rows predicted: {audit['rows_prepared_for_prediction']}")
    print(f"Predicted fraud cases: {audit['prediction_positive_count']}")
    print(f"Prediction threshold: {threshold}")
    print(f"Features CSV: {feature_path}")
    print(f"Predictions CSV: {predictions_path}")
    print(f"Rejected rows CSV: {rejected_path}")
    print(f"Quality report JSON: {report_path}")

    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and preprocess transaction data for fraud prediction."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/paysim_feature_engineered.csv"),
        help="Input CSV containing model input features.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/week3_day3"),
        help="Directory for prediction-ready outputs and the quality report.",
    )
    parser.add_argument(
        "--preprocessor",
        type=Path,
        default=Path("data/processed/week2_day3/preprocessor.joblib"),
        help="Saved training-fitted preprocessing pipeline.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/random_forest_model.joblib"),
        help="Saved Random Forest model.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fraud probability threshold from 0 to 1.",
    )
    args = parser.parse_args()

    prepare_and_predict(
        input_path=args.input,
        output_dir=args.output_dir,
        preprocessor_path=args.preprocessor,
        model_path=args.model,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
