"""PaySim Dataset Preparation and Preprocessing Module.

Prepares raw PaySim transactions into clean, validated, and enriched data
ready for ingestion by the Kafka producer and storage in Cassandra.
"""
from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np

REQUIRED_PAYSIM_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

DEFAULT_BASE_TIMESTAMP = datetime(2026, 10, 1, 0, 0, 0, tzinfo=timezone.utc)


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate that the input DataFrame contains all expected PaySim columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input transaction DataFrame.

    Returns
    -------
    Tuple[bool, List[str]]
        Boolean indicating validity and list of missing columns if any.
    """
    missing_cols = [col for col in REQUIRED_PAYSIM_COLUMNS if col not in df.columns]
    return (len(missing_cols) == 0, missing_cols)


def clean_paysim_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean raw PaySim data by checking missing values, types, and domain bounds.

    Parameters
    ----------
    df : pd.DataFrame
        Raw PaySim DataFrame.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Cleaned DataFrame and data quality audit statistics.
    """
    initial_count = len(df)
    null_counts = df.isnull().sum().to_dict()

    # Drop rows with null values across any essential columns
    df_clean = df.dropna(subset=REQUIRED_PAYSIM_COLUMNS).copy()

    # Standardize string fields
    df_clean["type"] = df_clean["type"].astype(str).str.strip().str.upper()
    df_clean["nameOrig"] = df_clean["nameOrig"].astype(str).str.strip()
    df_clean["nameDest"] = df_clean["nameDest"].astype(str).str.strip()

    # Ensure numeric columns are properly typed
    numeric_cols = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    int_cols = ["step", "isFraud", "isFlaggedFraud"]
    for col in int_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0).astype(int)

    # Filter out invalid records (e.g. non-positive amounts, invalid steps)
    valid_mask = (
        (df_clean["amount"] > 0)
        & (df_clean["step"] >= 1)
        & (df_clean["oldbalanceOrg"] >= 0)
        & (df_clean["newbalanceOrig"] >= 0)
        & (df_clean["oldbalanceDest"] >= 0)
        & (df_clean["newbalanceDest"] >= 0)
    )
    df_clean = df_clean[valid_mask].copy()

    # Remove duplicate transactions if any
    duplicates_removed = int(df_clean.duplicated(subset=["step", "type", "amount", "nameOrig", "nameDest"]).sum())
    df_clean = df_clean.drop_duplicates(subset=["step", "type", "amount", "nameOrig", "nameDest"]).copy()

    cleaned_count = len(df_clean)
    dropped_count = initial_count - cleaned_count

    stats = {
        "initial_rows": initial_count,
        "cleaned_rows": cleaned_count,
        "dropped_rows": dropped_count,
        "null_counts": null_counts,
        "duplicates_removed": duplicates_removed,
        "clean_retention_rate_pct": round((cleaned_count / initial_count) * 100, 2) if initial_count > 0 else 0.0,
    }

    return df_clean, stats


def engineer_features(
    df: pd.DataFrame,
    base_timestamp: Optional[datetime] = None,
) -> pd.DataFrame:
    """Engineer fraud detection and pipeline integration features.

    Features engineered:
    - errorBalanceOrig: Discrepancy between old balance, amount, and new balance.
    - errorBalanceDest: Discrepancy between destination balance changes and amount.
    - timestamp: ISO 8601 timestamp mapped from simulation step (1 step = 1 hour).
    - transaction_id: Unique UUID string per transaction.
    - user_id: Mapped from nameOrig.
    - payment_type: Standardized transaction type.
    - currency: Fixed standard currency ("USD").
    - is_synthetic_test_event: False (real PaySim simulation data).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned PaySim DataFrame.
    base_timestamp : Optional[datetime]
        Starting timestamp for step 1. Defaults to 2026-10-01T00:00:00Z.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame.
    """
    if base_timestamp is None:
        base_timestamp = DEFAULT_BASE_TIMESTAMP

    df_out = df.copy()

    # Balance discrepancy features (crucial for fraud detection modeling)
    # Legit: oldbalanceOrg - amount == newbalanceOrig => error == 0
    df_out["errorBalanceOrig"] = np.round(
        (df_out["oldbalanceOrg"] - df_out["amount"]) - df_out["newbalanceOrig"], 2
    )
    # Legit: oldbalanceDest + amount == newbalanceDest => error == 0
    df_out["errorBalanceDest"] = np.round(
        (df_out["oldbalanceDest"] + df_out["amount"]) - df_out["newbalanceDest"], 2
    )

    # Pipeline integration fields matching Kafka & Cassandra
    df_out["transaction_id"] = [str(uuid.uuid4()) for _ in range(len(df_out))]
    df_out["user_id"] = df_out["nameOrig"]
    df_out["payment_type"] = df_out["type"]
    df_out["currency"] = "USD"
    df_out["is_synthetic_test_event"] = False

    # Compute ISO 8601 timestamp from step
    # step 1 -> base_timestamp + 1 hour
    timestamps = [
        (base_timestamp + timedelta(hours=int(s))).isoformat()
        for s in df_out["step"]
    ]
    df_out["timestamp"] = timestamps

    return df_out


def export_pipeline_events(
    df: pd.DataFrame,
    output_json_path: str,
    max_records: Optional[int] = None,
) -> int:
    """Export transaction records to a JSON format ready for Kafka streaming.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched PaySim DataFrame.
    output_json_path : str
        Target file path for JSON output.
    max_records : Optional[int]
        Optional record limit for streaming samples.

    Returns
    -------
    int
        Number of exported records.
    """
    records_df = df.head(max_records) if max_records is not None else df

    # Prepare dictionaries matching Kafka event format and Cassandra table
    events: List[Dict[str, Any]] = []
    for _, row in records_df.iterrows():
        events.append({
            "transaction_id": str(row["transaction_id"]),
            "timestamp": str(row["timestamp"]),
            "user_id": str(row["user_id"]),
            "amount": float(row["amount"]),
            "currency": str(row["currency"]),
            "payment_type": str(row["payment_type"]),
            "is_synthetic_test_event": bool(row["is_synthetic_test_event"]),
            "name_dest": str(row["nameDest"]),
            "old_balance_orig": float(row["oldbalanceOrg"]),
            "new_balance_orig": float(row["newbalanceOrig"]),
            "old_balance_dest": float(row["oldbalanceDest"]),
            "new_balance_dest": float(row["newbalanceDest"]),
            "error_balance_orig": float(row["errorBalanceOrig"]),
            "error_balance_dest": float(row["errorBalanceDest"]),
            "is_fraud": int(row["isFraud"]),
            "is_flagged_fraud": int(row["isFlaggedFraud"]),
        })

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)

    return len(events)


def prepare_dataset(
    input_csv_path: str,
    output_csv_path: str,
    output_json_path: Optional[str] = None,
    max_stream_records: Optional[int] = 1000,
) -> Dict[str, Any]:
    """Execute end-to-end PaySim preparation pipeline.

    Parameters
    ----------
    input_csv_path : str
        Path to raw PaySim CSV.
    output_csv_path : str
        Path to save cleaned CSV.
    output_json_path : Optional[str]
        Path to save Kafka stream-ready JSON.
    max_stream_records : Optional[int]
        Limit of records for stream JSON sample.

    Returns
    -------
    Dict[str, Any]
        Pipeline execution report.
    """
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input PaySim file not found at '{input_csv_path}'")

    raw_df = pd.read_csv(input_csv_path)

    is_valid, missing_cols = validate_schema(raw_df)
    if not is_valid:
        raise ValueError(f"Schema validation failed. Missing required columns: {missing_cols}")

    cleaned_df, audit_stats = clean_paysim_data(raw_df)
    enriched_df = engineer_features(cleaned_df)

    # Save cleaned CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    enriched_df.to_csv(output_csv_path, index=False)

    exported_stream_count = 0
    if output_json_path:
        exported_stream_count = export_pipeline_events(
            enriched_df, output_json_path, max_records=max_stream_records
        )

    return {
        "audit": audit_stats,
        "output_csv": output_csv_path,
        "output_json": output_json_path,
        "exported_stream_count": exported_stream_count,
        "total_fraud_retained": int(enriched_df["isFraud"].sum()),
        "fraud_rate_pct": round(float(enriched_df["isFraud"].mean()) * 100, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare and clean PaySim transaction dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/paysim_sample.csv",
        help="Path to input raw PaySim CSV",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/processed/paysim_cleaned.csv",
        help="Path to output cleaned CSV",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="data/processed/paysim_stream_ready.json",
        help="Path to output Kafka-ready JSON file",
    )
    parser.add_argument(
        "--stream-limit",
        type=int,
        default=1000,
        help="Max events for streaming JSON sample",
    )
    args = parser.parse_args()

    report = prepare_dataset(
        input_csv_path=args.input,
        output_csv_path=args.output_csv,
        output_json_path=args.output_json,
        max_stream_records=args.stream_limit,
    )

    print("=== PaySim Data Preparation Report ===")
    print(f"Initial Rows: {report['audit']['initial_rows']}")
    print(f"Cleaned Rows: {report['audit']['cleaned_rows']}")
    print(f"Dropped Rows: {report['audit']['dropped_rows']}")
    print(f"Retention Rate: {report['audit']['clean_retention_rate_pct']}%")
    print(f"Retained Frauds: {report['total_fraud_retained']} ({report['fraud_rate_pct']}%)")
    print(f"Cleaned CSV: {report['output_csv']}")
    print(f"Stream-ready JSON: {report['output_json']} ({report['exported_stream_count']} records)")


if __name__ == "__main__":
    main()
