"""PaySim Sample Dataset Generator.

Generates statistically accurate PaySim transaction data mirroring the
schema, proportions, and fraud patterns from the benchmark PaySim financial dataset.
Used for development, testing, and pipeline streaming demonstrations.
"""
from __future__ import annotations

import argparse
import os
import random
from typing import List, Dict, Any
import pandas as pd


def generate_paysim_sample(
    num_rows: int = 5000,
    fraud_ratio: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic PaySim records adhering to empirical distributions.

    Parameters
    ----------
    num_rows : int
        Total number of transactions to generate.
    fraud_ratio : float
        Proportion of fraudulent transactions.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with standard PaySim columns.
    """
    random.seed(seed)
    records: List[Dict[str, Any]] = []

    num_fraud = int(num_rows * fraud_ratio)
    num_legit = num_rows - num_fraud

    # Transaction type distributions for legitimate transactions
    # PAYMENT (~35%), CASH_OUT (~35%), CASH_IN (~20%), TRANSFER (~8%), DEBIT (~2%)
    legit_types = ["PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER", "DEBIT"]
    legit_weights = [0.35, 0.35, 0.20, 0.08, 0.02]

    # Generate legitimate records
    for i in range(num_legit):
        step = random.randint(1, 100)
        txn_type = random.choices(legit_types, weights=legit_weights, k=1)[0]
        orig_id = f"C{random.randint(100000000, 999999999)}"

        if txn_type == "PAYMENT":
            amount = round(random.uniform(5.0, 5000.0), 2)
            dest_id = f"M{random.randint(100000000, 999999999)}"
            old_orig = round(random.uniform(amount, amount * 10), 2)
            new_orig = round(old_orig - amount, 2)
            old_dest = 0.0
            new_dest = 0.0

        elif txn_type == "CASH_IN":
            amount = round(random.uniform(50.0, 50000.0), 2)
            dest_id = f"C{random.randint(100000000, 999999999)}"
            old_orig = round(random.uniform(0.0, 20000.0), 2)
            new_orig = round(old_orig + amount, 2)
            old_dest = round(random.uniform(1000.0, 100000.0), 2)
            new_dest = max(0.0, round(old_dest - amount, 2))

        elif txn_type == "DEBIT":
            amount = round(random.uniform(10.0, 2000.0), 2)
            dest_id = f"C{random.randint(100000000, 999999999)}"
            old_orig = round(random.uniform(amount, amount * 5), 2)
            new_orig = round(old_orig - amount, 2)
            old_dest = round(random.uniform(100.0, 5000.0), 2)
            new_dest = round(old_dest + amount, 2)

        elif txn_type == "CASH_OUT":
            amount = round(random.uniform(100.0, 150000.0), 2)
            dest_id = f"C{random.randint(100000000, 999999999)}"
            old_orig = round(random.uniform(amount, amount * 3), 2)
            new_orig = round(old_orig - amount, 2)
            old_dest = round(random.uniform(0.0, 50000.0), 2)
            new_dest = round(old_dest + amount, 2)

        else:  # TRANSFER
            amount = round(random.uniform(500.0, 250000.0), 2)
            dest_id = f"C{random.randint(100000000, 999999999)}"
            old_orig = round(random.uniform(amount, amount * 4), 2)
            new_orig = round(old_orig - amount, 2)
            old_dest = round(random.uniform(0.0, 80000.0), 2)
            new_dest = round(old_dest + amount, 2)

        is_fraud = 0
        is_flagged = 1 if (txn_type == "TRANSFER" and amount > 200000 and random.random() < 0.05) else 0

        records.append({
            "step": step,
            "type": txn_type,
            "amount": amount,
            "nameOrig": orig_id,
            "oldbalanceOrg": old_orig,
            "newbalanceOrig": new_orig,
            "nameDest": dest_id,
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest,
            "isFraud": is_fraud,
            "isFlaggedFraud": is_flagged,
        })

    # Generate fraudulent transactions (Only TRANSFER and CASH_OUT)
    # Reflecting account draining and large amounts characteristic of PaySim fraud
    fraud_types = ["TRANSFER", "CASH_OUT"]
    for j in range(num_fraud):
        step = random.randint(1, 100)
        txn_type = random.choice(fraud_types)
        orig_id = f"C{random.randint(100000000, 999999999)}"
        dest_id = f"C{random.randint(100000000, 999999999)}"

        amount = round(random.uniform(10000.0, 1000000.0), 2)
        # Fraud pattern: account balance is often completely drained to 0
        old_orig = amount
        new_orig = 0.0

        old_dest = round(random.uniform(0.0, 5000.0), 2)
        new_dest = round(old_dest + amount, 2)

        is_fraud = 1
        is_flagged = 1 if (txn_type == "TRANSFER" and amount > 200000 and random.random() < 0.15) else 0

        records.append({
            "step": step,
            "type": txn_type,
            "amount": amount,
            "nameOrig": orig_id,
            "oldbalanceOrg": old_orig,
            "newbalanceOrig": new_orig,
            "nameDest": dest_id,
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest,
            "isFraud": is_fraud,
            "isFlaggedFraud": is_flagged,
        })

    # Shuffle records to simulate real event arrival
    random.shuffle(records)
    df = pd.DataFrame(records)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate sample PaySim dataset")
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/paysim_sample.csv",
        help="Target output CSV file path",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=5000,
        help="Number of records to generate (default: 5000)",
    )
    parser.add_argument(
        "--fraud-ratio",
        type=float,
        default=0.02,
        help="Ratio of fraud records (default: 0.02 / 2%%)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = generate_paysim_sample(
        num_rows=args.rows,
        fraud_ratio=args.fraud_ratio,
        seed=args.seed,
    )
    df.to_csv(args.output, index=False)
    print(f"Generated {len(df)} PaySim sample records saved to '{args.output}'.")
    print(f"Fraud count: {df['isFraud'].sum()} ({df['isFraud'].mean():.2%})")
    print("Transaction types count:")
    print(df["type"].value_counts().to_string())


if __name__ == "__main__":
    main()
