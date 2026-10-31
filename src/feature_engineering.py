import json
import math
from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/processed/paysim_stream_ready.json")
OUTPUT_FILE = Path("data/processed/paysim_feature_engineered.csv")

with INPUT_FILE.open("r", encoding="utf-8") as file:
    records = json.load(file)

df = pd.DataFrame(records)
print("Original dataset shape:", df.shape)

numeric_columns = [
    "amount",
    "old_balance_orig",
    "new_balance_orig",
    "old_balance_dest",
    "new_balance_dest",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df["orig_balance_change"] = df["old_balance_orig"] - df["new_balance_orig"]
df["dest_balance_change"] = df["new_balance_dest"] - df["old_balance_dest"]
df["orig_balance_error"] = df["orig_balance_change"] - df["amount"]
df["dest_balance_error"] = df["dest_balance_change"] - df["amount"]
df["amount_to_orig_balance_ratio"] = (
    df["amount"] / df["old_balance_orig"].replace(0, float("nan"))
)
df["log_amount"] = df["amount"].map(
    lambda x: math.log1p(max(x, 0)) if pd.notna(x) else float("nan")
)

timestamp = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
df["transaction_hour"] = timestamp.dt.hour
df["transaction_day_of_week"] = timestamp.dt.dayofweek
df["transaction_month"] = timestamp.dt.month

print("Missing values:")
print(df.isnull().sum().to_string())
print("Duplicate transaction IDs:", df["transaction_id"].duplicated().sum())

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("Saved dataset:", OUTPUT_FILE)
print("Final dataset shape:", df.shape)
