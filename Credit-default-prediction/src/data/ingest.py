import pandas as pd
import hashlib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml


#load config and read config values
"""
this is to:
    - Externalizes parameters
    - Avoids hard-coded paths and seeds
    - Enables controlled experiment changes
    - Centralizes dataset behavior
    - Ensures one source of truth
"""

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

RAW_PATH = config["data"]["raw_path"]
PROCESSED_PATH = config["data"]["processed_path"]
TARGET = config["data"]["target"]
SEED = config["data"]["random_seed"]
TEST_SIZE = config["data"]["test_size"]

#load raw data (reads raw data without modification, and raw data remains immutable)
df = pd.read_csv(RAW_PATH)

#Rename target
df = df.rename(columns={TARGET: "default"})

#Train-test split
"""
    - Deterministic (random_state)
    - Class-balance (stratify)
    - production-safe
"""
train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=df["default"]
)

# Save the processed dataset
"""
- Stores splits in a single artifac
- Enables consistent loading later
- Parquet chosen for speed and schema safety
"""
processed = {
    "train": train_df,
    "test": test_df
}
pd.concat(
    [train_df.assign(split="train"), test_df.assign(split="test")]
).to_parquet(PROCESSED_PATH, index=False)

# Compute Dataset Hash
"""
This is to:
- Create an immutable dataset fingerprint, and
- Prevent silent data drift
"""
with open(PROCESSED_PATH, "rb") as f:
    dataset_hash = hashlib.sha256(f.read()).hexdigest()

# Save version metadata
"""
This is to:
- Capture dataset identity
- Store imbalance statistics
- Enable reproducible reporting
"""
version_info = {
    "dataset": "credit_default",
    "version": "v1",
    "rows": len(df),
    "train_rows": len(train_df),
    "test_rows": len(test_df),
    "positive_rate": df["default"].mean(),
    "hash": dataset_hash,
    "random_seed": SEED
}

with open("artifacts/data_versions.json", "w") as f:
    json.dump(version_info, f, indent=2)

print("Data ingestion completed.")
print(version_info)