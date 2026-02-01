import pandas as pd
from pathlib import Path


# Define project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "churn_clean.csv"


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load raw churn dataset."""
    return pd.read_csv(path)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names."""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert target variable to binary."""
    df["churn"] = df["churn"].map({"Yes": 1, "No": 0})
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values."""
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
    df = df.dropna()
    return df


def save_processed_data(df: pd.DataFrame, path: Path) -> None:
    """Save cleaned dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    df = load_raw_data(RAW_DATA_PATH)
    df = clean_column_names(df)
    df = encode_target(df)
    df = handle_missing_values(df)
    save_processed_data(df, PROCESSED_DATA_PATH)


if __name__ == "__main__":
    main()
