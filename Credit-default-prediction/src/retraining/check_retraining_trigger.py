from pathlib import Path
import pandas as pd


# -------------------------------------------------
# Path resolution
# -------------------------------------------------
def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# -------------------------------------------------
# Load training reference distribution
# -------------------------------------------------
def load_training_reference(root: Path):
    data_path = root / "data" / "processed" / "credit_default_v1.parquet"

    df = pd.read_parquet(data_path)

    train_df = df[df["split"] == "train"]

    return train_df["default"].mean()


# -------------------------------------------------
# Load inference distribution
# -------------------------------------------------
def load_inference_distribution(root: Path):
    log_path = root / "artifacts" / "inference_logs.csv"

    if not log_path.exists():
        return None

    df = pd.read_csv(log_path)

    if len(df) == 0:
        return None

    return df["probability"].mean()


# -------------------------------------------------
# Retraining decision
# -------------------------------------------------
def should_retrain(training_rate, inference_rate, threshold=0.20):

    if inference_rate is None:
        return False

    drift = abs(inference_rate - training_rate)

    print(f"Training mean: {training_rate:.4f}")
    print(f"Inference mean: {inference_rate:.4f}")
    print(f"Absolute drift: {drift:.4f}")

    return drift > threshold


# -------------------------------------------------
# Main execution
# -------------------------------------------------
def main():
    root = get_project_root()

    training_rate = load_training_reference(root)
    inference_rate = load_inference_distribution(root)

    retrain = should_retrain(
        training_rate,
        inference_rate
    )

    if retrain:
        print("\nRetraining triggered.")
    else:
        print("\nNo retraining needed.")


if __name__ == "__main__":
    main()