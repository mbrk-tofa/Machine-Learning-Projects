from pathlib import Path
import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def detect_input_drift():
    root = get_project_root()

    training_data = pd.read_parquet(
        root / "data" / "processed" / "credit_default_v1.parquet"
    )

    training_means = training_data.mean(numeric_only=True)

    log_path = root / "artifacts" / "inference_logs.csv"

    if not log_path.exists():
        print("No inference logs yet.")
        return

    inference_data = pd.read_csv(log_path)

    inference_means = inference_data.mean(numeric_only=True)

    drift = (inference_means - training_means).abs()

    print("Input Drift (absolute mean difference):")
    print(drift.sort_values(ascending=False).head(10))

# Prediction drift

def monitor_prediction_distribution():
    root = get_project_root()
    log_path = root / "artifacts" / "inference_logs.csv"

    df = pd.read_csv(log_path)

    print("Mean predicted probability:", df["probability"].mean())
    print("Std predicted probability:", df["probability"].std())