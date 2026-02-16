from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score


# -------------------------------------------------
# Path resolution
# -------------------------------------------------
def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# -------------------------------------------------
# Load test data
# -------------------------------------------------
def load_test_data(root: Path):
    data_path = root / "data" / "processed" / "credit_default_v1.parquet"
    df = pd.read_parquet(data_path)

    test_df = df[df["split"] == "test"].drop(columns=["split"])
    X_test = test_df.drop(columns=["default"])
    y_test = test_df["default"].values

    return X_test, y_test


# -------------------------------------------------
# Compute precision & recall across thresholds
# -------------------------------------------------
def compute_precision_recall_curve(y_true, y_proba):
    thresholds = np.linspace(0.01, 0.99, 200)

    precisions = []
    recalls = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))

    return thresholds, np.array(precisions), np.array(recalls)


# -------------------------------------------------
# Plot function
# -------------------------------------------------
def plot_precision_recall_threshold(model_name, model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]

    thresholds, precisions, recalls = compute_precision_recall_curve(
        y_test, y_proba
    )

    plt.figure()
    plt.plot(thresholds, precisions, label="precision")
    plt.plot(thresholds, recalls, label="recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.title(f"Precision & Recall vs Threshold — {model_name}")
    plt.show()


# -------------------------------------------------
# Main execution
# -------------------------------------------------
def main():
    root = get_project_root()
    X_test, y_test = load_test_data(root)

    model_dir = root / "artifacts" / "models"

    for model_path in model_dir.glob("*.joblib"):
        model_name = model_path.stem
        model = joblib.load(model_path)

        plot_precision_recall_threshold(
            model_name, model, X_test, y_test
        )


if __name__ == "__main__":
    main()
