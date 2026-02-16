from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


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
# Expected cost function
# -------------------------------------------------
def compute_expected_cost(y_true, y_proba, threshold, cost_fp=1, cost_fn=5):
    y_pred = (y_proba >= threshold).astype(int)

    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    total_cost = cost_fp * fp + cost_fn * fn
    avg_cost = total_cost / len(y_true)

    return avg_cost


# -------------------------------------------------
# Plot threshold curve
# -------------------------------------------------
def plot_threshold_curve(model_name, model, X_test, y_test):
    thresholds = np.linspace(0.01, 0.99, 200)
    costs = []

    y_proba = model.predict_proba(X_test)[:, 1]

    for t in thresholds:
        cost = compute_expected_cost(y_test, y_proba, t)
        costs.append(cost)

    costs = np.array(costs)

    best_idx = np.argmin(costs)
    best_threshold = thresholds[best_idx]
    best_cost = costs[best_idx]

    plt.figure()
    plt.plot(thresholds, costs)
    plt.scatter(best_threshold, best_cost)
    plt.xlabel("Threshold")
    plt.ylabel("Expected Cost")
    plt.title(f"Threshold vs Expected Cost — {model_name}")
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

        plot_threshold_curve(model_name, model, X_test, y_test)


if __name__ == "__main__":
    main()
