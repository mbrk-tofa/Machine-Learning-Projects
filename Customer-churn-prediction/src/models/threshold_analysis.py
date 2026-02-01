import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

from sklearn.metrics import confusion_matrix


# -------------------------
# Paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "logistic_model_calibrated.pkl"
FALLBACK_MODEL_PATH = PROJECT_ROOT / "models" / "logistic_model.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "processed"


# -------------------------
# Cost assumptions
# -------------------------
COST_FP = 1    # unnecessary intervention
COST_FN = 5   # missed churner


# -------------------------
# Load artifacts
# -------------------------
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return joblib.load(FALLBACK_MODEL_PATH)


def load_data():
    X_test = joblib.load(DATA_PATH / "X_test.pkl")
    y_test = joblib.load(DATA_PATH / "y_test.pkl")
    return X_test, y_test


# -------------------------
# Cost computation
# -------------------------
def expected_cost(y_true, y_pred):
    _, fp, fn, _ = confusion_matrix(y_true, y_pred).ravel()
    return fp * COST_FP + fn * COST_FN


# -------------------------
# Threshold sweep
# -------------------------
def threshold_sweep(y_true, y_prob, thresholds):
    costs = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        # Handle degenerate case explicitly
        if y_pred.sum() == 0:
            cost = y_true.sum() * COST_FN
        else:
            cost = expected_cost(y_true, y_pred)

        costs.append(cost)

    return np.array(costs)


# -------------------------
# Main
# -------------------------
def main():
    model = load_model()
    X_test, y_test = load_data()

    y_prob = model.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.0, 1.0, 200)
    costs = threshold_sweep(y_test, y_prob, thresholds)

    best_idx = np.argmin(costs)
    best_threshold = thresholds[best_idx]
    best_cost = costs[best_idx]

    # -------------------------
    # Plot
    # -------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, costs)
    plt.axvline(best_threshold, linestyle="--", label=f"Optimal threshold = {best_threshold:.2f}")
    plt.scatter(best_threshold, best_cost)
    plt.xlabel("Decision Threshold")
    plt.ylabel("Expected Cost")
    plt.title("Expected Cost vs Decision Threshold")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Optimal threshold: {best_threshold:.3f}")
    print(f"Minimum expected cost: {best_cost}")


if __name__ == "__main__":
    main()
