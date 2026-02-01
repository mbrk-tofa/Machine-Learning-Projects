""""
this script:
- Loads the trained model
- Loads test data
- Compute metrics
- Quantify uncertainty (confidence intervals)
- Inspects calibration
- Report results
"""

from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)


# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "logistic_model.pkl"
ARTIFACTS_PATH = PROJECT_ROOT / "data" / "processed"


def load_artifacts():
    """Load trained model and test data."""
    model = joblib.load(MODEL_PATH)
    X_test = joblib.load(ARTIFACTS_PATH / "X_test.pkl")
    y_test = joblib.load(ARTIFACTS_PATH / "y_test.pkl")
    return model, X_test, y_test


def compute_metrics(y_true, y_pred, y_prob):
    """Compute standard classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def bootstrap_ci(metric_fn, y_true, y_prob, n_bootstrap=1000, random_state=42):
    """Compute bootstrap confidence interval for a metric."""
    rng = np.random.default_rng(random_state)
    scores = []

    n = len(y_true)

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, n)
        score = metric_fn(y_true[indices], y_prob[indices])
        scores.append(score)

    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)

    return lower, upper


def main():
    model, X_test, y_test = load_artifacts()

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = compute_metrics(y_test, y_pred, y_prob)

    print("Evaluation Metrics (Point Estimates)")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k}: {v:.4f}")

    print("\nConfusion Matrix")
    print(metrics["confusion_matrix"])

    # Confidence intervals
    print("\n95% Confidence Intervals (Bootstrap)")
    for name, fn in [
        ("accuracy", lambda y, p: accuracy_score(y, (p >= 0.5).astype(int))),
        ("roc_auc", roc_auc_score),
    ]:
        low, high = bootstrap_ci(fn, y_test.values, y_prob)
        print(f"{name}: [{low:.4f}, {high:.4f}]")


if __name__ == "__main__":
    main()
