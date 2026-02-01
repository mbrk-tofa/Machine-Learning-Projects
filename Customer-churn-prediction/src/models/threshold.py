from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "logistic_model.pkl"
ARTIFACTS_PATH = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROJECT_ROOT / "models" / "decision_threshold.json"


# Cost configuration (business-driven)
COST_FP = 1   # cost of contacting a non-churner
COST_FN = 5   # cost of missing a churner


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    X_test = joblib.load(ARTIFACTS_PATH / "X_test.pkl")
    y_test = joblib.load(ARTIFACTS_PATH / "y_test.pkl")
    return model, X_test, y_test


def expected_cost(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp * COST_FP + fn * COST_FN


def evaluate_thresholds(y_true, y_prob, thresholds):
    results = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        results.append({
            "threshold": float(t),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred),
            "expected_cost": expected_cost(y_true, y_prob, t).item()
        })

    return results


def main():
    model, X_test, y_test = load_artifacts()
    y_prob = model.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.1, 0.9, 81)
    results = evaluate_thresholds(y_test.values, y_prob, thresholds)

    best = min(results, key=lambda x: x["expected_cost"])

    print("Optimal Threshold Selection")
    print("---------------------------")
    print(f"Threshold: {best['threshold']:.2f}")
    print(f"Precision: {best['precision']:.3f}")
    print(f"Recall: {best['recall']:.3f}")
    print(f"Expected Cost: {best['expected_cost']}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(best, f, indent=4)


if __name__ == "__main__":
    main()
