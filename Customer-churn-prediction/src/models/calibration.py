from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "logistic_model.pkl"
ARTIFACTS_PATH = PROJECT_ROOT / "data" / "processed"
CALIBRATED_MODEL_PATH = PROJECT_ROOT / "models" / "logistic_model_calibrated.pkl"


def load_artifacts():
    """Load trained model and test data."""
    model = joblib.load(MODEL_PATH)
    X_test = joblib.load(ARTIFACTS_PATH / "X_test.pkl")
    y_test = joblib.load(ARTIFACTS_PATH / "y_test.pkl")
    return model, X_test, y_test


def plot_calibration_curve(y_true, y_prob, n_bins=10):
    """Plot reliability (calibration) curve."""
    frac_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    plt.figure(figsize=(6, 6))
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve (Reliability Diagram)")
    plt.legend()
    plt.grid(True)
    plt.show()


def compute_brier_score(y_true, y_prob):
    """Compute Brier score."""
    return brier_score_loss(y_true, y_prob)


def calibrate_model(model, X_test, y_test, method="sigmoid"):
    """
    Apply post-hoc calibration.
    method = 'sigmoid' (Platt scaling) or 'isotonic'
    """
    calibrated_model = CalibratedClassifierCV(
        base_estimator=model,
        method=method,
        cv="prefit"
    )
    calibrated_model.fit(X_test, y_test)
    return calibrated_model


def main():
    model, X_test, y_test = load_artifacts()

    # Original probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Original Model Calibration")
    brier = compute_brier_score(y_test, y_prob)
    print(f"Brier score: {brier:.4f}")

    plot_calibration_curve(y_test, y_prob)

    # Optional: apply calibration
    print("\nApplying Platt scaling (sigmoid calibration)...")
    calibrated_model = calibrate_model(model, X_test, y_test, method="sigmoid")

    y_prob_cal = calibrated_model.predict_proba(X_test)[:, 1]
    brier_cal = compute_brier_score(y_test, y_prob_cal)

    print(f"Calibrated Brier score: {brier_cal:.4f}")

    plot_calibration_curve(y_test, y_prob_cal)

    # Save calibrated model
    joblib.dump(calibrated_model, CALIBRATED_MODEL_PATH)
    print(f"\nCalibrated model saved to: {CALIBRATED_MODEL_PATH}")


if __name__ == "__main__":
    main()
