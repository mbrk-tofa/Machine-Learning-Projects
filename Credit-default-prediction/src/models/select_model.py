"""This script is responsible for:
- Loading trained model pipelines
- Using test probabilities
- Defining a business cost matrix
- Computing expected cost across thresholds
- Selecting optimal threshold per model
- Selecting best model under cost
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

#Path resolution
def get_projetc_root() -> Path:
    return Path(__file__).resolve().parents[2]

#Data loading
def load_processed_data(root: Path) -> pd.DataFrame:
    data_path = root / "data" / "processed" / "credit_default_v1.parquet"
    return pd.read_parquet(data_path)

def split_test_data(df: pd.DataFrame):
    test_df = df[df["split"]== "test"].drop(columns=["split"])
    X_test = test_df.drop(columns=["default"])
    y_test = test_df["default"].values

    return X_test, y_test

#Cost function
def compute_expected_cost(y_true, y_prob, threshold, cost_fp=1, cost_fn=5):
    y_pred = (y_prob >= threshold).astype(int)

    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    total_cost = cost_fp*fp + cost_fn*fn
    avg_cost = total_cost / len(y_true)

    return avg_cost

#Threshold search
def find_optimal_threshold(y_true, y_prob):
    thresholds = np.linspace(0.01, 0.99, 200)

    costs = []
    for t in thresholds:
        cost = compute_expected_cost(y_true, y_prob, t)
        costs.append(cost)

    costs = np.array(costs)

    best_index = np.argmin(costs)
    best_threshold = thresholds[best_index]
    best_cost = costs[best_index]

    return best_threshold, best_cost

#Model Evaluation
def evaluate_models_under_cost(root: Path, X_test, y_test):
    model_dir = root / "artifacts" / "models"
    results = {}

    for model_path in model_dir.glob("*.joblib"):
        model_name = model_path.stem
        model = joblib.load(model_path)

        y_prob = model.predict_proba(X_test)[:, 1]

        best_threshold, best_cost = find_optimal_threshold(
            y_test, y_prob
        )

        results[model_name] = {
            "optimal_threshold": float(best_threshold),
            "expected_cost": float(best_cost)
        }

        print(f"\nModel: {model_name}")
        print(f"Optimal_threshold: {best_threshold:.3f}")
        print(f"Expected_cost: {best_cost:.4f}")
    
    return results

# Main execution
def main():
    root = get_projetc_root()

    df = load_processed_data(root)
    X_test, y_test = split_test_data(df)

    results = evaluate_models_under_cost(root, X_test, y_test)

    output_path = root / "artifacts" / "model_selection.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\Model selection completed and saved sucessfully")

if __name__ == "__main__":
    main()