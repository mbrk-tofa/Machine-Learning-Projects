"""This script is responsible for:
- Loading trained model pipelines
- Evaluating on the held-out test set
- Computing:
 - ROC-AUC
 - Precision
 - Recall
 - Estimating bootstrap confidence intervals
- then saving evaluation results
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json

from sklearn.metrics import roc_auc_score, precision_score, recall_score

# Path resolution 
def get_project_root():
    return Path(__file__).resolve().parents[2]

#Data loading
def load_processed_data(root: Path) -> pd.DataFrame:
    data_path = root / "data" / "processed" / "credit_default_v1.parquet"
    return pd.read_parquet(data_path)

def split_test_data(df: pd.DataFrame):
    test_df = df[df["split"]=="test"].drop(columns=["split"])
    X_test = test_df.drop(columns=["default"])
    y_test = test_df["default"]
    return X_test, y_test

#Metrics computation
def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred)
    }

    return metrics

#Bootstrap confidence intervals
def bootstrap_auc(y_true, y_prob, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed) #random number generator
    scores = []

    n = len(y_true)

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, n)
        score = roc_auc_score(y_true[indices], y_prob[indices])
        scores.append(score)
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)

    return lower, upper

#Evaluation loop
def evaluate_models(root: Path, X_test, y_test):
    model_dir = root / "artifacts" / "models"
    results = {}

    for model_path in model_dir.glob("*.joblib"):
        model_name = model_path.stem
        model = joblib.load(model_path)

        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test.values, y_prob)

        ci_lower, ci_upper = bootstrap_auc(y_test.values, y_prob)

        metrics["roc_auc_ci_lower"] = ci_lower
        metrics["roc_auc_ci_upper"] = ci_upper

        results[model_name] = metrics

        print(f"\nModel: {model_name}")
        print(metrics)
    
    return results

# Main execution
def main():
    root = get_project_root()

    df = load_processed_data(root)
    X_test, y_test = split_test_data(df)

    results = evaluate_models(root, X_test, y_test)

    output_path = root / "artifacts" / "evaluation_results.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nEvaluation completed and saved.")

if __name__ == "__main__":
    main()
