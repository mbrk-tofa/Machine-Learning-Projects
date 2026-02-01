from pathlib import Path
import json
import joblib
import pandas as pd


# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "logistic_model_calibrated.pkl"
FALLBACK_MODEL_PATH = PROJECT_ROOT / "models" / "logistic_model.pkl"
POLICY_PATH = PROJECT_ROOT / "models" / "decision_threshold.json"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "churn_clean.csv"

def load_model():
    """Load calibrated model if available, else fallback."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return joblib.load(FALLBACK_MODEL_PATH)


def load_policy():
    """Load decision threshold."""
    with open(POLICY_PATH, "r") as f:
        policy = json.load(f)
    return float(policy["threshold"])


def predict_single(input_features: dict) -> dict:
    """
    Predict churn probability and decision for a single sample.
    input_features: dict of raw feature values
    """
    model = load_model()
    threshold = load_policy()

    # Convert input to DataFrame (1 row)
    X = pd.DataFrame([input_features])

    # Predict probability
    prob = model.predict_proba(X)[0, 1]

    # Decision
    decision = int(prob >= threshold)

    return {
        "churn_probability": float(prob),
        "decision": decision,
        "threshold": threshold
    }

def load_example_from_training_data(path, index=0):
    """
    Load a single example from processed training data.
    Removes target column and returns feature dict.
    """
    df = pd.read_csv(path)
    if 'churn' in df.columns:
        df = df.drop(columns=['churn'])
    
    #select one row
    example = df.iloc[index]
    return example.to_dict()

if __name__ == "__main__":
   
    input_data = load_example_from_training_data(DATA_PATH, 0)

    result = predict_single(input_data)
    print(input_data)
    print(result)
