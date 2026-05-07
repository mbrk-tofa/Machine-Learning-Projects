from pathlib import Path
import pandas as pd
from datetime import datetime


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def log_prediction(
        input_data: dict, probability: float,
         decision: str, model_name: str ):
    root = get_project_root()
    log_path = root / "artifacts" / "inference_logs.csv"

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_name": model_name,
        "probability": probability,
        "decision": decision,
        **input_data
    }

    df = pd.DataFrame([log_entry])

    if log_path.exists():
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, index=False)