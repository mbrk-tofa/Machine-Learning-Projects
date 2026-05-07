from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from src.monitoring.log_predictions import log_prediction


from fastapi import FastAPI
from pydantic import BaseModel


# -------------------------------------------------
# Path resolution
# -------------------------------------------------
def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


# -------------------------------------------------
# Load model selection metadata
# -------------------------------------------------
def load_model_selection(root: Path):
    path = root / "artifacts" / "model_selection.json"

    with open(path, "r") as f:
        data = json.load(f)

    return data


# -------------------------------------------------
# Determine best model
# -------------------------------------------------
def get_best_model_name(selection_data: dict) -> str:
    # Select model with lowest expected cost
    best_model = min(
        selection_data.items(),
        key=lambda x: x[1]["expected_cost"]
    )[0]

    return best_model


# -------------------------------------------------
# Load selected model
# -------------------------------------------------
def load_model(root: Path, model_name: str):
    model_path = root / "artifacts" / "models" / f"{model_name}.joblib"
    return joblib.load(model_path)


# -------------------------------------------------
# Request Schema
# -------------------------------------------------
class CreditRequest(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: float
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


# -------------------------------------------------
# Response Schema
# -------------------------------------------------
class PredictionResponse(BaseModel):
    model_name: str
    default_probability: float
    decision: str
    threshold_used: float


# -------------------------------------------------
# App Initialization
# -------------------------------------------------
app = FastAPI(title="Credit Default Prediction API")

ROOT = get_project_root()
MODEL_SELECTION = load_model_selection(ROOT)
MODEL_NAME = get_best_model_name(MODEL_SELECTION)
MODEL = load_model(ROOT, MODEL_NAME)
OPTIMAL_THRESHOLD = MODEL_SELECTION[MODEL_NAME]["optimal_threshold"]


# -------------------------------------------------
# Health Endpoint
# -------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------------------------------------
# Metadata Endpoint
# -------------------------------------------------
@app.get("/metadata")
def metadata():
    return {
        "model_name": MODEL_NAME,
        "threshold": OPTIMAL_THRESHOLD,
        "expected_cost": MODEL_SELECTION[MODEL_NAME]["expected_cost"]
    }


# -------------------------------------------------
# Prediction Endpoint
# -------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(request: CreditRequest):

    input_data = pd.DataFrame([{
        "LIMIT_BAL": request.LIMIT_BAL,
        "SEX": request.SEX,
        "EDUCATION": request.EDUCATION,
        "MARRIAGE": request.MARRIAGE,
        "AGE": request.AGE,
        "PAY_0": request.PAY_0,
        "PAY_2": request.PAY_2,
        "PAY_3": request.PAY_3,
        "PAY_4": request.PAY_4,
        "PAY_5": request.PAY_5,
        "PAY_6": request.PAY_6,
        "BILL_AMT1": request.BILL_AMT1,
        "BILL_AMT2": request.BILL_AMT2,
        "BILL_AMT3": request.BILL_AMT3,
        "BILL_AMT4": request.BILL_AMT4,
        "BILL_AMT5": request.BILL_AMT5,
        "BILL_AMT6": request.BILL_AMT6,
        "PAY_AMT1": request.PAY_AMT1,
        "PAY_AMT2": request.PAY_AMT2,
        "PAY_AMT3": request.PAY_AMT3,
        "PAY_AMT4": request.PAY_AMT4,
        "PAY_AMT5": request.PAY_AMT5,
        "PAY_AMT6": request.PAY_AMT6,
    }])

    probability = MODEL.predict_proba(input_data)[0, 1]

    decision = "reject" if probability >= OPTIMAL_THRESHOLD else "approve"
    
    

    log_prediction(
        input_data=request.dict(),
        probability=float(probability),
        decision=decision,
        model_name=MODEL_NAME
    )
    
    
    return PredictionResponse(
        model_name=MODEL_NAME,
        default_probability=float(probability),
        decision=decision,
        threshold_used=OPTIMAL_THRESHOLD,
    )

    
