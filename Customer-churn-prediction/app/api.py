# app/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from pathlib import Path
import sys

# Allow imports from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.predict import predict_single


# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn probability and decision",
    version="1.0.0"
)


# -------------------------
# Input Schema
# -------------------------
class ChurnRequest(BaseModel):
    customerid        :   object 
    gender            :   object 
    seniorcitizen     :   int  
    partner           :   object 
    dependents        :   object 
    tenure            :   int  
    phoneservice      :   object 
    multiplelines     :   object 
    internetservice   :   object 
    onlinesecurity    :   object 
    onlinebackup      :   object 
    deviceprotection  :   object 
    techsupport       :   object 
    streamingtv       :   object 
    streamingmovies   :   object 
    contract          :   object 
    paperlessbilling  :   object 
    paymentmethod     :   object 
    monthlycharges    :   float
    totalcharges      :   float



# -------------------------
# Output Schema
# -------------------------
class ChurnResponse(BaseModel):
    churn_probability   : float
    decision            : int
    threshold           : float


# -------------------------
# Health Check
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------------
# Prediction Endpoint
# -------------------------
@app.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest):
    try:
        input_data: Dict[str, Any] = request.dict()
        result = predict_single(input_data)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
