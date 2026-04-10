from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

app = FastAPI(title="Log Anomaly Detection Service")

DATA_PATH = Path("data/processed/anomalies_explained.csv")
MODEL_PATH = Path("src/models/model.pkl")

# -----------------------
# Load model once
# -----------------------

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
else:
    model = None

# -----------------------
# Health
# -----------------------

@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------
# Batch Detection
# -----------------------

@app.post("/detect")
def detect_anomalies(limit: int = 10):

    if not DATA_PATH.exists():
        return {"error": "Anomaly file not found"}

    df = pd.read_csv(DATA_PATH)

    anomalies = df[df["is_anomaly"] == 1]

    return {
        "total_anomalies": int(anomalies.shape[0]),
        "sample": anomalies.head(limit).to_dict(orient="records")
    }

# -----------------------
# Real-time Prediction
# -----------------------

class LogRequest(BaseModel):
    requests_per_minute: float
    error_rate: float
    avg_response_size: float


@app.post("/predict_anomaly")
def predict_anomaly(log: LogRequest):

    if model is None:
        return {"error": "Model not found. Run training first."}

    X = np.array([[
        log.requests_per_minute,
        log.error_rate,
        log.avg_response_size
    ]])

    prediction = model.predict(X)[0]
    score = model.decision_function(X)[0]

    return {
        "is_anomaly": 1 if prediction == -1 else 0,
        "anomaly_score": float(score)
    }