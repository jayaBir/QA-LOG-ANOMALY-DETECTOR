from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import os
import json

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_ENABLED = True
except ImportError:
    mlflow = None
    MLFLOW_ENABLED = False

app = FastAPI(title="Log Anomaly Detection Service")

DATA_PATH = Path("data/processed/anomalies_explained.csv")
MODEL_PATH = Path("src/models/model.pkl")
DRIFT_REPORT_PATH = Path("data/processed/drift_report.json")
DEFAULT_MODEL_URI = "models:/qa-log-anomaly-detector@production"

# -----------------------
# Load model once
# -----------------------

def load_prediction_model():
    model_uri = os.getenv("MLFLOW_MODEL_URI")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    app_env = os.getenv("APP_ENV", "dev").lower()

    if not model_uri and app_env == "prod":
        model_uri = DEFAULT_MODEL_URI

    if model_uri:
        if not MLFLOW_ENABLED:
            raise RuntimeError("MLflow is required to load MLFLOW_MODEL_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        return mlflow.sklearn.load_model(model_uri)

    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    return None


model = load_prediction_model()

# -----------------------
# Health
# -----------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/drift")
def drift_status():
    if not DRIFT_REPORT_PATH.exists():
        return {"status": "not_available", "error": "Drift report not found"}

    with DRIFT_REPORT_PATH.open("r", encoding="utf-8") as report_file:
        return json.load(report_file)

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
