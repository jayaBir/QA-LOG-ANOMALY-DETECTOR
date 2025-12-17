from fastapi import FastAPI
import pandas as pd
from pathlib import Path

app = FastAPI(title="Log Anomaly Detection Service")

DATA_PATH = Path("data/processed/anomalies_explained.csv")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/detect")
def detect_anomalies(limit: int = 10):
    if not DATA_PATH.exists():
        return {"error": "Anomaly file not found. Run pipeline first."}

    df = pd.read_csv(DATA_PATH)
    anomalies = df[df["is_anomaly"] == 1]

    response = {
        "total_anomalies": int(anomalies.shape[0]),
        "sample": anomalies.head(limit).to_dict(orient="records")
    }

    return response