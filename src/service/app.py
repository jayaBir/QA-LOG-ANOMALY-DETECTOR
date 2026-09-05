from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import os
import json
import logging
from threading import Lock

from fastapi import HTTPException

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_ENABLED = True
except ImportError:
    mlflow = None
    MLFLOW_ENABLED = False

app = FastAPI(title="Log Anomaly Detection Service")
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/processed/anomalies_explained.csv")
MODEL_PATH = Path("src/models/model.pkl")
DRIFT_REPORT_PATH = Path("data/processed/drift_report.json")
DEFAULT_MODEL_URI = "models:/qa-log-anomaly-detector@production"

# -----------------------
# Load model once
# -----------------------

def resolve_model_uri(model_uri):
    """Resolve an aliased registered model to this project's run artifact.

    Models logged by ``train_model`` are stored under the ``model`` artifact
    path. Recent MLflow registries can record a model-ID URI (``models:/m-…``)
    as a model version's source. Some clients incorrectly treat that nested
    URI as a local artifact and fail with ``No such artifact: ''``. The model
    version retains its originating run ID, which is a stable way to retrieve
    the same artifact from this tracking server.
    """
    if not model_uri.startswith("models:/"):
        return model_uri

    model_reference = model_uri.removeprefix("models:/")
    if "@" not in model_reference:
        return model_uri

    model_name, model_alias = model_reference.rsplit("@", maxsplit=1)
    if not model_name or not model_alias:
        return model_uri

    version = mlflow.MlflowClient().get_model_version_by_alias(
        model_name,
        model_alias,
    )
    if not version.run_id:
        return model_uri

    resolved_uri = f"runs:/{version.run_id}/model"
    logger.info(
        "Loading registered model %s@%s from run artifact %s",
        model_name,
        model_alias,
        resolved_uri,
    )
    return resolved_uri


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
        return mlflow.sklearn.load_model(resolve_model_uri(model_uri))

    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    return None


# Do not load the serving model while this module is imported.  A temporary
# MLflow outage (or an unavailable artifact) must not make Uvicorn exit before
# it can expose its health endpoint.  The first prediction loads the model and
# returns a useful 503 if the registry is still unavailable.
model = None
model_load_error = None
model_lock = Lock()


def get_prediction_model():
    global model, model_load_error

    if model is not None:
        return model

    with model_lock:
        if model is not None:
            return model
        try:
            model = load_prediction_model()
            model_load_error = None
        except Exception as error:
            model_load_error = error
            logger.exception("Unable to load the prediction model")
            return None

    return model

# -----------------------
# Health
# -----------------------

@app.get("/health")
def health():
    # This is intentionally a liveness endpoint.  It confirms that the HTTP
    # server is accepting traffic even while MLflow is recovering.
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Return ready only when the configured serving model can be loaded."""
    if get_prediction_model() is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Check the MLflow registry and artifact store.",
        )
    return {"status": "ready"}


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

    prediction_model = get_prediction_model()
    if prediction_model is None:
        detail = "Model is not available."
        if model_load_error is not None:
            detail += " Check the MLflow registry and artifact store."
        raise HTTPException(status_code=503, detail=detail)

    X = np.array([[
        log.requests_per_minute,
        log.error_rate,
        log.avg_response_size
    ]])

    prediction = prediction_model.predict(X)[0]
    score = prediction_model.decision_function(X)[0]

    return {
        "is_anomaly": 1 if prediction == -1 else 0,
        "anomaly_score": float(score)
    }
