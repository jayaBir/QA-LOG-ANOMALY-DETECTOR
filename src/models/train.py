import pandas as pd
import joblib
from pathlib import Path
import os
from urllib.error import URLError
from urllib.request import urlopen

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_ENABLED = True
except ImportError:
    mlflow = None
    MLFLOW_ENABLED = False

from src.models.isolation_forest import create_model

INPUT_PATH = Path("data/processed/features.csv")
MODEL_PATH = Path("src/models/model.pkl")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "qa-log-anomaly-detector")
REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "qa-log-anomaly-detector")
DEFAULT_MLFLOW_SERVER_URI = os.getenv("DEFAULT_MLFLOW_SERVER_URI", "http://localhost:5000")

FEATURE_COLS = [
    "requests_per_minute",
    "error_rate",
    "avg_response_size"
]


def is_mlflow_server_available(tracking_uri):
    try:
        with urlopen(f"{tracking_uri.rstrip('/')}/health", timeout=2) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def configure_mlflow():
    if not MLFLOW_ENABLED:
        return

    app_env = os.getenv("APP_ENV", "dev").lower()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if app_env == "prod" and not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is required when APP_ENV=prod")

    if tracking_uri:
        print(f"Using MLflow Tracking URI: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
    elif is_mlflow_server_available(DEFAULT_MLFLOW_SERVER_URI):
        print(f"Using local MLflow server: {DEFAULT_MLFLOW_SERVER_URI}")
        mlflow.set_tracking_uri(DEFAULT_MLFLOW_SERVER_URI)
    else:
        db_path = Path.cwd() / "mlflow.db"
        local_tracking_uri = f"sqlite:///{db_path.resolve()}"
        print(f"Using local MLflow Tracking URI: {local_tracking_uri}")
        mlflow.set_tracking_uri(local_tracking_uri)

    print("Setting up MLflow experiment...")
    mlflow.set_experiment(EXPERIMENT_NAME)


def train_model():
    print("Loading features...")
    df = pd.read_csv(INPUT_PATH)
    X = df[FEATURE_COLS]

    configure_mlflow()

    if MLFLOW_ENABLED:
        with mlflow.start_run():

            print("Creating Isolation Forest model...")
            model = create_model()

            print("Training model...")
            model.fit(X)

            print("Logging to MLflow...")

            # Parameters
            mlflow.log_param("model_type", "IsolationForest")
            mlflow.log_param("features", FEATURE_COLS)
            mlflow.log_param("num_features", len(FEATURE_COLS))

            # Metrics
            mlflow.log_metric("training_rows", len(X))

            # Log model artifact
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=REGISTERED_MODEL_NAME
            )

            # Save model locally for API usage
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

            print("Saving model locally...")
            joblib.dump(model, MODEL_PATH)

            print(f"Model saved at {MODEL_PATH}")
            print("MLflow run logged successfully.")

    else:
        print("Creating Isolation Forest model...")
        model = create_model()

        print("Training model...")
        model.fit(X)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        print("Saving model...")
        joblib.dump(model, MODEL_PATH)

        print(f"Model saved at {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
