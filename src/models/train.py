import pandas as pd
import joblib
from pathlib import Path

try:
    import mlflow
    MLFLOW_ENABLED = True
except ImportError:
    mlflow = None
    MLFLOW_ENABLED = False

from src.models.isolation_forest import create_model

INPUT_PATH = Path("data/processed/features.csv")
MODEL_PATH = Path("src/models/model.pkl")

FEATURE_COLS = [
    "requests_per_minute",
    "error_rate",
    "avg_response_size"
]


def train_model():

    print("Loading features...")
    df = pd.read_csv(INPUT_PATH)

    X = df[FEATURE_COLS]

    if MLFLOW_ENABLED:
        MLFLOW_DIR = Path.cwd() / "mlruns"
        MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

        mlflow.set_tracking_uri(f"file://{MLFLOW_DIR.resolve()}")

        print("Setting up MLflow experiment...")
        mlflow.set_experiment("qa-log-anomaly-detector")

    if MLFLOW_ENABLED:
        with mlflow.start_run():

            print("Creating Isolation Forest model...")
            model = create_model()

            print("Training model...")
            model.fit(X)

            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

            print("Saving model...")
            joblib.dump(model, MODEL_PATH)

            # Log parameters
            mlflow.log_param("model_type", "IsolationForest")
            mlflow.log_param("features", FEATURE_COLS)
            mlflow.log_param("num_features", len(FEATURE_COLS))

            # Log metrics
            mlflow.log_metric("training_rows", len(X))

            # Log saved pickle file as artifact
            mlflow.log_artifact(str(MODEL_PATH))

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