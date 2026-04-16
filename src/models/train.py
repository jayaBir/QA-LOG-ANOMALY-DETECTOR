import pandas as pd
import joblib
from pathlib import Path

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