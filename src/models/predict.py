import pandas as pd
import joblib
from pathlib import Path

FEATURE_PATH = Path("data/processed/features.csv")
MODEL_PATH = Path("src/models/model.pkl")
OUTPUT_PATH = Path("data/processed/anomalies.csv")

FEATURE_COLS = [
    "requests_per_minute",
    "error_rate",
    "avg_response_size"
]

def predict():

    print("Loading features...")
    df = pd.read_csv(FEATURE_PATH)

    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    X = df[FEATURE_COLS]

    print("Detecting anomalies...")
    df["anomaly_score"] = model.predict(X)
    df["is_anomaly"] = df["anomaly_score"].map({1: 0, -1: 1})

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Anomalies saved to {OUTPUT_PATH}")
    print(f"Total anomalies detected: {df['is_anomaly'].sum()}")


if __name__ == "__main__":
    predict()