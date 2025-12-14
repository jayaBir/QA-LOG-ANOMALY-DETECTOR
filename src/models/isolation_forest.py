import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest

INPUT_PATH = Path("data/processed/features.csv")
OUTPUT_PATH = Path("data/processed/anomalies.csv")

def detect_anomalies():
    df = pd.read_csv(INPUT_PATH)

    feature_cols = [
        "requests_per_minute",
        "error_rate",
        "avg_response_size"
    ]

    X = df[feature_cols]

    model = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42
    )

    df["anomaly_score"] = model.fit_predict(X)
    df["is_anomaly"] = df["anomaly_score"].map({1: 0, -1: 1})

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Anomaly detection complete.")
    print(f"Total anomalies detected: {df['is_anomaly'].sum()}")

if __name__ == "__main__":
    detect_anomalies()