import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/clean_logs.csv")
OUTPUT_PATH = Path("data/processed/features.csv")

def build_features():
    df = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])

    # Round timestamp to minute
    df["minute"] = df["timestamp"].dt.floor("min")

    # 1️⃣ Requests per minute per host
    rpm = (
        df.groupby(["host", "minute"])
        .size()
        .reset_index(name="requests_per_minute")
    )

    # 2️⃣ Error rate per host
    df["is_error"] = df["status"].astype(str).str.startswith(("4", "5"))
    error_rate = (
        df.groupby("host")["is_error"]
        .mean()
        .reset_index(name="error_rate")
    )

    # 3️⃣ Average response size per host
    avg_size = (
        df.groupby("host")["size"]
        .mean()
        .reset_index(name="avg_response_size")
    )

    # Merge features
    features = rpm.merge(error_rate, on="host", how="left")
    features = features.merge(avg_size, on="host", how="left")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)

    print(f"Feature file created: {OUTPUT_PATH}")
    print(f"Total feature rows: {len(features)}")

if __name__ == "__main__":
    build_features()