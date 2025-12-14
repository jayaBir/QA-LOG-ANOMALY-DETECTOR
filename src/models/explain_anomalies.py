import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/anomalies.csv")
OUTPUT_PATH = Path("data/processed/anomalies_explained.csv")

FEATURE_COLS = [
    "requests_per_minute",
    "error_rate",
    "avg_response_size"
]

Z_THRESHOLD = 3.0  # standard statistical cutoff

def explain():
    df = pd.read_csv(INPUT_PATH)

    explanations = []

    for feature in FEATURE_COLS:
        mean = df[feature].mean()
        std = df[feature].std()

        z_col = f"{feature}_z"
        df[z_col] = (df[feature] - mean) / std

    for _, row in df.iterrows():
        reasons = []

        if row["is_anomaly"] == 1:
            for feature in FEATURE_COLS:
                if abs(row[f"{feature}_z"]) >= Z_THRESHOLD:
                    reasons.append(f"abnormal_{feature}")

        explanations.append(", ".join(reasons))

    df["explanation"] = explanations

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("Explainability added.")
    print(f"Explained anomalies saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    explain()