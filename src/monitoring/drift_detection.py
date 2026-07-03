import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "requests_per_minute",
    "error_rate",
    "avg_response_size"
]

CURRENT_FEATURES_PATH = Path("data/processed/features.csv")
BASELINE_FEATURES_PATH = Path("data/reference/features_baseline.csv")
DRIFT_REPORT_PATH = Path("data/processed/drift_report.json")

PSI_THRESHOLD = 0.2
MEAN_SHIFT_STD_THRESHOLD = 2.0
EPSILON = 1e-6


def calculate_psi(expected, actual, bins=10):
    expected_values = pd.Series(expected).dropna().astype(float)
    actual_values = pd.Series(actual).dropna().astype(float)

    if expected_values.empty or actual_values.empty:
        return 0.0

    if expected_values.nunique() <= 1:
        return 0.0

    _, bin_edges = np.histogram(expected_values, bins=bins)
    expected_counts, _ = np.histogram(expected_values, bins=bin_edges)
    actual_counts, _ = np.histogram(actual_values, bins=bin_edges)

    expected_pct = expected_counts / max(expected_counts.sum(), 1)
    actual_pct = actual_counts / max(actual_counts.sum(), 1)

    expected_pct = np.clip(expected_pct, EPSILON, None)
    actual_pct = np.clip(actual_pct, EPSILON, None)

    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return float(np.sum(psi_values))


def build_feature_drift_summary(feature_name, baseline_df, current_df):
    baseline = baseline_df[feature_name]
    current = current_df[feature_name]

    baseline_mean = float(baseline.mean())
    current_mean = float(current.mean())
    baseline_std = float(baseline.std(ddof=0))
    mean_shift_std = 0.0

    if baseline_std > 0:
        mean_shift_std = abs(current_mean - baseline_mean) / baseline_std

    psi = calculate_psi(baseline, current)
    drift_detected = psi >= PSI_THRESHOLD or mean_shift_std >= MEAN_SHIFT_STD_THRESHOLD

    return {
        "baseline_mean": baseline_mean,
        "current_mean": current_mean,
        "baseline_std": baseline_std,
        "current_std": float(current.std(ddof=0)),
        "psi": psi,
        "mean_shift_std": float(mean_shift_std),
        "drift_detected": bool(drift_detected)
    }


def detect_drift(
    current_path=CURRENT_FEATURES_PATH,
    baseline_path=BASELINE_FEATURES_PATH,
    report_path=DRIFT_REPORT_PATH
):
    current_path = Path(current_path)
    baseline_path = Path(baseline_path)
    report_path = Path(report_path)

    current_df = pd.read_csv(current_path)

    if not baseline_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        current_df.to_csv(baseline_path, index=False)
        report = {
            "status": "baseline_created",
            "drift_detected": False,
            "baseline_path": str(baseline_path),
            "current_path": str(current_path),
            "features": {}
        }
        write_report(report, report_path)
        return report

    baseline_df = pd.read_csv(baseline_path)
    missing_features = [
        feature for feature in FEATURE_COLS
        if feature not in baseline_df.columns or feature not in current_df.columns
    ]

    if missing_features:
        raise ValueError(f"Missing drift feature columns: {missing_features}")

    feature_summaries = {
        feature: build_feature_drift_summary(feature, baseline_df, current_df)
        for feature in FEATURE_COLS
    }
    drifted_features = [
        feature for feature, summary in feature_summaries.items()
        if summary["drift_detected"]
    ]

    report = {
        "status": "checked",
        "drift_detected": bool(drifted_features),
        "drifted_features": drifted_features,
        "thresholds": {
            "psi": PSI_THRESHOLD,
            "mean_shift_std": MEAN_SHIFT_STD_THRESHOLD
        },
        "baseline_path": str(baseline_path),
        "current_path": str(current_path),
        "features": feature_summaries
    }
    write_report(report, report_path)
    return report


def write_report(report, report_path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)


if __name__ == "__main__":
    result = detect_drift()
    print(json.dumps(result, indent=2))
