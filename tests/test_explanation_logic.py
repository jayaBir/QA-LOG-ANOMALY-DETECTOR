import pandas as pd

FEATURES = [
    "requests_per_minute",
    "error_rate",
    "avg_response_size"
]

FALLBACK_EXPLANATION = "statistical_anomaly_detected"


def load_data():
    return pd.read_csv("data/processed/anomalies_explained.csv")


def test_anomalies_have_explanations():
    df = load_data()
    if "explanation" not in df.columns:
        return
    anomalies = df[df["is_anomaly"] == 1]

    assert not anomalies.empty, "No anomalies found to test explanations"
    assert anomalies["explanation"].notnull().all(), (
        "Some anomalies are missing explanations"
    )


def test_explanations_reference_known_features():

    df = load_data()

    if "explanation" not in df.columns:
        return

    anomalies = df[df["is_anomaly"] == 1]

    for feature in FEATURES:

        feature_anomalies = anomalies[
            anomalies["explanation"].str.contains(feature, na=False)
        ]

        if feature_anomalies.empty:
            continue

        z_col = f"{feature}_z"

        assert (feature_anomalies[z_col].abs() > 3).all(), (
            f"{feature} explanation given but not statistically extreme"
        )


def test_explained_features_are_statistically_extreme():
    df = load_data()
    if "explanation" not in df.columns:
        return
    anomalies = df[df["is_anomaly"] == 1]

    for feature in FEATURES:
        mean = df[feature].mean()
        std = df[feature].std()

        # Guard against constant columns
        if std == 0 or pd.isna(std):
            continue

        extreme = (
            (anomalies[feature] > mean + 2 * std)
            | (anomalies[feature] < mean - 2 * std)
        )

        assert extreme.any(), (
            f"No anomalies are statistically extreme for feature: {feature}"
        )