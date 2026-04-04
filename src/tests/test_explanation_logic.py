import pandas as pd

FEATURES = [
    "request_per_minute",
    "error_rate",
    "avg_response_size",
]


def load_data():
    return pd.read_csv("data/processed/anomalies.csv")


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

    for exp in anomalies["explanation"].dropna():
        assert any(feature in exp for feature in FEATURES), (
            f"Unknown feature referenced in explanation: {exp}"
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