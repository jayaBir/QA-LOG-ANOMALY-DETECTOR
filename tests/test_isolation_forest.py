import pandas as pd


def load_data():
    return pd.read_csv("data/processed/anomalies.csv")


def test_model_output_columns_exist():
    df = load_data()

    for col in ["is_anomaly", "anomaly_score"]:
        assert col in df.columns, f"Missing model output column: {col}"
        
def test_anomaly_rate_is_reasonable():
    df = load_data()
    
    anomaly_rate = df["is_anomaly"].mean()
    assert 0.001 < anomaly_rate < 0.2, (f"Unreasonable anomaly rate detected: {anomaly_rate}")
    
def test_anomalies_have_higher_scores():
    df = load_data()

    anomalies = df[df["is_anomaly"] == 1]
    normals = df[df["is_anomaly"] == 0]

    assert not anomalies.empty, "No anomalies found"
    assert not normals.empty, "No normal points found"

    assert anomalies["anomaly_score"].mean() < normals["anomaly_score"].mean(), (
        "Anomalies do not have lower scores than normal scores"
    )