import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train import train_model
train_model()

from fastapi.testclient import TestClient
from src.service.app import app

client = TestClient(app)

SAMPLE_PAYLOAD = {
    "requests_per_minute": 120,
    "error_rate": 0.05,
    "avg_response_size": 5000
}


def test_predict_endpoint_status():
    resp = client.post("/predict_anomaly", json=SAMPLE_PAYLOAD)
    assert resp.status_code == 200


def test_predict_endpoint_structure():
    resp = client.post("/predict_anomaly", json=SAMPLE_PAYLOAD)
    data = resp.json()

    assert "is_anomaly" in data
    assert "anomaly_score" in data


def test_predict_anomaly_value():
    resp = client.post("/predict_anomaly", json=SAMPLE_PAYLOAD)
    data = resp.json()

    assert data["is_anomaly"] in [0, 1]


def test_predict_score_numeric():
    resp = client.post("/predict_anomaly", json=SAMPLE_PAYLOAD)
    data = resp.json()

    assert isinstance(data["anomaly_score"], (int, float))