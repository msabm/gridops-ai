from fastapi.testclient import TestClient
from src.api.main import app

def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["features"] == 10


def test_forecast_endpoint():
    with TestClient(app) as client:
        response = client.get("/forecast")

    assert response.status_code == 200

    data = response.json()

    assert "forecast_origin" in data
    assert "target_timestamp" in data
    assert "predicted_total_power_consumption" in data
    assert data["model"] == "LSTM"
    assert data["features"] == 10
    assert data["lookback_steps"] == 144
    assert data["predicted_total_power_consumption"] > 0