import math

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)
PAYLOAD = {
    "cows": 350, "buffaloes": 85, "fodder_price_rs": 5, "temp_c": 28.0,
    "vet_visits": 3, "feed_kg": 1500.0, "breed_mix": "mixed",
    "village_cluster": "Cluster A",
}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_prediction_with_exported_model():
    response = client.post("/predict", json=PAYLOAD)
    assert response.status_code == 200
    assert math.isfinite(response.json()["prediction"])


def test_invalid_requests_do_not_load_model(monkeypatch):
    def unexpected_model_load():
        raise AssertionError("Invalid input reached the model loader")

    monkeypatch.setattr("app.main.load_model", unexpected_model_load)
    for fields in ({"cows": 351}, {"fodder_price_rs": 4.99}, {"date": "2026-07-01"}):
        assert client.post("/predict", json={**PAYLOAD, **fields}).status_code == 422
