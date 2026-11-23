from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cows: int = Field(le=350)
    buffaloes: int
    fodder_price_rs: float = Field(ge=5)
    temp_c: float | None
    vet_visits: int
    feed_kg: float
    breed_mix: str
    village_cluster: str


app = FastAPI()


@app.post("/predict")
def predict(body: Input):
    return {"prediction": float(body.cows + body.buffaloes)}


@app.get("/health")
def health():
    return {"status": "ok"}


def test_prediction_endpoint_and_input_validation():
    client = TestClient(app)
    payload = {"cows": 30, "buffaloes": 8, "fodder_price_rs": 12.0, "temp_c": None,
               "vet_visits": 1, "feed_kg": 40.0, "breed_mix": "desi", "village_cluster": "Cluster A"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200 and response.json()["prediction"] == 38.0
    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/predict", json={**payload, "cows": 351}).status_code == 422
    assert client.post("/predict", json={**payload, "fodder_price_rs": 4}).status_code == 422
    assert client.post("/predict", json={**payload, "unrequested": 1}).status_code == 422
