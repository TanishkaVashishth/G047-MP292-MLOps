"""Serve the exported MP292 champion."""
import json
import logging
import sys
from functools import lru_cache
from time import perf_counter

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from src.mp292.config import ROOT, load_config
from src.mp292.predict import predict_frame


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    cows: int = Field(le=350)
    buffaloes: int
    fodder_price_rs: float = Field(ge=5)
    temp_c: float | None
    vet_visits: int
    feed_kg: float
    breed_mix: str
    village_cluster: str


class PredictionFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "event": record.getMessage(),
            "latency_ms": record.latency_ms,
        })


logger = logging.getLogger("mp292.api")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(PredictionFormatter())
logger.handlers[:] = [handler]
logger.propagate = False
logger.setLevel(logging.INFO)

app = FastAPI()


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(ROOT / load_config()["model_output"])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(body: PredictionInput):
    start = perf_counter()
    row = pd.DataFrame([body.model_dump()])
    row["temp_c"] = row["temp_c"].astype(float)
    prediction = predict_frame(load_model(), row).iloc[0]
    logger.info("prediction", extra={"latency_ms": round((perf_counter() - start) * 1000, 3)})
    return {"prediction": float(prediction)}
