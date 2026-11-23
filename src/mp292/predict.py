"""Shared prediction helpers for batch use and a future service."""
import pandas as pd

from .preprocessing import clean_data, prepare_features


def predict_frame(model, rows: pd.DataFrame) -> pd.Series:
    cleaned = clean_data(rows)
    return pd.Series(model.predict(prepare_features(cleaned)), index=rows.index, name="prediction")
