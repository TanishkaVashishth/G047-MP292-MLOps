"""Shared prediction helpers for batch use and a future service."""
import pandas as pd

from .preprocessing import clean_data, prepare_features


def predict_frame(model, rows: pd.DataFrame) -> pd.Series:
    cleaned = clean_data(rows)
    # The registered v1 also expects date-derived columns. An eight-field
    # request leaves those columns missing for its fitted median imputer.
    features = prepare_features(cleaned)
    expected = model.feature_names_in_
    for column in ("date_month", "date_day_of_week"):
        if column in expected and column not in features:
            features[column] = float("nan")
    features = features.loc[:, expected]
    return pd.Series(model.predict(features), index=rows.index, name="prediction")
