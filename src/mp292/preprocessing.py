"""Deterministic cleaning and train-fitted feature preprocessing."""
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import load_config

ID_COLUMNS = ("record_id", "date")


def clean_data(frame: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Replace documented sensor failures with NaN; leave imputing to fitted pipeline.

    Stage 1 found 102 consecutive readings of buffaloes=114 covering the stated
    inclusive dates. This rule scopes removal to that verified run, rather than
    treating every legitimate 114 reading elsewhere as invalid.
    """
    cfg = config or load_config()
    cleaned = frame.copy(deep=True)
    if "temp_c" in cleaned:
        cleaned["temp_c"] = pd.to_numeric(cleaned["temp_c"], errors="coerce")
        cleaned.loc[cleaned["temp_c"].eq(999), "temp_c"] = np.nan
    sensor = cfg["data"]["stuck_sensor"]
    col = sensor["column"]
    if col in cleaned and "date" in cleaned:
        dates = pd.to_datetime(cleaned["date"], errors="coerce")
        interval = dates.between(pd.Timestamp(sensor["start_date"]), pd.Timestamp(sensor["end_date"]))
        cleaned.loc[interval & cleaned[col].eq(sensor["value"]), col] = np.nan
    return cleaned


def prepare_features(frame: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Create model inputs and enforce the target-leakage exclusion."""
    cfg = config or load_config()
    target = cfg["data"]["target"]
    forbidden = {target, cfg["data"]["leakage_column"], *ID_COLUMNS}
    features = frame.drop(columns=[c for c in forbidden if c in frame], errors="ignore").copy()
    if "date" in frame:
        date = pd.to_datetime(frame["date"], errors="coerce")
        features["date_month"] = date.dt.month.astype("float64")
        features["date_day_of_week"] = date.dt.dayofweek.astype("float64")
    return features


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in features.columns if c not in numeric]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop")
