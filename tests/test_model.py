import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from mp292.preprocessing import build_preprocessor, clean_data, prepare_features


def test_pipeline_fits_with_missing_and_unseen_categories():
    frame = pd.DataFrame({"date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
                          "cows": [10, np.nan, 20, 15], "buffaloes": [3, 4, 5, 2],
                          "temp_c": [20, 22, np.nan, 25], "breed_mix": ["desi", "mixed", "desi", "crossbred"],
                          "litres_per_day": [100, 110, 130, 115]})
    X = prepare_features(clean_data(frame))
    model = Pipeline([("prep", build_preprocessor(X)),
                      ("reg", RandomForestRegressor(n_estimators=10, random_state=3))])
    model.fit(X, frame.litres_per_day)
    query = pd.DataFrame({"date": ["2026-02-01"], "cows": [None], "buffaloes": [3],
                          "temp_c": [None], "breed_mix": ["new category"]})
    assert np.isfinite(model.predict(prepare_features(query))).all()
