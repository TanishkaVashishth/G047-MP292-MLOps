import numpy as np
import pandas as pd

from mp292.preprocessing import clean_data, prepare_features


def test_invalid_temperature_and_stuck_sensor_are_missing():
    rows = pd.DataFrame({
        "date": ["2026-03-12", "2026-03-20", "2026-03-29", "2026-04-01"],
        "temp_c": [999, 20, np.nan, 30],
        "buffaloes": [114, 114, 114, 114],
    })
    out = clean_data(rows)
    assert out.loc[0, "temp_c"] != 999 and pd.isna(out.loc[0, "temp_c"])
    assert pd.isna(out.loc[0, "buffaloes"]) and pd.isna(out.loc[1, "buffaloes"])
    assert out.loc[2, "buffaloes"] == 114
    assert rows.loc[0, "temp_c"] == 999  # input was not mutated


def test_features_exclude_target_leakage_and_engineer_date():
    rows = pd.DataFrame({"date": ["2026-03-12"], "record_id": ["r1"],
                         "cows": [20], "litres_paid_in_passbook": [300],
                         "litres_per_day": [299], "breed_mix": ["desi"]})
    X = prepare_features(rows)
    assert "litres_paid_in_passbook" not in X
    assert "litres_per_day" not in X
    assert {"date_month", "date_day_of_week"} <= set(X)
