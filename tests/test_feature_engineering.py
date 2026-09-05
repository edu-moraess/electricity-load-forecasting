import numpy as np
import pandas as pd

from src.features.engineering import build_feature_matrix, get_feature_columns


def test_build_feature_matrix_has_no_leakage(sample_series):
    featured = build_feature_matrix(sample_series)
    # lag_1 at row i must equal load_mw at the corresponding earlier row,
    # never the current row's own value.
    merged = sample_series.set_index("timestamp")
    for _, row in featured.sample(min(20, len(featured)), random_state=0).iterrows():
        expected_ts = row["timestamp"] - pd.Timedelta(minutes=30)
        if expected_ts in merged.index:
            assert np.isclose(row["lag_1"], merged.loc[expected_ts, "load_mw"])


def test_build_feature_matrix_drops_incomplete_leading_rows(sample_series):
    featured = build_feature_matrix(sample_series, drop_incomplete=True)
    assert featured.isna().sum().sum() == 0 or featured[[c for c in featured.columns if c.startswith(("lag_", "rolling_"))]].isna().sum().sum() == 0


def test_calendar_features_are_bounded(sample_series):
    featured = build_feature_matrix(sample_series)
    assert featured["hour"].between(0, 23).all()
    assert featured["day_of_week"].between(0, 6).all()
    assert featured["is_weekend"].isin([0, 1]).all()


def test_get_feature_columns_excludes_target_and_timestamp(sample_series):
    featured = build_feature_matrix(sample_series)
    cols = get_feature_columns(featured)
    assert "load_mw" not in cols
    assert "timestamp" not in cols
    assert len(cols) > 0
