import numpy as np
import pandas as pd

from src.data.validation import validate_load_series, validate_weather_series


def test_validate_clean_series_reports_no_issues(sample_load_series):
    cleaned, report = validate_load_series(sample_load_series, expected_freq_minutes=30)
    assert report.n_duplicates_removed == 0
    assert report.n_nulls_found == 0
    assert len(cleaned) == len(sample_load_series)


def test_validate_detects_duplicates(sample_load_series):
    dup_row = sample_load_series.iloc[[5]]
    with_dup = pd.concat([sample_load_series, dup_row], ignore_index=True)
    cleaned, report = validate_load_series(with_dup, expected_freq_minutes=30)
    assert report.n_duplicates_removed == 1
    assert len(cleaned) == len(sample_load_series)


def test_validate_detects_gap(sample_load_series):
    with_gap = sample_load_series.drop(index=range(10, 20)).reset_index(drop=True)
    _, report = validate_load_series(with_gap, expected_freq_minutes=30)
    assert len(report.gaps) >= 1


def test_validate_flags_impossible_values(sample_load_series):
    corrupted = sample_load_series.copy()
    corrupted.loc[0, "load_mw"] = -500.0
    corrupted.loc[1, "load_mw"] = 999_999.0
    _, report = validate_load_series(corrupted, expected_freq_minutes=30)
    assert report.n_impossible_values == 2


def test_validate_never_fills_gaps_with_invented_values(sample_load_series):
    with_gap = sample_load_series.drop(index=range(10, 20)).reset_index(drop=True)
    cleaned, _ = validate_load_series(with_gap, expected_freq_minutes=30)
    # row count should reflect the removed rows, not be padded back up
    assert len(cleaned) == len(sample_load_series) - 10


def test_validate_weather_series_basic(sample_series):
    weather_cols = [c for c in sample_series.columns if c != "load_mw"]
    _, report = validate_weather_series(sample_series[weather_cols])
    assert report.n_rows == len(sample_series)
