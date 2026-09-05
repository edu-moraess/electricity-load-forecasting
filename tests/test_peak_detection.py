import numpy as np
import pandas as pd

from src.analysis.peak_detection import detect_next_peak


def test_detect_next_peak_finds_max_and_correct_time():
    ts = pd.date_range("2024-01-01", periods=10, freq="30min")
    values = np.array([10, 12, 15, 30, 14, 11, 9, 8, 7, 6])
    history = pd.Series([20.0] * 100)
    peak = detect_next_peak(ts, values, history)
    assert peak.peak_value_mw == 30
    assert peak.peak_time == ts[3]
    assert np.isclose(peak.pct_vs_recent_average, 50.0)


def test_detect_next_peak_raises_on_empty_forecast():
    import pytest
    with pytest.raises(ValueError):
        detect_next_peak(pd.DatetimeIndex([]), np.array([]), pd.Series([1.0, 2.0]))
