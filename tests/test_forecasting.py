import numpy as np
import pandas as pd

from src.forecasting.forecast import generate_forecast, horizon_to_periods


def test_horizon_to_periods_matches_base_frequency():
    assert horizon_to_periods("15min") == 1
    assert horizon_to_periods("1h") == 2
    assert horizon_to_periods("24h") == 48
    assert horizon_to_periods("7d") == 336


def test_generate_forecast_seasonal_naive_with_bands(sample_load_series):
    residuals = np.random.RandomState(0).normal(0, 8, 200)
    result = generate_forecast(sample_load_series, "24h", "Seasonal Naive", residuals=residuals)
    assert len(result.p50) == 48
    assert result.p10 is not None and result.p90 is not None
    assert (result.p10 <= result.p50).all()
    assert (result.p50 <= result.p90).all()
    assert result.band_method == "residual_bootstrap"


def test_generate_forecast_without_residuals_has_no_bands(sample_load_series):
    result = generate_forecast(sample_load_series, "1h", "Moving Average", residuals=None)
    assert result.p10 is None and result.p90 is None
    assert result.band_method == "none"
