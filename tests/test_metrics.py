"""Tests for evaluation metrics."""
from __future__ import annotations

import numpy as np

from src.evaluation.metrics import mae, rmse, mape, smape, wape, all_metrics


def test_perfect_forecast_gives_zero_error():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert mape(y, y) == 0.0
    assert smape(y, y) == 0.0
    assert wape(y, y) == 0.0


def test_mape_ignores_near_zero_actuals():
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([0.1, 1.1, 2.1])
    # Should not produce inf
    m = mape(y_true, y_pred)
    assert np.isfinite(m)


def test_all_metrics_returns_expected_keys():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.1, 2.1, 2.9])
    d = all_metrics(y, p)
    assert set(d.keys()) >= {"mae", "rmse", "mape", "smape", "wape"}
