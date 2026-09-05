import numpy as np

from src.evaluation.metrics import all_metrics, mae, mape, rmse, smape, wape


def test_perfect_forecast_gives_zero_error():
    y = np.array([100.0, 200.0, 300.0])
    assert mae(y, y) == 0
    assert rmse(y, y) == 0
    assert mape(y, y) == 0
    assert smape(y, y) == 0
    assert wape(y, y) == 0


def test_mape_ignores_near_zero_actuals():
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([5.0, 110.0])
    # only the second row should count
    assert np.isclose(mape(y_true, y_pred), 10.0)


def test_all_metrics_returns_expected_keys():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 19.0, 33.0])
    result = all_metrics(y_true, y_pred)
    assert set(result.keys()) == {"MAE", "RMSE", "MAPE", "sMAPE", "WAPE"}
    assert all(v >= 0 for v in result.values())
