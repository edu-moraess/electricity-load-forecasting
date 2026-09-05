"""
Forecast accuracy metrics: MAE, RMSE, MAPE, sMAPE, WAPE.

MAPE and sMAPE are undefined (division by ~0) when actual load is near
zero -- which does not happen for grid-scale electricity demand, but the
functions still guard against it defensively rather than raising or
silently returning inf/NaN into a report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-6


def _to_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error, in percent. Rows where |y_true| is
    near zero are excluded from the average (documented, not silently
    zeroed) since the ratio is not meaningful there."""
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    mask = np.abs(y_true) > _EPS
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE, in percent. Denominator is (|y_true|+|y_pred|)/2;
    rows where that is near zero are excluded."""
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom > _EPS
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def wape(y_true, y_pred) -> float:
    """Weighted Absolute Percentage Error: sum(|error|) / sum(|actual|), in percent."""
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    denom = np.sum(np.abs(y_true))
    if denom <= _EPS:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)


def all_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
    }


def metrics_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Turn {model_name: {metric: value}} into a tidy DataFrame for display."""
    return pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
