"""Error analysis utilities for forecast residuals."""
from __future__ import annotations

import pandas as pd


def build_error_frame(y_true: pd.Series, y_pred: pd.Series, timestamps: pd.Series | None = None) -> pd.DataFrame:
    frame = pd.DataFrame({
        "y_true": y_true.reset_index(drop=True),
        "y_pred": y_pred.reset_index(drop=True),
    })
    frame["error"] = frame["y_true"] - frame["y_pred"]
    frame["abs_error"] = frame["error"].abs()
    if timestamps is not None:
        frame["timestamp"] = pd.to_datetime(timestamps.reset_index(drop=True))
        frame["hour"] = frame["timestamp"].dt.hour
        frame["dow"] = frame["timestamp"].dt.dayofweek
        frame["month"] = frame["timestamp"].dt.month
    return frame


def error_by_hour(error_frame: pd.DataFrame) -> pd.DataFrame:
    if "hour" not in error_frame.columns:
        return pd.DataFrame()
    return (
        error_frame.groupby("hour", as_index=False)["abs_error"]
        .mean()
        .rename(columns={"abs_error": "mean_abs_error"})
    )


def error_during_peaks(error_frame: pd.DataFrame, peak_threshold_quantile: float = 0.9) -> dict:
    if "y_true" not in error_frame.columns:
        return {}
    threshold = error_frame["y_true"].quantile(peak_threshold_quantile)
    peaks = error_frame[error_frame["y_true"] >= threshold]
    non_peaks = error_frame[error_frame["y_true"] < threshold]
    return {
        "peak_mean_abs_error": float(peaks["abs_error"].mean()) if len(peaks) else float("nan"),
        "non_peak_mean_abs_error": float(non_peaks["abs_error"].mean()) if len(non_peaks) else float("nan"),
        "n_peak": int(len(peaks)),
        "n_non_peak": int(len(non_peaks)),
    }
