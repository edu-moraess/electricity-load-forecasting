"""Peak detection on forecast series."""
from __future__ import annotations

import pandas as pd


def detect_next_peak(forecast: pd.DataFrame, recent_avg: float | None = None) -> dict:
    """Return the maximum forecasted load and its timestamp within the horizon.

    Parameters
    ----------
    forecast : DataFrame with at least columns timestamp and p50 (or load_mw).
    recent_avg : optional recent average load for percentage comparison.
    """
    if forecast.empty:
        raise ValueError("Forecast is empty; cannot detect peak.")

    value_col = "p50" if "p50" in forecast.columns else "load_mw"
    if value_col not in forecast.columns:
        raise ValueError(f"Forecast must contain '{value_col}' or 'load_mw'.")

    idx = forecast[value_col].idxmax()
    row = forecast.loc[idx]
    peak_mw = float(row[value_col])
    peak_time = row["timestamp"] if "timestamp" in forecast.columns else idx

    result = {
        "peak_mw": peak_mw,
        "peak_time": peak_time,
    }
    if recent_avg is not None and recent_avg != 0:
        result["pct_above_avg"] = 100.0 * (peak_mw - recent_avg) / recent_avg
    else:
        result["pct_above_avg"] = float("nan")
    return result
