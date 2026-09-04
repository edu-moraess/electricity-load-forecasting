"""
Peak detection: identify the next forecast peak, its timing, and how it
compares to the recent average load.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PeakInfo:
    peak_value_mw: float
    peak_time: pd.Timestamp
    recent_average_mw: float
    pct_vs_recent_average: float


def detect_next_peak(
    forecast_timestamps: pd.DatetimeIndex,
    forecast_values: np.ndarray,
    recent_history: pd.Series,
    recent_window: int = 48,
) -> PeakInfo:
    if len(forecast_values) == 0:
        raise ValueError("Cannot detect a peak in an empty forecast")

    idx = int(np.argmax(forecast_values))
    peak_value = float(forecast_values[idx])
    peak_time = forecast_timestamps[idx]

    recent_avg = float(recent_history.tail(recent_window).mean())
    pct_vs_avg = ((peak_value - recent_avg) / recent_avg * 100) if recent_avg else float("nan")

    return PeakInfo(
        peak_value_mw=peak_value,
        peak_time=peak_time,
        recent_average_mw=recent_avg,
        pct_vs_recent_average=pct_vs_avg,
    )
