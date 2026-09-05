"""
Feature engineering for the load forecasting pipeline.

Produces lag features, rolling statistics, calendar features, and weather
features, all computed strictly from past information relative to each
row's timestamp -- no data leakage. Lag/rolling windows are expressed in
number of periods and adapted to the series' actual sampling frequency
(BASE_FREQUENCY_MINUTES), per the project's "adapt to real frequency"
requirement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import BASE_FREQUENCY_MINUTES
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Lags expressed in *hours* -- converted to periods based on actual frequency.
LAG_HOURS = [0.5, 1, 2, 12, 24, 48, 168]  # 30min, 1h, 2h, half-day, 1d, 2d, 1 week
ROLLING_WINDOWS_HOURS = [3, 24, 168]


def _hours_to_periods(hours: float) -> int:
    periods = round((hours * 60) / BASE_FREQUENCY_MINUTES)
    return max(periods, 1)


def add_lag_features(df: pd.DataFrame, target_col: str = "load_mw") -> pd.DataFrame:
    out = df.copy()
    for hours in LAG_HOURS:
        periods = _hours_to_periods(hours)
        col_name = f"lag_{periods}"
        out[col_name] = out[target_col].shift(periods)
    return out


def add_rolling_features(df: pd.DataFrame, target_col: str = "load_mw") -> pd.DataFrame:
    out = df.copy()
    for hours in ROLLING_WINDOWS_HOURS:
        window = _hours_to_periods(hours)
        # shift(1) first so the rolling window never includes the current row
        shifted = out[target_col].shift(1)
        out[f"rolling_mean_{window}"] = shifted.rolling(window=window, min_periods=max(1, window // 2)).mean()
        out[f"rolling_std_{window}"] = shifted.rolling(window=window, min_periods=max(1, window // 2)).std()
        out[f"rolling_min_{window}"] = shifted.rolling(window=window, min_periods=max(1, window // 2)).min()
        out[f"rolling_max_{window}"] = shifted.rolling(window=window, min_periods=max(1, window // 2)).max()
    return out


def add_calendar_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col])
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["day_of_month"] = ts.dt.day
    out["month"] = ts.dt.month
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    # Cyclical encodings so models see hour 23 and hour 0 as close together.
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    return out


WEATHER_COLUMNS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
]


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pass through available weather columns; log which ones are missing.

    Weather is only used contemporaneously (it is exogenous and, for
    forecasting horizons, comes from Open-Meteo's own forecast) -- so no
    shifting is applied here, unlike the load-derived lag features above.
    """
    out = df.copy()
    present = [c for c in WEATHER_COLUMNS if c in out.columns]
    missing = [c for c in WEATHER_COLUMNS if c not in out.columns]
    if missing:
        logger.warning("Weather columns missing from frame, skipping: %s", missing)
    return out


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "load_mw",
    timestamp_col: str = "timestamp",
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Run the full feature engineering pipeline in the correct order.

    Order matters: lag/rolling features are derived strictly from
    `target_col` shifted into the past; calendar features are derived from
    the timestamp; weather features are passed through as-is. No feature
    here can see its own row's target value or any future value.
    """
    out = df.sort_values(timestamp_col).reset_index(drop=True)
    out = add_lag_features(out, target_col)
    out = add_rolling_features(out, target_col)
    out = add_calendar_features(out, timestamp_col)
    out = add_weather_features(out)

    if drop_incomplete:
        feature_cols = [c for c in out.columns if c.startswith(("lag_", "rolling_"))]
        n_before = len(out)
        out = out.dropna(subset=feature_cols)
        n_after = len(out)
        if n_after < n_before:
            logger.info(
                "Dropped %d leading rows without full lag/rolling history", n_before - n_after
            )

    return out.reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame, target_col: str = "load_mw", timestamp_col: str = "timestamp") -> list[str]:
    exclude = {target_col, timestamp_col}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
