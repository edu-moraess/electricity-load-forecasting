"""
Error analysis: break down forecast error by hour of day, day of week,
month, load level, peak periods, and (when available) temperature, so it's
possible to see *when* the model tends to be wrong, not just its average
error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_error_frame(
    timestamps: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    temperature: np.ndarray | None = None,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "y_true": y_true,
            "y_pred": y_pred,
            "error": np.asarray(y_pred) - np.asarray(y_true),
            "abs_error": np.abs(np.asarray(y_pred) - np.asarray(y_true)),
        }
    )
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["month"] = df["timestamp"].dt.month
    df["load_quartile"] = pd.qcut(df["y_true"], q=4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"], duplicates="drop")
    if temperature is not None:
        df["temperature"] = temperature
    return df


def error_by_hour(err_df: pd.DataFrame) -> pd.DataFrame:
    return err_df.groupby("hour")["abs_error"].mean().reset_index().rename(columns={"abs_error": "mean_abs_error"})


def error_by_day_of_week(err_df: pd.DataFrame) -> pd.DataFrame:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    grouped = err_df.groupby("day_of_week")["abs_error"].mean().reindex(order)
    return grouped.reset_index().rename(columns={"abs_error": "mean_abs_error"})


def error_by_month(err_df: pd.DataFrame) -> pd.DataFrame:
    return err_df.groupby("month")["abs_error"].mean().reset_index().rename(columns={"abs_error": "mean_abs_error"})


def error_by_load_level(err_df: pd.DataFrame) -> pd.DataFrame:
    return err_df.groupby("load_quartile", observed=True)["abs_error"].mean().reset_index().rename(
        columns={"abs_error": "mean_abs_error"}
    )


def error_during_peaks(err_df: pd.DataFrame, top_pct: float = 0.10) -> dict:
    threshold = err_df["y_true"].quantile(1 - top_pct)
    peak_rows = err_df[err_df["y_true"] >= threshold]
    non_peak_rows = err_df[err_df["y_true"] < threshold]
    return {
        "peak_mean_abs_error": float(peak_rows["abs_error"].mean()) if len(peak_rows) else float("nan"),
        "non_peak_mean_abs_error": float(non_peak_rows["abs_error"].mean()) if len(non_peak_rows) else float("nan"),
        "n_peak_rows": int(len(peak_rows)),
    }


def error_vs_temperature(err_df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame | None:
    if "temperature" not in err_df.columns:
        return None
    binned = pd.cut(err_df["temperature"], bins=n_bins)
    return err_df.groupby(binned, observed=True)["abs_error"].mean().reset_index().rename(
        columns={"abs_error": "mean_abs_error"}
    )
