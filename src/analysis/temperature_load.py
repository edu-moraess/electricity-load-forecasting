"""Observed temperature–load relationship helpers."""
from __future__ import annotations

import pandas as pd


def observed_correlation(load: pd.Series, temperature: pd.Series) -> float:
    aligned = pd.concat([load, temperature], axis=1).dropna()
    if len(aligned) < 2:
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def load_by_temperature_bin(load: pd.Series, temperature: pd.Series, n_bins: int = 8) -> pd.DataFrame:
    frame = pd.DataFrame({"load_mw": load, "temperature_2m": temperature}).dropna()
    if frame.empty:
        return pd.DataFrame(columns=["temperature_bin", "mean_load_mw", "n"])
    frame["temperature_bin"] = pd.cut(frame["temperature_2m"], bins=n_bins)
    summary = (
        frame.groupby("temperature_bin", observed=True)
        .agg(mean_load_mw=("load_mw", "mean"), n=("load_mw", "size"))
        .reset_index()
    )
    summary["temperature_bin"] = summary["temperature_bin"].astype(str)
    return summary
