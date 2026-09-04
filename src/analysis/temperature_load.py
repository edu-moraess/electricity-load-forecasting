"""
Descriptive analysis of the observed relationship between temperature and
load in the historical dataset.

Deliberately descriptive, not causal: this module computes correlation and
a binned average load-by-temperature view, and every public function name
and docstring avoids causal language, per the project's requirement to
label this "Observed relationship in the historical dataset" rather than
imply causation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def observed_correlation(load: pd.Series, temperature: pd.Series) -> float:
    valid = pd.DataFrame({"load": load, "temperature": temperature}).dropna()
    if len(valid) < 2:
        return float("nan")
    return float(valid["load"].corr(valid["temperature"]))


def load_by_temperature_bin(load: pd.Series, temperature: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    valid = pd.DataFrame({"load": load, "temperature": temperature}).dropna()
    if valid.empty:
        return pd.DataFrame(columns=["temperature_bin", "mean_load_mw", "n_observations"])
    valid["temperature_bin"] = pd.cut(valid["temperature"], bins=n_bins)
    grouped = (
        valid.groupby("temperature_bin", observed=True)["load"]
        .agg(mean_load_mw="mean", n_observations="count")
        .reset_index()
    )
    return grouped
