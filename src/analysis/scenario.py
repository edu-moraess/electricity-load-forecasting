"""
Temperature scenario analysis: shows how the trained model's forecast
shifts if temperature were offset by a fixed amount, holding all other
features constant. This is a controlled what-if exercise, not a causal
claim -- documented as such wherever it is surfaced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.lightgbm_model import LightGBMQuantileModel

SCENARIO_OFFSETS_C = [-2, -1, 0, 1, 2]


def run_temperature_scenarios(
    model: LightGBMQuantileModel,
    feature_row: pd.DataFrame,
    temperature_col: str = "temperature_2m",
    offsets: list[float] = SCENARIO_OFFSETS_C,
) -> pd.DataFrame:
    if temperature_col not in feature_row.columns:
        raise ValueError(f"'{temperature_col}' not present in the feature row; cannot run scenario")

    rows = []
    for offset in offsets:
        scenario_row = feature_row.copy()
        scenario_row[temperature_col] = scenario_row[temperature_col] + offset
        preds = model.predict(scenario_row[model.feature_names_])
        rows.append(
            {
                "temperature_offset_c": offset,
                "p10": float(preds["P10"].iloc[0]),
                "p50": float(preds["P50"].iloc[0]),
                "p90": float(preds["P90"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)
