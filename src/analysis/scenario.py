"""Temperature scenario analysis (controlled what-if)."""
from __future__ import annotations

import pandas as pd


def run_temperature_scenarios(model, last_row: pd.DataFrame, deltas: list[float] | None = None) -> pd.DataFrame:
    """Re-run a fitted LightGBM quantile model under temperature shifts.

    This is a controlled sensitivity exercise, not a causal estimate.
    """
    if deltas is None:
        deltas = [-2.0, -1.0, 0.0, 1.0, 2.0]

    rows = []
    base = last_row.copy()
    temp_col = "temperature_2m"
    if temp_col not in base.columns:
        raise ValueError(f"Column '{temp_col}' required for temperature scenarios.")

    for d in deltas:
        row = base.copy()
        row[temp_col] = row[temp_col] + d
        # model is expected to expose predict / predict_quantiles interface
        if hasattr(model, "predict_quantiles"):
            preds = model.predict_quantiles(row)
            p10, p50, p90 = preds["p10"].iloc[0], preds["p50"].iloc[0], preds["p90"].iloc[0]
        else:
            p50 = float(model.predict(row)[0])
            p10 = p50
            p90 = p50
        rows.append({"delta_c": d, "p10": p10, "p50": p50, "p90": p90})
    return pd.DataFrame(rows)
