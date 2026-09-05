"""
Multi-horizon forecasting.

Horizons supported: 15min, 1h, 24h, 7d (see src.utils.config.HORIZONS_MINUTES).

Strategy: recursive multi-step forecasting with LightGBM when it is the
tournament winner (each step's prediction feeds the lag features of the
next step), or the baseline/statistical model's native multi-step
`predict(horizon)` otherwise. Recursive forecasting is used rather than
training a separate direct model per horizon because the horizons share
the same underlying feature set and recomputing lag/rolling features
after each step is the standard, well-understood way to extend a
tabular-ML model beyond one step without multiplying model count --
"não force uma arquitetura complexa" per the project brief.

Probabilistic bands (P10/P50/P90):
  - If the winning model is LightGBMQuantileModel, its own quantile
    predictions are used directly (methodologically correct quantile
    regression).
  - Otherwise (a baseline or Exponential Smoothing won the tournament),
    bands are derived from the empirical distribution of that model's
    backtest residuals, added around its point forecast. This is a
    residual-based approximation and is labeled as such everywhere it is
    surfaced -- never presented as true quantile regression when it isn't.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.engineering import build_feature_matrix, get_feature_columns
from src.models.baseline import MovingAverageModel, SeasonalNaiveModel
from src.models.lightgbm_model import LightGBMQuantileModel
from src.models.statistical import ExponentialSmoothingModel
from src.utils.config import BASE_FREQUENCY_MINUTES, HORIZONS_MINUTES, QUANTILES
from src.utils.logging import get_logger

logger = get_logger(__name__)


def horizon_to_periods(horizon_label: str) -> int:
    if horizon_label not in HORIZONS_MINUTES:
        raise ValueError(f"Unknown horizon '{horizon_label}'. Valid: {list(HORIZONS_MINUTES)}")
    minutes = HORIZONS_MINUTES[horizon_label]
    return max(round(minutes / BASE_FREQUENCY_MINUTES), 1)


@dataclass
class ForecastResult:
    horizon_label: str
    timestamps: pd.DatetimeIndex
    p50: np.ndarray
    p10: np.ndarray | None
    p90: np.ndarray | None
    band_method: str  # "quantile_regression" or "residual_bootstrap"
    model_name: str


def _future_timestamps(last_timestamp: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    freq = pd.Timedelta(minutes=BASE_FREQUENCY_MINUTES)
    return pd.date_range(last_timestamp + freq, periods=periods, freq=freq)


def _recursive_lightgbm_forecast(
    model: LightGBMQuantileModel, history: pd.DataFrame, periods: int
) -> pd.DataFrame:
    """Recursively forecast `periods` steps ahead, feeding each prediction's
    P50 back into the load history so the next step's lag features are
    computable. Returns a DataFrame indexed by step with P10/P50/P90."""
    working = history.copy()
    rows = []

    for _ in range(periods):
        featured = build_feature_matrix(working, drop_incomplete=False)
        last_row = featured.iloc[[-1]]
        feature_cols = [c for c in model.feature_names_ if c in last_row.columns]
        missing = [c for c in model.feature_names_ if c not in last_row.columns]
        if missing:
            raise ValueError(f"Missing features needed for recursive forecast: {missing}")

        preds = model.predict(last_row[feature_cols])
        rows.append(preds.iloc[0].to_dict())

        next_ts = working["timestamp"].iloc[-1] + pd.Timedelta(minutes=BASE_FREQUENCY_MINUTES)
        new_row = {c: np.nan for c in working.columns}
        new_row["timestamp"] = next_ts
        new_row["load_mw"] = preds.iloc[0]["P50"]
        # Carry forward the last known weather reading as a placeholder;
        # in production this step should be replaced with Open-Meteo's own
        # forecast for `next_ts` (see src/data/weather.fetch_forecast_weather).
        for col in working.columns:
            if col not in ("timestamp", "load_mw") and col in working.columns:
                new_row[col] = working[col].iloc[-1]
        working = pd.concat([working, pd.DataFrame([new_row])], ignore_index=True)

    return pd.DataFrame(rows)


def generate_forecast(
    history: pd.DataFrame,
    horizon_label: str,
    winning_model_name: str,
    residuals: np.ndarray | None = None,
    fitted_lightgbm: LightGBMQuantileModel | None = None,
) -> ForecastResult:
    """Produce a forecast for the given horizon using whichever model won
    the tournament. `residuals` (from backtesting) are required for
    non-LightGBM models to build the residual-bootstrap bands.
    """
    periods = horizon_to_periods(horizon_label)
    last_ts = history["timestamp"].max()
    timestamps = _future_timestamps(last_ts, periods)

    if winning_model_name == "LightGBM":
        if fitted_lightgbm is None:
            raise ValueError("fitted_lightgbm model instance is required when LightGBM won the tournament")
        preds_df = _recursive_lightgbm_forecast(fitted_lightgbm, history, periods)
        return ForecastResult(
            horizon_label=horizon_label,
            timestamps=timestamps,
            p50=preds_df["P50"].values,
            p10=preds_df["P10"].values,
            p90=preds_df["P90"].values,
            band_method="quantile_regression",
            model_name=winning_model_name,
        )

    y = history["load_mw"]
    if winning_model_name == "Seasonal Naive":
        point = SeasonalNaiveModel().fit(y).predict(periods)
    elif winning_model_name == "Moving Average":
        point = MovingAverageModel().fit(y).predict(periods)
    elif winning_model_name == "Exponential Smoothing":
        point = ExponentialSmoothingModel().fit(y).predict(periods)
    else:
        raise ValueError(f"Unknown winning model '{winning_model_name}'")

    p10 = p90 = None
    band_method = "none"
    if residuals is not None and len(residuals) > 10:
        q10, q90 = np.quantile(residuals, [QUANTILES["P10"], QUANTILES["P90"]])
        p10 = point + q10
        p90 = point + q90
        band_method = "residual_bootstrap"
    else:
        logger.warning(
            "No usable residual history for %s; returning point forecast only (no bands)",
            winning_model_name,
        )

    return ForecastResult(
        horizon_label=horizon_label,
        timestamps=timestamps,
        p50=point,
        p10=p10,
        p90=p90,
        band_method=band_method,
        model_name=winning_model_name,
    )
