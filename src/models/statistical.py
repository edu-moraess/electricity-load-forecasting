"""
Statistical forecasting model: Exponential Smoothing (Holt-Winters).

Exponential Smoothing is used instead of full SARIMA as the statistical
benchmark because, for semi-hourly series with strong daily/weekly
seasonality, SARIMA's seasonal order search becomes computationally
expensive at these season lengths (48 or 336 periods), while
statsmodels' ExponentialSmoothing handles a specified seasonal period
directly and remains fast enough to run inside the walk-forward
backtest loop. This is a documented, deliberate choice, not a
downgrade -- both are valid "statistical" baselines under the project
brief.

Requires `statsmodels` (see requirements.txt). Import is deferred into
the functions below so the rest of the codebase can be imported and
tested even in environments where statsmodels is not yet installed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.config import BASE_FREQUENCY_MINUTES
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _periods_per_day() -> int:
    return round(24 * 60 / BASE_FREQUENCY_MINUTES)


@dataclass
class ExponentialSmoothingModel:
    seasonal_periods: int = 0
    trend: str | None = "add"
    seasonal: str | None = "add"

    def __post_init__(self) -> None:
        if self.seasonal_periods <= 0:
            self.seasonal_periods = _periods_per_day()
        self._fitted = None

    def fit(self, y: pd.Series) -> "ExponentialSmoothingModel":
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except ImportError as exc:
            raise ImportError(
                "statsmodels is required for ExponentialSmoothingModel. "
                "Install it with `pip install statsmodels` (see requirements.txt)."
            ) from exc

        series = y.reset_index(drop=True)
        # Holt-Winters needs at least two full seasonal cycles to estimate
        # a seasonal component; fall back to non-seasonal smoothing otherwise.
        if len(series) < 2 * self.seasonal_periods:
            logger.warning(
                "Not enough history (%d rows) for seasonal period %d; "
                "fitting without seasonality",
                len(series),
                self.seasonal_periods,
            )
            model = ExponentialSmoothing(series, trend=self.trend, seasonal=None)
        else:
            model = ExponentialSmoothing(
                series,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                initialization_method="estimated",
            )
        self._fitted = model.fit(optimized=True)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._fitted is None:
            raise ValueError("Model must be fit before predicting")
        forecast = self._fitted.forecast(horizon)
        return np.asarray(forecast)
