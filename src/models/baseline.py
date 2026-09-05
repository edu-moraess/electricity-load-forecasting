"""
Baseline forecasting models: Seasonal Naive and Moving Average.

These have no hyperparameters to fit in the ML sense -- "fitting" just
means remembering enough recent history to produce forecasts. They exist
primarily as a floor that SARIMA/LightGBM must beat to justify their
complexity (see src/models/tournament.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.config import BASE_FREQUENCY_MINUTES


def _periods_per_day() -> int:
    return round(24 * 60 / BASE_FREQUENCY_MINUTES)


@dataclass
class SeasonalNaiveModel:
    """Forecast = value observed exactly one seasonal period ago (1 day)."""

    season_periods: int = 0

    def __post_init__(self) -> None:
        if self.season_periods <= 0:
            self.season_periods = _periods_per_day()
        self._history: pd.Series | None = None

    def fit(self, y: pd.Series) -> "SeasonalNaiveModel":
        self._history = y.reset_index(drop=True)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._history is None or len(self._history) < self.season_periods:
            raise ValueError("Not enough history to seasonally forecast this horizon")
        hist = self._history.values
        preds = np.empty(horizon)
        for h in range(horizon):
            # index counting back from the end, wrapping by season length
            idx = len(hist) - self.season_periods + (h % self.season_periods)
            # if the wrapped index runs past the available history, fall back
            # to the most recent same-season observation
            while idx >= len(hist):
                idx -= self.season_periods
            preds[h] = hist[idx]
        return preds


@dataclass
class MovingAverageModel:
    """Forecast = flat average of the last `window` observations."""

    window: int = 0

    def __post_init__(self) -> None:
        if self.window <= 0:
            self.window = _periods_per_day()
        self._last_average: float | None = None

    def fit(self, y: pd.Series) -> "MovingAverageModel":
        if len(y) < self.window:
            self._last_average = float(y.mean())
        else:
            self._last_average = float(y.tail(self.window).mean())
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._last_average is None:
            raise ValueError("Model must be fit before predicting")
        return np.full(horizon, self._last_average)
