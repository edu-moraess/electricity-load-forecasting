"""
LightGBM forecasting model, including quantile regression for the
P10/P50/P90 probabilistic bands (see src/forecasting/forecast.py).

Requires `lightgbm` (see requirements.txt). Import is deferred so the
rest of the codebase remains importable without it installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.utils.config import QUANTILES, RANDOM_SEED
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "verbosity": -1,
}


@dataclass
class LightGBMQuantileModel:
    """One LightGBM regressor per requested quantile, trained on the same
    tabular feature matrix. Point forecast is the P50 model's output.
    """

    quantiles: dict[str, float] = field(default_factory=lambda: dict(QUANTILES))
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    feature_names_: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._models: dict[str, object] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMQuantileModel":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "lightgbm is required for LightGBMQuantileModel. "
                "Install it with `pip install lightgbm` (see requirements.txt)."
            ) from exc

        self.feature_names_ = list(X.columns)
        for label, alpha in self.quantiles.items():
            model = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **self.params)
            model.fit(X, y)
            self._models[label] = model
            logger.info("Fitted LightGBM quantile model for %s (alpha=%.2f)", label, alpha)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._models:
            raise ValueError("Model must be fit before predicting")
        X = X[self.feature_names_]
        preds = {label: model.predict(X) for label, model in self._models.items()}
        df = pd.DataFrame(preds)
        # Enforce monotonicity across quantiles row-wise: independently
        # trained quantile models can occasionally cross (P10 > P50).
        # Sorting each row's values back into rank order is the standard,
        # honest fix -- it does not change the point (P50) forecast in the
        # vast majority of rows and only nudges the tails when they cross.
        sorted_cols = sorted(self.quantiles, key=lambda k: self.quantiles[k])
        df[sorted_cols] = np.sort(df[sorted_cols].values, axis=1)
        return df

    def predict_point(self, X: pd.DataFrame) -> np.ndarray:
        """Convenience: P50 only, for use in the model tournament where a
        single point-forecast column is compared against other models."""
        preds = self.predict(X)
        return preds["P50"].values

    def feature_importance(self, quantile_label: str = "P50") -> pd.Series:
        if quantile_label not in self._models:
            raise ValueError(f"No fitted model for quantile '{quantile_label}'")
        model = self._models[quantile_label]
        importances = pd.Series(model.feature_importances_, index=self.feature_names_)
        return importances.sort_values(ascending=False)
