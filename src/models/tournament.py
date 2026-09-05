"""
Model tournament: evaluate Seasonal Naive, Moving Average, Exponential
Smoothing, and LightGBM under the same walk-forward backtesting
methodology, then rank them by average test-set WAPE (chosen as the
primary ranking metric because it is scale-stable and robust to the
near-zero-denominator issues that MAPE/sMAPE can have).

If a baseline beats LightGBM, the baseline wins -- there is no bias
toward the more complex model anywhere in this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.evaluation.backtesting import make_walk_forward_folds, iter_fold_frames
from src.evaluation.metrics import all_metrics
from src.features.engineering import build_feature_matrix, get_feature_columns
from src.models.baseline import MovingAverageModel, SeasonalNaiveModel
from src.models.statistical import ExponentialSmoothingModel
from src.models.lightgbm_model import LightGBMQuantileModel
from src.utils.logging import get_logger

logger = get_logger(__name__)

PRIMARY_METRIC = "WAPE"


@dataclass
class TournamentResult:
    per_fold_metrics: pd.DataFrame  # rows: (model, fold), cols: metrics
    summary: pd.DataFrame  # rows: model, cols: mean of each metric
    best_model_name: str
    errors: dict[str, list[str]] = field(default_factory=dict)


def _fit_predict_baseline(model_cls, train_df: pd.DataFrame, horizon: int, **kwargs) -> np.ndarray:
    model = model_cls(**kwargs).fit(train_df["load_mw"])
    return model.predict(horizon)


def _fit_predict_lightgbm(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    full = pd.concat([train_df, test_df], ignore_index=True)
    featured = build_feature_matrix(full)
    feature_cols = get_feature_columns(featured)

    # Re-split by timestamp so that lag/rolling features for the test period
    # were computed using only data available up to (and including) that
    # point in the *original* series -- consistent with a real deployment
    # where you always have the true history up to "now".
    cutoff_ts = train_df["timestamp"].max()
    train_feat = featured[featured["timestamp"] <= cutoff_ts]
    test_feat = featured[featured["timestamp"] > cutoff_ts]

    if train_feat.empty or test_feat.empty:
        raise ValueError("Feature engineering left an empty train or test split for this fold")

    model = LightGBMQuantileModel()
    model.fit(train_feat[feature_cols], train_feat["load_mw"])
    preds = model.predict_point(test_feat[feature_cols])
    # Align length: rows dropped during feature engineering (leading NaNs)
    # only ever come from the train side, so test length should match, but
    # guard defensively anyway.
    return preds, test_feat["load_mw"].values


def run_tournament(df: pd.DataFrame) -> TournamentResult:
    folds = make_walk_forward_folds(len(df))
    records = []
    errors: dict[str, list[str]] = {}

    for fold, train_df, test_df in iter_fold_frames(df, folds):
        horizon = len(test_df)
        if horizon == 0:
            continue
        y_test = test_df["load_mw"].values

        # Seasonal Naive
        try:
            preds = _fit_predict_baseline(SeasonalNaiveModel, train_df, horizon)
            records.append({"model": "Seasonal Naive", "fold": fold.fold_index, **all_metrics(y_test, preds)})
        except Exception as exc:  # noqa: BLE001 - report, do not crash the tournament
            errors.setdefault("Seasonal Naive", []).append(f"fold {fold.fold_index}: {exc}")
            logger.error("Seasonal Naive failed on fold %d: %s", fold.fold_index, exc)

        # Moving Average
        try:
            preds = _fit_predict_baseline(MovingAverageModel, train_df, horizon)
            records.append({"model": "Moving Average", "fold": fold.fold_index, **all_metrics(y_test, preds)})
        except Exception as exc:  # noqa: BLE001
            errors.setdefault("Moving Average", []).append(f"fold {fold.fold_index}: {exc}")
            logger.error("Moving Average failed on fold %d: %s", fold.fold_index, exc)

        # Exponential Smoothing
        try:
            preds = _fit_predict_baseline(ExponentialSmoothingModel, train_df, horizon)
            records.append({"model": "Exponential Smoothing", "fold": fold.fold_index, **all_metrics(y_test, preds)})
        except Exception as exc:  # noqa: BLE001
            errors.setdefault("Exponential Smoothing", []).append(f"fold {fold.fold_index}: {exc}")
            logger.error("Exponential Smoothing failed on fold %d: %s", fold.fold_index, exc)

        # LightGBM
        try:
            preds, y_test_lgbm = _fit_predict_lightgbm(train_df, test_df)
            records.append({"model": "LightGBM", "fold": fold.fold_index, **all_metrics(y_test_lgbm, preds)})
        except Exception as exc:  # noqa: BLE001
            errors.setdefault("LightGBM", []).append(f"fold {fold.fold_index}: {exc}")
            logger.error("LightGBM failed on fold %d: %s", fold.fold_index, exc)

    if not records:
        raise RuntimeError(
            "Tournament produced no results at all -- every model failed on every fold. "
            f"Errors: {errors}"
        )

    per_fold = pd.DataFrame(records)
    summary = per_fold.groupby("model")[["MAE", "RMSE", "MAPE", "sMAPE", "WAPE"]].mean()
    summary = summary.sort_values(PRIMARY_METRIC)

    best_model_name = summary.index[0]
    logger.info("Tournament complete. Best model by mean %s: %s", PRIMARY_METRIC, best_model_name)
    if errors:
        logger.warning("Some models had failures during the tournament: %s", errors)

    return TournamentResult(
        per_fold_metrics=per_fold,
        summary=summary.reset_index(),
        best_model_name=best_model_name,
        errors=errors,
    )
