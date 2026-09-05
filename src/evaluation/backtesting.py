"""
Walk-forward (expanding-window) backtesting.

Never uses random/shuffled train-test splits. Each fold trains on all data
up to a cutoff point and tests on the following window, then the cutoff
advances -- so a model is always evaluated only on data strictly after
what it was trained on.

    Fold 1: TRAIN [........]      TEST [..]
    Fold 2: TRAIN [...........]        TEST [..]
    Fold 3: TRAIN [..............]           TEST [..]
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils.config import (
    BACKTEST_MIN_TRAIN_DAYS,
    BACKTEST_N_FOLDS,
    BACKTEST_TEST_WINDOW_DAYS,
    BASE_FREQUENCY_MINUTES,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _periods_per_day() -> int:
    return round(24 * 60 / BASE_FREQUENCY_MINUTES)


@dataclass
class Fold:
    fold_index: int
    train_end: int  # exclusive index into the sorted frame
    test_start: int
    test_end: int  # exclusive


def make_walk_forward_folds(
    n_rows: int,
    min_train_days: int = BACKTEST_MIN_TRAIN_DAYS,
    test_window_days: int = BACKTEST_TEST_WINDOW_DAYS,
    n_folds: int = BACKTEST_N_FOLDS,
) -> list[Fold]:
    """Build expanding-window folds over a chronologically sorted series.

    Raises ValueError if there is not enough data for even one fold, rather
    than silently returning zero folds (which would make a tournament look
    like it ran when it did not).
    """
    periods_per_day = _periods_per_day()
    min_train_periods = min_train_days * periods_per_day
    test_window_periods = test_window_days * periods_per_day

    max_possible_folds = (n_rows - min_train_periods) // test_window_periods
    if max_possible_folds < 1:
        raise ValueError(
            f"Not enough data for walk-forward backtesting: {n_rows} rows available, "
            f"need at least {min_train_periods + test_window_periods} "
            f"({min_train_days} train days + {test_window_days} test days at "
            f"{periods_per_day} periods/day)."
        )

    n_folds_actual = min(n_folds, max_possible_folds)
    folds = []
    for i in range(n_folds_actual):
        train_end = min_train_periods + i * test_window_periods
        test_start = train_end
        test_end = min(test_start + test_window_periods, n_rows)
        folds.append(Fold(fold_index=i, train_end=train_end, test_start=test_start, test_end=test_end))

    logger.info(
        "Built %d walk-forward folds (requested %d, max possible %d)",
        len(folds),
        n_folds,
        max_possible_folds,
    )
    return folds


def iter_fold_frames(df: pd.DataFrame, folds: list[Fold]):
    """Yield (train_df, test_df) pairs for each fold, in chronological order."""
    for fold in folds:
        train_df = df.iloc[: fold.train_end]
        test_df = df.iloc[fold.test_start : fold.test_end]
        yield fold, train_df, test_df
