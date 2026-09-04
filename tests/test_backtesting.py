"""Tests for walk-forward backtesting fold construction."""
from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.backtesting import build_walk_forward_folds


def test_folds_never_leak_future_into_train(sample_series):
    folds = build_walk_forward_folds(sample_series, min_train_days=30, test_window_days=7, n_folds=3)
    for train_idx, test_idx in folds:
        assert sample_series.loc[train_idx, "timestamp"].max() < sample_series.loc[test_idx, "timestamp"].min()


def test_folds_are_chronologically_expanding(sample_series):
    folds = build_walk_forward_folds(sample_series, min_train_days=30, test_window_days=7, n_folds=3)
    train_sizes = [len(tr) for tr, _ in folds]
    assert train_sizes == sorted(train_sizes)


def test_raises_when_not_enough_data(sample_series):
    short = sample_series.head(10)
    with pytest.raises(ValueError):
        build_walk_forward_folds(short, min_train_days=60, test_window_days=7, n_folds=5)
