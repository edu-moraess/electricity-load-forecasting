import pandas as pd
import pytest

from src.evaluation.backtesting import iter_fold_frames, make_walk_forward_folds


def test_folds_never_leak_future_into_train():
    n = 5000
    df = pd.DataFrame({"x": range(n)})
    folds = make_walk_forward_folds(n)
    for fold, train, test in iter_fold_frames(df, folds):
        assert train["x"].max() < test["x"].min()


def test_folds_are_chronologically_expanding():
    n = 5000
    folds = make_walk_forward_folds(n)
    train_ends = [f.train_end for f in folds]
    assert train_ends == sorted(train_ends)
    assert len(set(train_ends)) == len(train_ends)


def test_raises_when_not_enough_data():
    with pytest.raises(ValueError):
        make_walk_forward_folds(50)
