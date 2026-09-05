"""Shared pytest fixtures. All fixtures use deterministic, locally generated
data -- no test in this suite makes a network call."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_series() -> pd.DataFrame:
    df = pd.read_csv(FIXTURES_DIR / "sample_merged_series.csv", parse_dates=["timestamp"])
    return df


@pytest.fixture
def sample_load_series(sample_series) -> pd.DataFrame:
    return sample_series[["timestamp", "load_mw"]].copy()
