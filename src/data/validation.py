"""
Data validation for load and weather series.

This module never fills gaps with invented values. It flags problems and
returns a structured report so the caller (and the Streamlit app) can
display exactly what was found, per the project's data-integrity rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationReport:
    n_rows: int
    n_duplicates_removed: int
    n_nulls_found: int
    n_impossible_values: int
    gaps: list[dict] = field(default_factory=list)
    timezone_note: str = ""
    is_sorted: bool = True
    issues: list[str] = field(default_factory=list)

    @property
    def has_critical_issues(self) -> bool:
        return len(self.issues) > 0

    def summary(self) -> str:
        lines = [
            f"Rows: {self.n_rows}",
            f"Duplicates removed: {self.n_duplicates_removed}",
            f"Null values found: {self.n_nulls_found}",
            f"Impossible values found: {self.n_impossible_values}",
            f"Gaps detected: {len(self.gaps)}",
        ]
        if self.issues:
            lines.append("Issues: " + "; ".join(self.issues))
        return " | ".join(lines)


def validate_load_series(
    frame: pd.DataFrame,
    expected_freq_minutes: int,
    value_col: str = "load_mw",
    timestamp_col: str = "timestamp",
    min_plausible_mw: float = 0.0,
    max_plausible_mw: float = 200_000.0,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate timestamps, ordering, duplicates, gaps, nulls, and value ranges.

    Returns the cleaned (but not gap-filled) frame plus a report describing
    everything found.
    """
    issues: list[str] = []
    df = frame.copy()

    if timestamp_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Expected columns '{timestamp_col}' and '{value_col}' in frame")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    is_sorted = df[timestamp_col].is_monotonic_increasing
    if not is_sorted:
        issues.append("Timestamps were not sorted; sorting now")
        df = df.sort_values(timestamp_col)

    n_before = len(df)
    df = df.drop_duplicates(subset=timestamp_col)
    n_duplicates = n_before - len(df)
    if n_duplicates:
        issues.append(f"{n_duplicates} duplicate timestamps removed")

    n_nulls = int(df[value_col].isna().sum())
    if n_nulls:
        issues.append(f"{n_nulls} null load values present")

    impossible_mask = (df[value_col] < min_plausible_mw) | (df[value_col] > max_plausible_mw)
    n_impossible = int(impossible_mask.sum())
    if n_impossible:
        issues.append(f"{n_impossible} values outside plausible range [{min_plausible_mw}, {max_plausible_mw}] MW")
        df.loc[impossible_mask, value_col] = np.nan

    # Gap detection based on expected frequency.
    gaps: list[dict] = []
    if len(df) > 1:
        deltas = df[timestamp_col].diff().dropna()
        expected = pd.Timedelta(minutes=expected_freq_minutes)
        gap_mask = deltas > expected * 1.5
        gap_indices = deltas[gap_mask].index
        for idx in gap_indices:
            pos = df.index.get_loc(idx)
            prev_ts = df.iloc[pos - 1][timestamp_col]
            curr_ts = df.iloc[pos][timestamp_col]
            gaps.append(
                {
                    "start": str(prev_ts),
                    "end": str(curr_ts),
                    "missing_minutes": (curr_ts - prev_ts).total_seconds() / 60 - expected_freq_minutes,
                }
            )
    if gaps:
        issues.append(f"{len(gaps)} time gaps detected (not filled with invented values)")

    tz_note = "naive (assumed America/Sao_Paulo, as published by ONS)"
    if df[timestamp_col].dt.tz is not None:
        tz_note = str(df[timestamp_col].dt.tz)

    report = ValidationReport(
        n_rows=len(df),
        n_duplicates_removed=n_duplicates,
        n_nulls_found=n_nulls,
        n_impossible_values=n_impossible,
        gaps=gaps,
        timezone_note=tz_note,
        is_sorted=is_sorted,
        issues=issues,
    )

    for issue in issues:
        logger.warning("Validation issue: %s", issue)
    logger.info("Validation complete: %s", report.summary())

    return df.reset_index(drop=True), report


def validate_weather_series(frame: pd.DataFrame, timestamp_col: str = "timestamp") -> tuple[pd.DataFrame, ValidationReport]:
    """Lightweight structural validation for weather data (no value-range checks
    beyond null counts, since plausible ranges vary a lot by variable)."""
    issues: list[str] = []
    df = frame.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    is_sorted = df[timestamp_col].is_monotonic_increasing
    if not is_sorted:
        df = df.sort_values(timestamp_col)
        issues.append("Timestamps were not sorted; sorting now")

    n_before = len(df)
    df = df.drop_duplicates(subset=timestamp_col)
    n_duplicates = n_before - len(df)

    numeric_cols = [c for c in df.columns if c != timestamp_col]
    n_nulls = int(df[numeric_cols].isna().sum().sum()) if numeric_cols else 0
    if n_nulls:
        issues.append(f"{n_nulls} null weather values present across {len(numeric_cols)} variables")

    report = ValidationReport(
        n_rows=len(df),
        n_duplicates_removed=n_duplicates,
        n_nulls_found=n_nulls,
        n_impossible_values=0,
        gaps=[],
        timezone_note="naive (America/Sao_Paulo)",
        is_sorted=is_sorted,
        issues=issues,
    )
    logger.info("Weather validation complete: %s", report.summary())
    return df.reset_index(drop=True), report
