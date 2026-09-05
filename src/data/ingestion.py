"""
Ingestion orchestration: pulls ONS load data + Open-Meteo weather data,
validates both, merges them, and caches the result to disk (CSV) with a
timestamp so the Streamlit app does not hit the external APIs on every
rerun.

Cache policy: a cached file is reused if it is younger than
CACHE_TTL_MINUTES (see src.utils.config). Otherwise a fresh ingestion is
attempted. If ingestion fails and a (stale) cache exists, the stale cache is
used and clearly flagged as stale -- never silently treated as fresh.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from src.data.ons import OnsDataError, fetch_load_series
from src.data.validation import ValidationReport, validate_load_series, validate_weather_series
from src.data.weather import WeatherDataError, fetch_historical_weather
from src.utils.config import BASE_FREQUENCY_MINUTES, CACHE_DIR, CACHE_TTL_MINUTES
from src.utils.logging import get_logger

logger = get_logger(__name__)


class IngestionError(RuntimeError):
    """Raised when both live ingestion and cache fallback fail."""


@dataclass
class IngestionResult:
    frame: pd.DataFrame  # merged load + weather, one row per timestamp
    load_report: ValidationReport
    weather_report: ValidationReport
    fetched_at: datetime
    from_cache: bool
    is_stale: bool
    source_errors: list[str]


def _cache_path(subsystem: str) -> Path:
    return Path(CACHE_DIR) / f"merged_{subsystem}.csv"


def _load_cache(subsystem: str) -> tuple[pd.DataFrame | None, datetime | None]:
    path = _cache_path(subsystem)
    if not path.exists():
        return None, None
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    fetched_at = datetime.fromtimestamp(path.stat().st_mtime)
    return frame, fetched_at


def _save_cache(subsystem: str, frame: pd.DataFrame) -> None:
    path = _cache_path(subsystem)
    frame.to_csv(path, index=False)
    logger.info("Cached merged dataset for %s to %s (%d rows)", subsystem, path, len(frame))


def _merge_load_and_weather(load_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Merge semi-hourly load onto hourly weather via as-of (backward) join."""
    load_sorted = load_df.sort_values("timestamp")
    weather_sorted = weather_df.sort_values("timestamp")
    merged = pd.merge_asof(
        load_sorted,
        weather_sorted,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(hours=1),
    )
    return merged


def run_ingestion(
    subsystem: str,
    start_date: date,
    end_date: date,
    force_refresh: bool = False,
) -> IngestionResult:
    cached_frame, cached_at = _load_cache(subsystem)
    cache_is_fresh = (
        cached_at is not None
        and (datetime.now() - cached_at) < timedelta(minutes=CACHE_TTL_MINUTES)
    )

    if cache_is_fresh and not force_refresh:
        logger.info("Using fresh cache for %s (age < %d min)", subsystem, CACHE_TTL_MINUTES)
        # Even cached data is re-validated so the report reflects current content.
        load_report_frame, load_report = validate_load_series(
            cached_frame[["timestamp", "load_mw"]], BASE_FREQUENCY_MINUTES
        )
        weather_cols = [c for c in cached_frame.columns if c not in ("load_mw",)]
        weather_report_frame, weather_report = validate_weather_series(cached_frame[weather_cols])
        return IngestionResult(
            frame=cached_frame,
            load_report=load_report,
            weather_report=weather_report,
            fetched_at=cached_at,
            from_cache=True,
            is_stale=False,
            source_errors=[],
        )

    errors: list[str] = []
    load_df = None
    weather_df = None

    try:
        load_series = fetch_load_series(subsystem, start_date, end_date)
        load_df, load_report = validate_load_series(load_series.frame, BASE_FREQUENCY_MINUTES)
    except OnsDataError as exc:
        errors.append(f"ONS ingestion failed: {exc}")
        logger.error(str(exc))

    try:
        weather_series = fetch_historical_weather(subsystem, start_date, end_date)
        weather_df, weather_report = validate_weather_series(weather_series.frame)
    except WeatherDataError as exc:
        errors.append(f"Open-Meteo ingestion failed: {exc}")
        logger.error(str(exc))

    if load_df is not None and weather_df is not None:
        merged = _merge_load_and_weather(load_df, weather_df)
        _save_cache(subsystem, merged)
        return IngestionResult(
            frame=merged,
            load_report=load_report,
            weather_report=weather_report,
            fetched_at=datetime.now(),
            from_cache=False,
            is_stale=False,
            source_errors=errors,
        )

    # Live ingestion failed (at least partially) -- fall back to stale cache
    # if one exists, but flag it clearly. Never fabricate data.
    if cached_frame is not None:
        logger.warning("Live ingestion failed; falling back to stale cache from %s", cached_at)
        load_report_frame, load_report = validate_load_series(
            cached_frame[["timestamp", "load_mw"]], BASE_FREQUENCY_MINUTES
        )
        weather_cols = [c for c in cached_frame.columns if c not in ("load_mw",)]
        weather_report_frame, weather_report = validate_weather_series(cached_frame[weather_cols])
        return IngestionResult(
            frame=cached_frame,
            load_report=load_report,
            weather_report=weather_report,
            fetched_at=cached_at,
            from_cache=True,
            is_stale=True,
            source_errors=errors,
        )

    raise IngestionError(
        "Live ingestion failed and no cache is available. Errors: " + "; ".join(errors)
    )
