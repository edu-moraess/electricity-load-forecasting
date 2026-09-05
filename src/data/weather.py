"""
Client for the Open-Meteo weather API (https://open-meteo.com/).

Two endpoints are used depending on the date range requested:
  - Archive API (https://archive-api.open-meteo.com/v1/archive) for
    historical dates, used to build training features.
  - Forecast API (https://api.open-meteo.com/v1/forecast) for
    today/near-future dates, used for live scenario/forecast features.
    The forecast endpoint also exposes a `past_days` parameter, which lets
    us pull a short recent window in the same call.

No API key is required for either endpoint. No synthetic weather data is
generated anywhere in this module.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date

import pandas as pd
import requests

from src.utils.config import (
    HOURLY_WEATHER_VARIABLES,
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_FORECAST_URL,
    REQUEST_MAX_RETRIES,
    REQUEST_RETRY_BACKOFF_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    SUBSYSTEM_COORDINATES,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class WeatherDataError(RuntimeError):
    """Raised when the Open-Meteo API is unreachable or returns unexpected data."""


@dataclass
class WeatherSeries:
    subsystem: str
    frame: pd.DataFrame  # columns: timestamp, temperature_2m, apparent_temperature, ...


def _request_with_retries(url: str, params: dict) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(1, REQUEST_MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning(
                "Open-Meteo request failed (attempt %d/%d): %s",
                attempt,
                REQUEST_MAX_RETRIES,
                exc,
            )
            if attempt < REQUEST_MAX_RETRIES:
                time.sleep(REQUEST_RETRY_BACKOFF_SECONDS * attempt)
    raise WeatherDataError(
        f"Failed to reach Open-Meteo API at {url} after {REQUEST_MAX_RETRIES} attempts"
    ) from last_exc


def _parse_hourly_payload(payload: dict) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise WeatherDataError(f"Unexpected Open-Meteo response shape: keys={list(payload.keys())}")

    frame = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"])})
    for var in HOURLY_WEATHER_VARIABLES:
        if var in hourly:
            frame[var] = hourly[var]
        else:
            logger.warning("Open-Meteo response missing expected variable '%s'", var)
    return frame


def fetch_historical_weather(subsystem: str, start_date: date, end_date: date) -> WeatherSeries:
    """Fetch hourly historical weather for a subsystem's reference city."""
    if subsystem not in SUBSYSTEM_COORDINATES:
        raise ValueError(f"No reference coordinates configured for subsystem '{subsystem}'")

    coords = SUBSYSTEM_COORDINATES[subsystem]
    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(HOURLY_WEATHER_VARIABLES),
        "timezone": "America/Sao_Paulo",
    }
    logger.info("Requesting historical weather for %s (%s): %s", subsystem, coords["name"], params)
    response = _request_with_retries(OPEN_METEO_ARCHIVE_URL, params)
    payload = response.json()
    frame = _parse_hourly_payload(payload)
    logger.info("Fetched %d historical weather observations for %s", len(frame), subsystem)
    return WeatherSeries(subsystem=subsystem, frame=frame)


def fetch_forecast_weather(subsystem: str, past_days: int = 3, forecast_days: int = 7) -> WeatherSeries:
    """Fetch recent + forecast hourly weather, used for live forecasting."""
    if subsystem not in SUBSYSTEM_COORDINATES:
        raise ValueError(f"No reference coordinates configured for subsystem '{subsystem}'")

    coords = SUBSYSTEM_COORDINATES[subsystem]
    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "hourly": ",".join(HOURLY_WEATHER_VARIABLES),
        "timezone": "America/Sao_Paulo",
        "past_days": past_days,
        "forecast_days": forecast_days,
    }
    logger.info("Requesting forecast weather for %s (%s): %s", subsystem, coords["name"], params)
    response = _request_with_retries(OPEN_METEO_FORECAST_URL, params)
    payload = response.json()
    frame = _parse_hourly_payload(payload)
    logger.info("Fetched %d forecast weather observations for %s", len(frame), subsystem)
    return WeatherSeries(subsystem=subsystem, frame=frame)
