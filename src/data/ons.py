"""
Client for the ONS (Operador Nacional do Sistema Elétrico) open data API.

Data source: https://dados.ons.org.br/dataset/carga-energia-verificada
Confirmed live request pattern (verified 2026-09):

    https://apicarga.ons.org.br/prd/cargaverificada
        ?dat_inicio=YYYY-MM-DD&dat_fim=YYYY-MM-DD&cod_areacarga=AREA

Where AREA is one of the SIN subsystem area codes: SECO (Sudeste/Centro-Oeste),
N (Norte), NE (Nordeste), S (Sul). The application keeps "SE" as its internal
label and maps it to the official ONS API code "SECO".

This module intentionally does NOT hardcode the exact JSON field names returned
by the API: ONS has changed field naming between dataset versions before, and
guessing wrong would silently produce garbage data. Instead, `_normalize_records`
inspects the real response and matches columns by pattern, logging exactly what
it found. If it cannot confidently identify a timestamp column and a load value
column, it raises `OnsDataError` with the raw keys included, rather than
fabricating data.

No synthetic fallback is used anywhere in this module. If the API is unreachable
or returns no data for the requested window, that is surfaced to the caller as
an exception -- never silently replaced with mock values.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
import requests

from src.utils.config import (
    ONS_API_AREA_CODES,
    ONS_CARGA_VERIFICADA_URL,
    ONS_SUBSYSTEMS,
    REQUEST_MAX_RETRIES,
    REQUEST_RETRY_BACKOFF_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Candidate substrings for auto-detecting the relevant columns in whatever
# schema the API returns. Based on ONS's documented naming conventions
# (din_* for datetime fields, val_* for numeric values).
_TIMESTAMP_CANDIDATES = ("din_instante", "din_referencia", "data", "timestamp", "instante")
_LOAD_VALUE_CANDIDATES = (
    "val_cargaglobalcons",
    "val_cargaenergiahomwmed",
    "val_carga",
    "carga",
    "val_cons",
    "valor",
)
_SUBSYSTEM_CANDIDATES = ("id_subsistema", "nom_subsistema", "cod_areacarga", "subsistema", "area")


class OnsDataError(RuntimeError):
    """Raised when the ONS API is unreachable or returns an unrecognized schema."""


@dataclass
class OnsLoadSeries:
    subsystem: str
    start_date: date
    end_date: date
    frame: pd.DataFrame  # columns: timestamp (UTC-naive, local America/Sao_Paulo), load_mw


def _request_with_retries(url: str, params: dict) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(1, REQUEST_MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:  # network error, timeout, 4xx/5xx
            last_exc = exc
            logger.warning(
                "ONS request failed (attempt %d/%d): %s", attempt, REQUEST_MAX_RETRIES, exc
            )
            if attempt < REQUEST_MAX_RETRIES:
                time.sleep(REQUEST_RETRY_BACKOFF_SECONDS * attempt)
    raise OnsDataError(f"Failed to reach ONS API at {url} after {REQUEST_MAX_RETRIES} attempts") from last_exc


def _find_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        for lower_name, original in lower_map.items():
            if candidate in lower_name:
                return original
    return None


def _normalize_records(records: list[dict], subsystem: str) -> pd.DataFrame:
    if not records:
        raise OnsDataError(
            f"ONS API returned zero records for subsystem '{subsystem}'. "
            "This can mean the requested window has no published data yet, "
            "or the area code is not accepted by the current API version. "
            "No synthetic data will be substituted."
        )

    df = pd.DataFrame.from_records(records)
    columns = list(df.columns)

    ts_col = _find_column(columns, _TIMESTAMP_CANDIDATES)
    load_col = _find_column(columns, _LOAD_VALUE_CANDIDATES)

    if ts_col is None or load_col is None:
        raise OnsDataError(
            "Could not identify timestamp/load columns in the ONS API response. "
            f"Columns returned by the API: {columns}. "
            "Update src/data/ons.py candidate lists to match the current schema "
            "instead of guessing -- inspect one live response first."
        )

    logger.info("ONS response schema detected: timestamp='%s', load='%s'", ts_col, load_col)

    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df[ts_col], errors="coerce"),
            "load_mw": pd.to_numeric(df[load_col], errors="coerce"),
        }
    )
    n_before = len(out)
    out = out.dropna(subset=["timestamp"])
    n_after = len(out)
    if n_after < n_before:
        logger.warning("Dropped %d rows with unparseable timestamps", n_before - n_after)

    out = out.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
    return out


def fetch_load_series(subsystem: str, start_date: date, end_date: date) -> OnsLoadSeries:
    """Fetch verified load (carga verificada) for one SIN subsystem and date range.

    Raises OnsDataError if the API is unreachable or the schema cannot be
    parsed. Never returns fabricated data.
    """
    if subsystem not in ONS_SUBSYSTEMS:
        raise ValueError(f"Unknown subsystem '{subsystem}'. Valid: {sorted(ONS_SUBSYSTEMS)}")

    params = {
        "dat_inicio": start_date.isoformat(),
        "dat_fim": end_date.isoformat(),
        "cod_areacarga": ONS_API_AREA_CODES[subsystem],
    }
    logger.info("Requesting ONS load data: %s", params)
    response = _request_with_retries(ONS_CARGA_VERIFICADA_URL, params)

    try:
        payload = response.json()
    except ValueError as exc:
        raise OnsDataError(f"ONS API returned non-JSON content: {response.text[:200]}") from exc

    # The API may return either a bare list or an envelope like {"data": [...]}.
    if isinstance(payload, dict):
        records = payload.get("data") or payload.get("results") or []
    else:
        records = payload

    frame = _normalize_records(records, subsystem)
    logger.info("Fetched %d ONS observations for subsystem %s", len(frame), subsystem)

    return OnsLoadSeries(subsystem=subsystem, start_date=start_date, end_date=end_date, frame=frame)
