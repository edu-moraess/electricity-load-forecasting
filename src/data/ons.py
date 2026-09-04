"""
Client for the ONS (Operador Nacional do Sistema Elétrico) open data API.

Data source: https://dados.ons.org.br/dataset/carga-energia-verificada
Confirmed live request pattern (verified 2026-09):

    https://apicarga.ons.org.br/prd/cargaverificada
        ?dat_inicio=YYYY-MM-DD&dat_fim=YYYY-MM-DD&cod_areacarga=AREA

ONS does not publish a stable machine-readable schema for the JSON field
names of this endpoint. This client therefore detects the timestamp and
load-value columns by pattern from the live response and raises a clear
OnsDataError (including the raw column list) if it cannot identify them.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date

import pandas as pd
import requests

from src.utils.config import (
    ONS_CARGA_VERIFICADA_URL,
    ONS_SUBSYSTEMS,
    REQUEST_MAX_RETRIES,
    REQUEST_RETRY_BACKOFF_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Candidate column name patterns (lowercased) used for schema auto-detection.
_TIMESTAMP_CANDIDATES = [
    "din_instante", "dat_instante", "timestamp", "data", "datetime", "hora", "instante",
]
_LOAD_CANDIDATES = [
    "val_cargaverificada", "val_carga", "carga", "load", "valor", "mw", "potencia",
]


class OnsDataError(RuntimeError):
    """Raised when the ONS API is unreachable or the response schema is unrecognizable."""


@dataclass
class OnsLoadSeries:
    subsystem: str
    start_date: date
    end_date: date
    frame: pd.DataFrame  # columns: timestamp, load_mw


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
                "ONS request failed (attempt %d/%d): %s",
                attempt,
                REQUEST_MAX_RETRIES,
                exc,
            )
            if attempt < REQUEST_MAX_RETRIES:
                time.sleep(REQUEST_RETRY_BACKOFF_SECONDS * attempt)
    raise OnsDataError(
        f"Failed to reach ONS API at {url} after {REQUEST_MAX_RETRIES} attempts"
    ) from last_exc


def _detect_columns(raw: list[dict]) -> tuple[str, str]:
    if not raw:
        raise OnsDataError("ONS response is an empty list")
    sample = raw[0]
    keys = list(sample.keys())
    lower_map = {k.lower(): k for k in keys}

    ts_col = None
    for cand in _TIMESTAMP_CANDIDATES:
        if cand in lower_map:
            ts_col = lower_map[cand]
            break
    load_col = None
    for cand in _LOAD_CANDIDATES:
        if cand in lower_map:
            load_col = lower_map[cand]
            break

    if ts_col is None or load_col is None:
        raise OnsDataError(
            f"Could not detect timestamp/load columns in ONS response. "
            f"Raw columns: {keys}"
        )
    logger.info("ONS response schema detected: timestamp=%r, load=%r", ts_col, load_col)
    return ts_col, load_col


def fetch_load_series(
    subsystem: str,
    start_date: date,
    end_date: date,
) -> OnsLoadSeries:
    if subsystem not in ONS_SUBSYSTEMS:
        raise OnsDataError(f"Unknown subsystem {subsystem!r}; expected one of {list(ONS_SUBSYSTEMS)}")

    params = {
        "dat_inicio": start_date.isoformat(),
        "dat_fim": end_date.isoformat(),
        "cod_areacarga": subsystem,
    }
    response = _request_with_retries(ONS_CARGA_VERIFICADA_URL, params)
    payload = response.json()

    if isinstance(payload, dict):
        # Some ONS endpoints wrap the list
        for key in ("data", "dados", "items", "result"):
            if key in payload and isinstance(payload[key], list):
                payload = payload[key]
                break

    if not isinstance(payload, list):
        raise OnsDataError(f"Unexpected ONS response type: {type(payload)}")

    if not payload:
        raise OnsDataError("ONS returned an empty list for the requested window")

    ts_col, load_col = _detect_columns(payload)
    frame = pd.DataFrame(payload)
    frame = frame[[ts_col, load_col]].rename(columns={ts_col: "timestamp", load_col: "load_mw"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["load_mw"] = pd.to_numeric(frame["load_mw"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "load_mw"]).sort_values("timestamp").reset_index(drop=True)

    if frame.empty:
        raise OnsDataError("ONS response contained no usable timestamp/load rows after parsing")

    return OnsLoadSeries(subsystem=subsystem, start_date=start_date, end_date=end_date, frame=frame)
