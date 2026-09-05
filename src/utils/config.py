"""
Central configuration for the Electricity Load Forecasting pipeline.

All configurable values live here so the rest of the codebase never hardcodes
URLs, area codes, or file paths. Values can be overridden via environment
variables (see .env.example).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
LOG_DIR = PROJECT_ROOT / "logs"

for _dir in (RAW_DIR, CACHE_DIR, LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# ONS - Operador Nacional do Sistema Elétrico (dados.ons.org.br)
# ---------------------------------------------------------------------------
# Confirmed live endpoint (documented at
# https://dados.ons.org.br/dataset/carga-energia-verificada) as of 2026-09:
#   https://apicarga.ons.org.br/prd/cargaverificada?dat_inicio=YYYY-MM-DD&dat_fim=YYYY-MM-DD&cod_areacarga=AREA
ONS_CARGA_VERIFICADA_URL = os.getenv(
    "ONS_CARGA_VERIFICADA_URL", "https://apicarga.ons.org.br/prd/cargaverificada"
)
ONS_CARGA_PROGRAMADA_URL = os.getenv(
    "ONS_CARGA_PROGRAMADA_URL", "https://apicarga.ons.org.br/prd/cargaprogramada"
)
# CKAN portal (used only to resolve/verify resources, not for data itself).
ONS_CKAN_BASE_URL = os.getenv("ONS_CKAN_BASE_URL", "https://dados.ons.org.br")

# Official ONS subsystem ("área de carga") codes for the SIN.
ONS_SUBSYSTEMS = {
    "N": "Norte",
    "NE": "Nordeste",
    "S": "Sul",
    "SE": "Sudeste/Centro-Oeste",
}
DEFAULT_SUBSYSTEM = os.getenv("ONS_SUBSYSTEM", "SE")

# ---------------------------------------------------------------------------
# Open-Meteo (open-meteo.com) - no API key required
# ---------------------------------------------------------------------------
OPEN_METEO_FORECAST_URL = os.getenv(
    "OPEN_METEO_FORECAST_URL", "https://api.open-meteo.com/v1/forecast"
)
OPEN_METEO_ARCHIVE_URL = os.getenv(
    "OPEN_METEO_ARCHIVE_URL", "https://archive-api.open-meteo.com/v1/archive"
)

HOURLY_WEATHER_VARIABLES = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
]

# Reference coordinates per subsystem (approximate load-weighted city used as
# a proxy for regional weather -- documented as an approximation, not a
# precise load-weighted centroid).
SUBSYSTEM_COORDINATES = {
    "N": {"name": "Belém", "latitude": -1.4558, "longitude": -48.4902},
    "NE": {"name": "Recife", "latitude": -8.0476, "longitude": -34.8770},
    "S": {"name": "Curitiba", "latitude": -25.4284, "longitude": -49.2733},
    "SE": {"name": "São Paulo", "latitude": -23.5505, "longitude": -46.6333},
}

# ---------------------------------------------------------------------------
# HTTP behaviour
# ---------------------------------------------------------------------------
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
REQUEST_MAX_RETRIES = int(os.getenv("REQUEST_MAX_RETRIES", "3"))
REQUEST_RETRY_BACKOFF_SECONDS = float(os.getenv("REQUEST_RETRY_BACKOFF_SECONDS", "2.0"))

# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
CACHE_TTL_MINUTES = int(os.getenv("CACHE_TTL_MINUTES", "30"))

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
HORIZONS_MINUTES = {
    "15min": 15,
    "1h": 60,
    "24h": 60 * 24,
    "7d": 60 * 24 * 7,
}

BASE_FREQUENCY_MINUTES = int(os.getenv("BASE_FREQUENCY_MINUTES", "30"))  # semi-hourly

QUANTILES = {"P10": 0.10, "P50": 0.50, "P90": 0.90}

BACKTEST_MIN_TRAIN_DAYS = int(os.getenv("BACKTEST_MIN_TRAIN_DAYS", "60"))
BACKTEST_TEST_WINDOW_DAYS = int(os.getenv("BACKTEST_TEST_WINDOW_DAYS", "7"))
BACKTEST_N_FOLDS = int(os.getenv("BACKTEST_N_FOLDS", "5"))

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))


@dataclass
class RunConfig:
    """Convenience bundle for a single ingestion/forecast run."""

    subsystem: str = DEFAULT_SUBSYSTEM
    start_date: str | None = None
    end_date: str | None = None
    horizons: tuple[str, ...] = field(default_factory=lambda: tuple(HORIZONS_MINUTES.keys()))

    def __post_init__(self) -> None:
        if self.subsystem not in ONS_SUBSYSTEMS:
            raise ValueError(
                f"Unknown ONS subsystem '{self.subsystem}'. "
                f"Valid options: {sorted(ONS_SUBSYSTEMS)}"
            )
