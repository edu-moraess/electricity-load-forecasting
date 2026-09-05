"""
Centralized logging setup.

Every stage of the pipeline (ingestion, validation, feature engineering,
training, backtesting, forecasting) logs through this module so a single
run produces one coherent, timestamped log file plus console output.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.utils.config import LOG_DIR

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring handlers on first call."""
    global _CONFIGURED
    logger = logging.getLogger(name)

    root = logging.getLogger()
    if not _CONFIGURED:
        root.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

        log_file = Path(LOG_DIR) / "pipeline.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

        _CONFIGURED = True

    return logger
