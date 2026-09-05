"""Ingestion tests use monkeypatched HTTP clients -- no real network calls.
This mirrors the project requirement that unit tests never depend on
external APIs."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.data import ons


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_fetch_load_series_parses_recognized_schema(monkeypatch):
    fake_payload = [
        {"din_instante": "2024-01-01T00:00:00", "val_cargaenergiahomwmed": 1000.5, "id_subsistema": "SE"},
        {"din_instante": "2024-01-01T00:30:00", "val_cargaenergiahomwmed": 1010.2, "id_subsistema": "SE"},
    ]

    def fake_get(url, params, timeout):
        return _FakeResponse(fake_payload)

    monkeypatch.setattr(ons.requests, "get", fake_get)
    result = ons.fetch_load_series("SE", date(2024, 1, 1), date(2024, 1, 1))
    assert len(result.frame) == 2
    assert list(result.frame.columns) == ["timestamp", "load_mw"]
    assert result.frame["load_mw"].iloc[0] == 1000.5


def test_fetch_load_series_raises_on_empty_response(monkeypatch):
    def fake_get(url, params, timeout):
        return _FakeResponse([])

    monkeypatch.setattr(ons.requests, "get", fake_get)
    with pytest.raises(ons.OnsDataError):
        ons.fetch_load_series("SE", date(2024, 1, 1), date(2024, 1, 1))


def test_fetch_load_series_raises_on_unrecognized_schema(monkeypatch):
    def fake_get(url, params, timeout):
        return _FakeResponse([{"totally_unexpected_field": 123}])

    monkeypatch.setattr(ons.requests, "get", fake_get)
    with pytest.raises(ons.OnsDataError):
        ons.fetch_load_series("SE", date(2024, 1, 1), date(2024, 1, 1))


def test_fetch_load_series_rejects_unknown_subsystem():
    with pytest.raises(ValueError):
        ons.fetch_load_series("XX", date(2024, 1, 1), date(2024, 1, 1))
