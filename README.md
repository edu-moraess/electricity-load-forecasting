# Electricity Load Forecasting

Short-term electricity demand forecasting using real operational and weather data.

## 1. Overview

This project forecasts short-term electricity load for the Brazilian National
Interconnected System (SIN) using real, publicly available data: verified load
from ONS (Operador Nacional do Sistema Elétrico) and weather observations from
Open-Meteo. It compares baseline, statistical, and machine-learning forecasters
under a single, leakage-free walk-forward backtesting methodology, and serves
the result through one Streamlit page.

No synthetic, mocked, or fabricated data is used anywhere — if a data source is
unavailable, the app surfaces the error instead of hiding it behind a fallback.

## 2. Problem

Grid operators and market participants need short-horizon load forecasts
(minutes to a week ahead) to plan dispatch, manage reserves, and anticipate
demand peaks. This project frames that as a supervised time-series problem:
predict `load_mw` at future timestamps from its own history and from weather
covariates, with an explicit, auditable comparison of how much a gradient-
boosted model actually improves on much simpler baselines.

## 3. Data sources

| Source | Dataset | Endpoint |
|---|---|---|
| ONS — Dados Abertos | Carga de Energia Verificada (semi-hourly verified load by subsystem) | `https://apicarga.ons.org.br/prd/cargaverificada?dat_inicio=YYYY-MM-DD&dat_fim=YYYY-MM-DD&cod_areacarga=AREA` — documented at [dados.ons.org.br/dataset/carga-energia-verificada](https://dados.ons.org.br/dataset/carga-energia-verificada) |
| Open-Meteo | Hourly weather (temperature, apparent temperature, humidity, precipitation, wind speed) | Archive API for historical data, Forecast API for recent/near-future data — no key required |

`AREA` is one of the four SIN subsystem codes: `N` (Norte), `NE` (Nordeste),
`S` (Sul), `SE` (Sudeste/Centro-Oeste).

**A note on the ONS response schema.** The request URL above is confirmed live
and documented by ONS. ONS does not publish a stable machine-readable schema
for the JSON field names in this particular endpoint (unlike its CSV/CKAN
datasets, which ship a data dictionary). `src/data/ons.py` therefore does not
hardcode field names: it detects the timestamp and load-value columns by
pattern from whatever the live response contains, logs exactly what it found,
and raises a clear `OnsDataError` — including the raw column list — if it
cannot confidently identify them, rather than guessing. The first time this
runs against the live API in an environment with network access, check the
log line `ONS response schema detected: timestamp='...', load='...'` and, if
needed, extend the candidate lists at the top of that file.

## 4. Architecture

```
ONS + Open-Meteo
      ↓
Data Validation
      ↓
Feature Engineering
      ↓
Temporal Backtesting
      ↓
Model Tournament
      ↓
Best Model
      ↓
Forecast (+ Peak Detection, Error Analysis, Scenarios)
      ↓
Streamlit
```

```
electricity-load-forecasting/
├── app.py                        # single-page Streamlit app
├── requirements.txt
├── .env.example
├── src/
│   ├── data/
│   │   ├── ons.py                # ONS API client (schema auto-detection)
│   │   ├── weather.py            # Open-Meteo API client
│   │   ├── ingestion.py          # orchestration + disk cache
│   │   └── validation.py         # timestamps, gaps, nulls, impossible values
│   ├── features/
│   │   └── engineering.py        # lags, rolling stats, calendar, weather
│   ├── models/
│   │   ├── baseline.py           # Seasonal Naive, Moving Average
│   │   ├── statistical.py        # Exponential Smoothing (Holt-Winters)
│   │   ├── lightgbm_model.py     # LightGBM quantile regression (P10/P50/P90)
│   │   └── tournament.py         # same-methodology model comparison
│   ├── forecasting/
│   │   └── forecast.py           # multi-horizon recursive forecasting
│   ├── evaluation/
│   │   ├── metrics.py            # MAE, RMSE, MAPE, sMAPE, WAPE
│   │   └── backtesting.py        # walk-forward fold construction
│   ├── analysis/
│   │   ├── peak_detection.py
│   │   ├── error_analysis.py
│   │   ├── temperature_load.py
│   │   └── scenario.py
│   └── utils/
│       ├── config.py
│       └── logging.py
└── tests/                        # fixtures + unit tests, no network calls
```

## 5. Feature engineering

- **Lags** (adapted to the series' real 30-minute frequency): 30min, 1h, 2h,
  12h, 24h, 48h, 168h (1 week).
- **Rolling statistics** (mean, std, min, max) over 3h, 24h, and 168h windows,
  computed on the series shifted by one period so the current row is never
  included in its own rolling window.
- **Calendar**: hour, day of week, day of month, month, weekend flag, plus
  cyclical (sin/cos) encodings of hour and day-of-week.
- **Weather**: temperature, apparent temperature, relative humidity,
  precipitation, wind speed — used contemporaneously (not lagged), since they
  are exogenous inputs rather than derived from the target.

`src/features/engineering.py` builds every lag/rolling feature via `shift()`
before any rolling operation, and `tests/test_feature_engineering.py` asserts
that `lag_1` at any row equals the target at the corresponding earlier
timestamp — i.e., no feature can see its own or a future row's target value.

## 6. Models

- **Baselines**: Seasonal Naive (yesterday's value at the same time), Moving
  Average.
- **Statistical**: Exponential Smoothing (Holt-Winters, seasonal period = 1
  day). SARIMA was evaluated as the other statistical option in the brief;
  Exponential Smoothing was chosen because its seasonal period is set
  directly and it remains fast enough to refit inside every walk-forward
  fold, which repeated SARIMA order search at 48/336-period seasonality would
  not.
- **Machine Learning**: LightGBM, trained separately per quantile (P10/P50/P90)
  via the `quantile` objective.

XGBoost was not added: LightGBM already covers the gradient-boosted-tree slot
and no experiment in this repo showed a clear benefit from adding a second
GBM implementation.

## 7. Backtesting

Walk-forward (expanding-window) validation only — `shuffle=True` splits are
never used anywhere in this codebase:

```
Fold 1: TRAIN [........]      TEST [..]
Fold 2: TRAIN [...........]        TEST [..]
Fold 3: TRAIN [..............]           TEST [..]
```

Each fold trains on all data up to a cutoff and tests on the following
7-day window (configurable), then the cutoff advances. `src/evaluation/
backtesting.py` raises `ValueError` rather than silently running zero folds
if there isn't enough data. `tests/test_backtesting.py` asserts, for every
fold, that `max(train timestamps) < min(test timestamps)`.

## 8. Metrics

MAE, RMSE, MAPE, sMAPE, and WAPE (`src/evaluation/metrics.py`). MAPE and
sMAPE exclude rows where their denominator is within `1e-6` of zero (this
does not occur in practice for grid-scale load, but the guard is explicit
rather than silently producing `inf`/`NaN`). WAPE is used as the tournament's
ranking metric since it is scale-stable and not sensitive to the near-zero
denominator issue.

## 9. Forecasting

Horizons: **15 min, 1 hour, 24 hours, 7 days**. All horizons use the same
underlying tabular model with a **recursive** multi-step strategy — each
step's prediction is fed back into the lag features for the next step —
rather than training a separate model per horizon.

## 10. Probabilistic forecasting

- If **LightGBM** wins the tournament, P10/P50/P90 come directly from three
  independently trained quantile regressors (true quantile regression).
  Predictions are sorted row-wise to guarantee P10 ≤ P50 ≤ P90, since
  independently trained quantile models can occasionally cross.
- If a **baseline or Exponential Smoothing** wins, bands are a **residual-
  bootstrap approximation**: the empirical 10th/90th percentile of recent
  errors, added around the point forecast. This is explicitly labeled in the
  app (`band_method = "residual_bootstrap"`) — it is never presented as true
  quantile regression when it isn't.

## 11. Peak detection

`src/analysis/peak_detection.py` finds the maximum value in the current
forecast window, its timestamp, and its percentage difference from the
recent (last 24h) average load.

## 12. Error analysis

`src/analysis/error_analysis.py` breaks down absolute error by hour of day,
day of week, month, load quartile, peak vs. non-peak periods, and (when
temperature is available) temperature bins.

## 13. Temperature–Load Relationship

Described throughout as an **observed relationship in the historical
dataset** — correlation and binned average load by temperature — never as a
causal claim.

## 14. Scenario analysis

`src/analysis/scenario.py` re-runs the trained LightGBM model with the most
recent observation's temperature shifted by −2°C to +2°C, holding all other
features fixed, to show how the forecast responds. This is a controlled
what-if, not a causal estimate, and is labeled as such in the app.

## 15. Installation

```bash
git clone https://github.com/edu-moraess/electricity-load-forecasting.git
cd electricity-load-forecasting
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\\Scripts\\activate         # Windows
pip install -r requirements.txt
cp .env.example .env               # optional; defaults work without edits
```

## 16. Usage

```bash
# Run the test suite (uses local fixtures only, no network calls)
pytest -q

# Launch the app
streamlit run app.py
```

The sidebar lets you choose the SIN subsystem (N/NE/S/SE) and training
lookback window. The main page shows data status, the forecast chart with
probabilistic bands, the next forecast peak, model tournament results,
error analysis, the temperature–load relationship, and (when LightGBM wins)
feature importances and a temperature scenario table.

## 17. Limitations

- **ONS response schema was not empirically verified against a live call
  during development** of this repository — see §3. The request URL is
  confirmed and documented by ONS; the exact JSON field names are detected
  defensively at runtime rather than hardcoded from an untested guess. This
  is a deliberate, documented choice to avoid silently shipping wrong field
  mappings, not a shortcut.
- The reference weather coordinates per subsystem (one city standing in for
  an entire subsystem) are an approximation, not a load-weighted centroid.
- Probabilistic bands default to a residual-bootstrap approximation whenever
  a non-LightGBM model wins the tournament (see §10).
- No database is used; the merged dataset is cached to CSV under `data/cache/`
  with a TTL, which is sufficient at this project's scale and avoids the
  operational overhead of standing up PostgreSQL for a single-page app.

## 18. Future improvements

- Empirically validate and, if needed, extend the ONS column-detection
  candidates against a live response, then pin the exact field names.
- Replace the single reference-city weather proxy with a load-weighted
  aggregate across each subsystem's major population centers.
- Add a direct (non-recursive) multi-horizon model for the 7-day horizon to
  avoid recursive error accumulation over very long forecast paths.
- Persist backtest history over time to track model drift, rather than
  recomputing the tournament from scratch on every data refresh.
