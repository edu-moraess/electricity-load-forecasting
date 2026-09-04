"""
Electricity Load Forecasting -- single-page Streamlit application.

Short-term electricity demand forecasting using real operational (ONS) and
weather (Open-Meteo) data. No synthetic or mocked data is used anywhere in
this app: if a data source is unavailable, the problem is shown, not hidden.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src.analysis.error_analysis import build_error_frame, error_by_hour, error_during_peaks
from src.analysis.peak_detection import detect_next_peak
from src.analysis.scenario import run_temperature_scenarios
from src.analysis.temperature_load import load_by_temperature_bin, observed_correlation
from src.data.ingestion import IngestionError, run_ingestion
from src.features.engineering import build_feature_matrix, get_feature_columns
from src.forecasting.forecast import generate_forecast, horizon_to_periods
from src.models.lightgbm_model import LightGBMQuantileModel
from src.models.tournament import run_tournament
from src.utils.config import HORIZONS_MINUTES, ONS_SUBSYSTEMS

st.set_page_config(page_title="Electricity Load Forecasting", layout="wide")

# ---------------------------------------------------------------------------
# Minimal, technical visual style -- no gradients, no decorative elements.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Electricity Load Forecasting")
st.caption("Short-term electricity demand forecasting using real operational and weather data.")

with st.sidebar:
    st.subheader("Region")
    subsystem = st.selectbox(
        "SIN subsystem",
        options=list(ONS_SUBSYSTEMS.keys()),
        format_func=lambda k: f"{k} — {ONS_SUBSYSTEMS[k]}",
        index=list(ONS_SUBSYSTEMS.keys()).index("SE"),
    )
    st.subheader("Training window")
    lookback_days = st.slider("Lookback days", min_value=30, max_value=180, value=90, step=15)
    st.subheader("Forecast horizon")
    horizon_label = st.selectbox(
        "Horizon",
        options=list(HORIZONS_MINUTES.keys()),
        index=2,
    )
    run_btn = st.button("Run pipeline", type="primary")

if not run_btn:
    st.info("Configure the sidebar and click **Run pipeline** to load real ONS + Open-Meteo data and produce forecasts.")
    st.stop()

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
end = date.today()
start = end - timedelta(days=lookback_days)

with st.spinner("Ingesting ONS load + Open-Meteo weather…"):
    try:
        merged = run_ingestion(subsystem=subsystem, start=start, end=end)
    except IngestionError as exc:
        st.error(f"Data ingestion failed: {exc}")
        st.stop()

st.success(f"Loaded {len(merged):,} rows · {merged['timestamp'].min()} → {merged['timestamp'].max()}")

col1, col2, col3 = st.columns(3)
col1.metric("Subsystem", f"{subsystem} — {ONS_SUBSYSTEMS[subsystem]}")
col2.metric("Rows", f"{len(merged):,}")
col3.metric("Mean load (MW)", f"{merged['load_mw'].mean():,.0f}")

# Feature matrix
features = build_feature_matrix(merged)
feature_cols = get_feature_columns(features)

# Tournament
with st.spinner("Running model tournament (walk-forward)…"):
    tournament = run_tournament(features, feature_cols)

st.subheader("Model tournament (walk-forward WAPE)")
st.dataframe(tournament["summary"], use_container_width=True)
best_name = tournament["best_model_name"]
st.caption(f"Selected model: **{best_name}**")

# Forecast
horizon_min = HORIZONS_MINUTES[horizon_label]
n_periods = horizon_to_periods(horizon_min, freq_minutes=30)

with st.spinner(f"Generating {horizon_label} forecast…"):
    forecast_df = generate_forecast(
        features,
        feature_cols,
        model_name=best_name,
        n_periods=n_periods,
        residuals=tournament.get("residuals"),
    )

st.subheader(f"Forecast — {horizon_label}")
st.line_chart(forecast_df.set_index("timestamp")[["p50", "p10", "p90"]].rename(columns={"p50": "P50", "p10": "P10", "p90": "P90"}))

# Peak
peak = detect_next_peak(forecast_df, recent_avg=merged["load_mw"].tail(48).mean())
st.metric("Next peak (forecast)", f"{peak['peak_mw']:,.0f} MW", f"{peak['pct_above_avg']:+.1f}% vs 24h avg · {peak['peak_time']}")

# Error analysis (from backtest residuals if available)
if "error_frame" in tournament:
    st.subheader("Error analysis")
    err = tournament["error_frame"]
    st.dataframe(error_by_hour(err), use_container_width=True)

# Temperature-load
st.subheader("Temperature × Load (observed)")
binned = load_by_temperature_bin(merged)
st.dataframe(binned, use_container_width=True)
corr = observed_correlation(merged)
st.caption(f"Pearson correlation (temp vs load): {corr:.3f}")

# Scenarios (only if LightGBM)
if best_name.lower().startswith("lightgbm") and "model" in tournament:
    st.subheader("Temperature scenarios (±2 °C)")
    scenarios = run_temperature_scenarios(tournament["model"], features, feature_cols)
    st.dataframe(scenarios, use_container_width=True)
