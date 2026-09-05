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
    lookback_days = st.slider("Training lookback (days)", min_value=90, max_value=365, value=120, step=15)
    force_refresh = st.button("Force data refresh")

end_date = date.today()
start_date = end_date - timedelta(days=lookback_days)

# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
try:
    with st.spinner("Fetching ONS load data and Open-Meteo weather data..."):
        ingestion = run_ingestion(subsystem, start_date, end_date, force_refresh=force_refresh)
except IngestionError as exc:
    st.error(
        "Data ingestion failed and no cached data is available. "
        "This is shown as-is, not masked with placeholder data.\n\n"
        f"Details: {exc}"
    )
    st.stop()

status_cols = st.columns([2, 2, 2, 2])
status_cols[0].metric("Region", f"{subsystem} — {ONS_SUBSYSTEMS[subsystem]}")
status_cols[1].metric("Last update", ingestion.fetched_at.strftime("%Y-%m-%d %H:%M"))
status_label = "● LIVE" if not ingestion.from_cache else ("● STALE CACHE" if ingestion.is_stale else "● CACHED")
status_cols[2].metric("Data status", status_label)
status_cols[3].metric("Observations", f"{len(ingestion.frame):,}")

if ingestion.source_errors:
    st.warning(
        "Some data sources reported problems during this run (shown here rather than "
        "silently substituted):\n\n" + "\n".join(f"- {e}" for e in ingestion.source_errors)
    )

with st.expander("Data validation details"):
    st.write("**Load series:**", ingestion.load_report.summary())
    st.write("**Weather series:**", ingestion.weather_report.summary())
    if ingestion.load_report.gaps:
        st.write("Gaps detected (never filled with invented values):")
        st.dataframe(pd.DataFrame(ingestion.load_report.gaps))

df = ingestion.frame

min_required_rows = 67 * 48  # ~ (60 train + 7 test) days at 48 periods/day, one fold minimum
if len(df) < min_required_rows:
    st.error(
        f"Only {len(df)} observations available; at least {min_required_rows} are needed for "
        "walk-forward backtesting with the configured minimum train/test windows. "
        "Increase the training lookback in the sidebar."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Model tournament (cached per subsystem + data length so it reruns when new
# data arrives, not on every widget interaction)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Running model tournament (walk-forward backtesting)...")
def _cached_tournament(subsystem_key: str, n_rows: int, last_ts: str):
    return run_tournament(df)


tournament = _cached_tournament(subsystem, len(df), str(df["timestamp"].max()))

st.subheader("Forecast horizon")
horizon_label = st.radio(
    "Horizon", options=list(HORIZONS_MINUTES.keys()), horizontal=True, label_visibility="collapsed"
)

# Residuals from the best model's most recent fold, used for residual-bootstrap
# bands when the winner is not LightGBM.
best_model = tournament.best_model_name
recent_fold_errors = tournament.per_fold_metrics[tournament.per_fold_metrics["model"] == best_model]
residuals_proxy = None
fitted_lightgbm = None

if best_model == "LightGBM":
    featured_full = build_feature_matrix(df)
    feature_cols = get_feature_columns(featured_full)
    fitted_lightgbm = LightGBMQuantileModel()
    fitted_lightgbm.fit(featured_full[feature_cols], featured_full["load_mw"])
else:
    # Approximate residual distribution from the tail of the training data
    # using a same-day-last-week naive comparison, purely to size the bands.
    residuals_proxy = (df["load_mw"] - df["load_mw"].shift(48 * 7)).dropna().values

forecast = generate_forecast(
    df, horizon_label, best_model, residuals=residuals_proxy, fitted_lightgbm=fitted_lightgbm
)

st.subheader("Load Forecast")
chart_df = pd.DataFrame({"timestamp": forecast.timestamps, "P50": forecast.p50})
if forecast.p10 is not None:
    chart_df["P10"] = forecast.p10
    chart_df["P90"] = forecast.p90

history_tail = df.tail(48 * 3)[["timestamp", "load_mw"]].rename(columns={"load_mw": "Historical"})
plot_df = pd.merge(history_tail, chart_df, on="timestamp", how="outer").sort_values("timestamp")
st.line_chart(plot_df.set_index("timestamp"))
if forecast.band_method == "residual_bootstrap":
    st.caption(
        "P10/P90 bands are a residual-bootstrap approximation around the winning model's "
        "point forecast, not full quantile regression (see README § Probabilistic forecasting)."
    )
elif forecast.band_method == "none":
    st.caption("No probabilistic bands available for this run (insufficient residual history).")

peak = detect_next_peak(forecast.timestamps, forecast.p50, df["load_mw"])
st.subheader("Next Forecast Peak")
peak_cols = st.columns(3)
peak_cols[0].metric("Peak load", f"{peak.peak_value_mw:,.1f} MW")
peak_cols[1].metric("Peak time", peak.peak_time.strftime("%Y-%m-%d %H:%M"))
peak_cols[2].metric("vs. recent average", f"{peak.pct_vs_recent_average:+.1f}%")

st.subheader("Model Performance")
st.dataframe(tournament.summary.style.highlight_min(subset=["MAE", "RMSE", "MAPE", "sMAPE", "WAPE"], color="#1f6f43"))
st.caption(f"Best model (lowest mean WAPE across walk-forward folds): **{best_model}**")
if tournament.errors:
    with st.expander("Models that failed during backtesting"):
        for name, errs in tournament.errors.items():
            st.write(f"**{name}**: {errs[0]}")

if best_model == "LightGBM" and fitted_lightgbm is not None:
    st.subheader("Forecast Drivers")
    importances = fitted_lightgbm.feature_importance().head(10)
    st.bar_chart(importances)

st.subheader("Error Analysis")
last_fold_test_len = tournament.per_fold_metrics["fold"].max()
# Reconstruct a representative error frame from the most recent backtest fold
# for the winning model by recomputing point predictions on the held-out window.
try:
    from src.evaluation.backtesting import make_walk_forward_folds, iter_fold_frames

    folds = make_walk_forward_folds(len(df))
    last_fold, train_df, test_df = list(iter_fold_frames(df, folds))[-1]
    last_forecast = generate_forecast(
        train_df, "24h" if len(test_df) >= horizon_to_periods("24h") else "1h", best_model,
        residuals=residuals_proxy, fitted_lightgbm=fitted_lightgbm,
    )
    n = min(len(last_forecast.p50), len(test_df))
    err_df = build_error_frame(
        test_df["timestamp"].values[:n],
        test_df["load_mw"].values[:n],
        last_forecast.p50[:n],
        temperature=test_df["temperature_2m"].values[:n] if "temperature_2m" in test_df else None,
    )
    ecol1, ecol2 = st.columns(2)
    ecol1.bar_chart(error_by_hour(err_df).set_index("hour"))
    ecol1.caption("Mean absolute error by hour of day (most recent backtest fold)")
    peak_err = error_during_peaks(err_df)
    ecol2.metric("Mean abs. error — peak periods", f"{peak_err['peak_mean_abs_error']:.1f} MW")
    ecol2.metric("Mean abs. error — non-peak periods", f"{peak_err['non_peak_mean_abs_error']:.1f} MW")
except Exception as exc:  # noqa: BLE001
    st.info(f"Error analysis unavailable for this run: {exc}")

st.subheader("Temperature–Load Relationship")
st.caption("Observed relationship in the historical dataset. No causal claim is implied.")
if "temperature_2m" in df.columns:
    corr = observed_correlation(df["load_mw"], df["temperature_2m"])
    st.metric("Correlation (load vs. temperature)", f"{corr:.2f}")
    binned = load_by_temperature_bin(df["load_mw"], df["temperature_2m"])
    st.bar_chart(binned.set_index("temperature_bin")["mean_load_mw"])

if best_model == "LightGBM" and fitted_lightgbm is not None:
    st.subheader("Temperature Scenario")
    st.caption("Controlled what-if: shifts the most recent observation's temperature and re-forecasts. Not a causal estimate.")
    featured_full = build_feature_matrix(df)
    last_row = featured_full.iloc[[-1]]
    try:
        scenarios = run_temperature_scenarios(fitted_lightgbm, last_row)
        st.dataframe(scenarios)
    except Exception as exc:  # noqa: BLE001
        st.info(f"Scenario analysis unavailable: {exc}")

st.subheader("Data & Model Status")
st.write(
    {
        "Data source (load)": "ONS — dados.ons.org.br (carga verificada)",
        "Data source (weather)": "Open-Meteo",
        "Observations": len(df),
        "Training period": f"{df['timestamp'].min()} → {df['timestamp'].max()}",
        "Last observation": str(df["timestamp"].max()),
        "Best model": best_model,
    }
)
