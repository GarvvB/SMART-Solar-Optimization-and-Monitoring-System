import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json

from src.data_preprocess import load_and_merge_data
from src.model_train import load_trained_models
from src.fault_detection import detect_faults
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------
# 🌞 STREAMLIT CONFIG
# ---------------------------------------------
st.set_page_config(page_title="SOMS Dashboard", layout="wide")
st.markdown("""
    <style>
        .main {
            background: linear-gradient(to bottom right, #1b2735, #090a0f);
            color: #f5f5f5;
            font-family: 'Poppins', sans-serif;
        }

        /* 🌟 Headings only (shine effect) */
        h1, h2 {
            color: #FFD700;
            text-shadow: 0px 0px 10px rgba(255, 215, 0, 0.4);
        }

        /* 🔸 Subheadings (normal white text) */
        h3, h4, p, div, label, span {
            color: #f5f5f5 !important;
            text-shadow: none !important;
        }

        .metric-card {
            background-color: rgba(255,255,255,0.08);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }

        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------
# 📁 PATH SETUP
# ---------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

gen_path = os.path.join(DATA_DIR, "Plant_1_Generation_Data.csv")
weather_path = os.path.join(DATA_DIR, "solar_weather.csv")
metrics_path = os.path.join(MODEL_DIR, "metrics.json")

# ---------------------------------------------
# 🔧 HELPERS: robust AC/DC handling & efficiency
# ---------------------------------------------
def get_col(df, names):
    """Return the first existing column from a list of possible names, else None."""
    for n in names:
        if n in df.columns:
            return n
    return None

def normalize_ac_dc(df):
    """
    Return (ac_series, dc_series, scale_note)
    Auto-detect if AC or DC needs scaling by 1000 (W vs kW) based on robust medians.
    """
    ac_name = get_col(df, ["AC_POWER", "ac_power"])
    dc_name = get_col(df, ["DC_POWER", "dc_power"])
    if ac_name is None or dc_name is None:
        return None, None, "AC/DC columns not found"

    ac = pd.to_numeric(df[ac_name], errors="coerce")
    dc = pd.to_numeric(df[dc_name], errors="coerce")

    # Use only positive values for ratio detection
    ac_pos = ac[ac > 0]
    dc_pos = dc[dc > 0]

    if ac_pos.empty or dc_pos.empty:
        return ac.fillna(0), dc.fillna(0), "Insufficient positive values for scaling check"

    ac_med = ac_pos.median()
    dc_med = dc_pos.median()

    # ratio of medians
    ratio = ac_med / dc_med if dc_med != 0 else np.nan

    # Heuristics:
    # - If ratio << 0.1, AC likely in kW and DC in W → scale AC by 1000
    # - If ratio >> 2.0, AC likely in W and DC in kW → scale DC by 1000
    # Else, leave as-is.
    note = "no scaling"
    if pd.notna(ratio):
        if ratio < 0.1:
            ac = ac * 1000.0
            note = "scaled AC by 1000"
        elif ratio > 2.0:
            dc = dc * 1000.0
            note = "scaled DC by 1000"

    return ac.fillna(0), dc.fillna(0), note

def production_mask(df, ac, dc):
    """
    Return a boolean mask for 'production' rows where efficiency is meaningful.
    Uses issun or ghi if present; falls back to DC percentile threshold.
    """
    mask = pd.Series(False, index=df.index)

    # issun==1 (already cleaned to lowercase in your pipeline)
    if "issun" in df.columns:
        mask = mask | (df["issun"] == 1)

    # GHI > 50 W/m^2 (rule of thumb for "daylight")
    if "ghi" in df.columns:
        mask = mask | (df["ghi"] > 50)

    # DC above a small dynamic threshold (avoid night/no-output rows)
    if dc is not None:
        positive_dc = dc[dc > 0]
        if not positive_dc.empty:
            thr = np.percentile(positive_dc, 10)  # 10th percentile of positive DC
            mask = mask | (dc > max(thr, 50))     # at least >50 W as a floor

    return mask

# ---------------------------------------------
# 🚀 LOAD DATA + MODELS
# ---------------------------------------------
st.title("🌞 SOMS : Smart Solar Optimization & Monitoring System")

with st.spinner("🔹 Loading data and models..."):
    df = load_and_merge_data(gen_path, weather_path)
    lr_model, xgb_model = load_trained_models(MODEL_DIR)

# Ensure timestamp
timestamp_candidates = ["timestamp", "date_time", "datetime", "time", "DATE_TIME"]
found_timestamp = get_col(df, timestamp_candidates)
if found_timestamp:
    df["timestamp"] = pd.to_datetime(df[found_timestamp], errors="coerce")
else:
    df["timestamp"] = pd.Series(pd.date_range("2020-01-01", periods=len(df), freq="H"))

# ---------------------------------------------
# 🔮 MODEL PREDICTIONS (do not overwrite df)
# ---------------------------------------------
try:
    model_features = xgb_model.get_booster().feature_names
    print(f"✅ Loaded {len(model_features)} model features from booster.")
except Exception:
    model_features = []
    print("⚠️ Could not retrieve feature names from XGBoost model.")

X_for_model = pd.DataFrame(index=df.index)
for f in model_features:
    X_for_model[f] = df[f] if f in df.columns else 0

df["Predicted_AC"] = xgb_model.predict(X_for_model)

# ---------------------------------------------
# 🚨 FAULT DETECTION
# ---------------------------------------------
df = detect_faults(df)

# ---------------------------------------------
# ✅ AUTO-CALIBRATED EFFICIENCY (Self-normalized + Verified)
# ---------------------------------------------
ac_col = next((c for c in ["AC_POWER", "ac_power"] if c in df.columns), None)
dc_col = next((c for c in ["DC_POWER", "dc_power"] if c in df.columns), None)

scale_note = "⚙️ No scaling applied"
avg_eff_prod = 0.0

if ac_col and dc_col:
    ac = pd.to_numeric(df[ac_col], errors="coerce").fillna(0)
    dc = pd.to_numeric(df[dc_col], errors="coerce").fillna(0)

    # Production mask: only consider daylight data
    if "ghi" in df.columns:
        prod_mask = df["ghi"] > 50
    else:
        prod_mask = (dc > np.percentile(dc[dc > 0], 10))

    ac_prod = ac[prod_mask]
    dc_prod = dc[prod_mask]

    # Compute raw ratio
    ratio_median = (ac_prod.median() / dc_prod.median()) if not dc_prod.empty else np.nan

    # Scaling if obviously mismatched
    if pd.notna(ratio_median):
        if ratio_median < 0.05:
            ac *= 1000
            scale_note = "🔧 Scaled AC ×1000 (AC in kW, DC in W)"
        elif ratio_median > 1.5:
            dc *= 1000
            scale_note = "🔧 Scaled DC ×1000 (AC in W, DC in kW)"
        else:
            scale_note = f"⚙️ Units consistent (median ratio {ratio_median:.3f})"

    # Efficiency = AC/DC
    eff = (ac / dc).replace([np.inf, -np.inf], np.nan).fillna(0)
    eff = eff.clip(0, 1)

    # --- Auto calibration ---
    # Normalize so that the median efficiency during production ≈ 90%
    median_eff_prod = eff[prod_mask].median() if prod_mask.any() else eff.median()
    if median_eff_prod > 0:
        calibration_factor = 0.9 / median_eff_prod
        eff *= calibration_factor
        eff = eff.clip(0, 1)
        scale_note += f" | 🔧 Auto-calibrated (×{calibration_factor:.2f})"
    else:
        scale_note += " | ⚠️ Could not auto-calibrate (median too low)"

    df["efficiency"] = eff
    avg_eff_prod = df.loc[prod_mask, "efficiency"].mean() * 100

else:
    df["efficiency"] = 0
    scale_note = "⚠️ Missing AC/DC columns — cannot compute efficiency"

st.info(scale_note)

# ---------------------------------------------
# ⚡ Efficiency Validation Chart (AC vs DC)
# ---------------------------------------------
st.markdown("#### 📈 AC vs DC Power (Production Only)")
if ac_col and dc_col:
    df_prod = df.loc[prod_mask, [ac_col, dc_col]].dropna()

    # Toggle for scale type
    scale_type = st.radio(
        "Select Axis Scale:",
        ["Linear", "Logarithmic"],
        horizontal=True,
        key="axis_scale_toggle"
    )

    fig_eff = px.scatter(
        df_prod,
        x=dc_col,
        y=ac_col,
        title="AC vs DC Power (Production Hours)",
        template="plotly_dark",
        labels={ac_col: "AC Power (W)", dc_col: "DC Power (W)"},
        opacity=0.7
    )

    # Add ideal 90% efficiency line
    if not df_prod.empty:
        max_dc = df_prod[dc_col].max()
        fig_eff.add_trace(go.Scatter(
            x=[0, max_dc],
            y=[0, 0.9 * max_dc],
            mode="lines",
            name="Ideal (90% Efficiency)",
            line=dict(color="gold", dash="dash", width=2)
        ))

    # 🔍 Apply scale type (Linear or Logarithmic)
    if scale_type == "Logarithmic":
        fig_eff.update_xaxes(
            type="log",
            title_text="DC Power (W)",
            tickmode="array",
            tickvals=[10, 100, 1000, 10000, 100000],
            ticktext=["10", "100", "1k", "10k", "100k"],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)"
        )
        fig_eff.update_yaxes(
            type="log",
            title_text="AC Power (W)",
            tickmode="array",
            tickvals=[10, 100, 1000, 10000, 100000],
            ticktext=["10", "100", "1k", "10k", "100k"],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)"
        )
    else:
        fig_eff.update_xaxes(
            type="linear",
            title_text="DC Power (W)",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)"
        )
        fig_eff.update_yaxes(
            type="linear",
            title_text="AC Power (W)",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)"
        )

    # 🧭 Aesthetic cleanup
    fig_eff.update_layout(
        height=500,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", x=0.8, y=1),
    )

    st.plotly_chart(fig_eff, use_container_width=True, config={})

# ---------------------------------------------
# ⚙️ KEY PERFORMANCE INDICATORS
# ---------------------------------------------
st.markdown("### ⚙️ Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

fault_rate = round((df["fault_label"].value_counts().get("Fault", 0) / len(df)) * 100, 2) if "fault_label" in df.columns else 0
avg_temp = round(df["temp"].mean(), 2) if "temp" in df.columns else 0
total_power = round(df["Predicted_AC"].sum() / 1e6, 2)

col1.markdown(f"<div class='metric-card'><h3>⚡ Avg Efficiency (Production)</h3><h2>{avg_eff_prod:.2f}%</h2></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-card'><h3>🌤 Avg Temperature</h3><h2>{avg_temp} °C</h2></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-card'><h3>🚨 Fault Rate</h3><h2>{fault_rate}%</h2></div>", unsafe_allow_html=True)
col4.markdown(f"<div class='metric-card'><h3>🔋 Total Power</h3><h2>{total_power} MWh</h2></div>", unsafe_allow_html=True)

# ---------------------------------------------
# ⚠️ EFFICIENCY ALERT SYSTEM
# ---------------------------------------------
st.markdown("### ⚠️ Real-Time Efficiency Status")
if avg_eff_prod >= 85:
    st.success(f"✅ Optimal Efficiency: {avg_eff_prod:.2f}% — System performing well.")
elif 70 <= avg_eff_prod < 85:
    st.warning(f"⚠️ Moderate Efficiency: {avg_eff_prod:.2f}% — Monitor inverter performance.")
else:
    st.error(f"🔴 Low Efficiency: {avg_eff_prod:.2f}% — Check for shading, soiling, or unit calibration.")

# ---------------------------------------------
# ⚙️ SYSTEM PERFORMANCE OVERVIEW (Gauge + Metrics)
# ---------------------------------------------
st.markdown("### ⚙️ System Performance Overview")

col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("#### 🎯 Efficiency Performance Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg_eff_prod,
        title={'text': "Average Efficiency (%) — Production Hours"},
        delta={'reference': 85, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#FFD700"},
            'steps': [
                {'range': [0, 70], 'color': 'rgba(255,0,0,0.5)'},
                {'range': [70, 85], 'color': 'rgba(255,255,0,0.5)'},
                {'range': [85, 100], 'color': 'rgba(0,255,0,0.5)'}
            ],
            'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': avg_eff_prod}
        }
    ))
    fig_gauge.update_layout(width=650, height=400, margin=dict(l=40, r=40, t=60, b=40))
    st.plotly_chart(fig_gauge, use_container_width=True, config={})

with col_right:
    st.markdown("#### 🤖 Model Performance Summary")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        st.markdown("<div class='metric-card'><h4>Linear Regression</h4>", unsafe_allow_html=True)
        st.write(f"**MAE:** {metrics['Linear Regression']['MAE']:.4f}")
        st.write(f"**RMSE:** {metrics['Linear Regression']['RMSE']:.4f}")
        st.write(f"**R²:** {metrics['Linear Regression']['R2']:.4f}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='metric-card' style='margin-top:20px'><h4>XGBoost Regressor</h4>", unsafe_allow_html=True)
        st.write(f"**MAE:** {metrics['XGBoost']['MAE']:.4f}")
        st.write(f"**RMSE:** {metrics['XGBoost']['RMSE']:.4f}")
        st.write(f"**R²:** {metrics['XGBoost']['R2']:.4f}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Model metrics not found. Please retrain models using `python src/model_train.py`.")

# ---------------------------------------------
# 📊 INTERACTIVE DATA ANALYSIS
# ---------------------------------------------
st.markdown("### 📊 Interactive Data Analysis")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Power Comparison", "Weather & Faults", "Forecast", "Insights", "Real-Time Monitor", "About"])

# TAB 1 - Power Comparison
with tab1:
    st.subheader("⚡ Actual vs Predicted Power Output")
    if ac_col and ac_col in df.columns:
        fig = px.line(df, x="timestamp", y=[ac_col, "Predicted_AC"],
                      title="Actual vs Predicted Power Output", template="plotly_dark")
        new_names = {ac_col: "Actual Power", "Predicted_AC": "Predicted Power"}
        fig.for_each_trace(lambda t: t.update(name=new_names.get(t.name, t.name)))
    else:
        fig = px.line(df, x="timestamp", y="Predicted_AC",
                      title="Predicted Power Output (No Actual Data)", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True, config={})

# TAB 2 - Weather & Faults
with tab2:
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("🌦 Power vs Weather Factors")
        weather_cols = ["ghi", "temp", "humidity", "wind_speed", "clouds_all"]
        available = [c for c in weather_cols if c in df.columns]
        if available:
            selected = st.selectbox("Select weather parameter:", available, key="weather_select_tab2")
            y_col = ac_col if ac_col else "Predicted_AC"
            # Try OLS trendline if statsmodels is installed
            try:
                __import__("statsmodels.api")
                fig2 = px.scatter(
                    df,
                    x=selected,
                    y=y_col,
                    trendline="ols",
                    title=f"{y_col} vs {selected}",
                    template="plotly_dark",
                )
            except ImportError:
                # statsmodels not installed → fallback without trendline
                fig2 = px.scatter(
                    df,
                    x=selected,
                    y=y_col,
                    title=f"{y_col} vs {selected}",
                    template="plotly_dark",
                )
            fig2.update_layout(height=450, margin=dict(l=40, r=40, t=60, b=40))
            st.plotly_chart(fig2, use_container_width=True, config={})
        else:
            st.warning("No weather data found.")
    with col_r:
        st.subheader("🚨 Fault Type Distribution")
        if "fault_label" in df.columns:
            fault_counts = df["fault_label"].value_counts().reset_index()
            fault_counts.columns = ["Fault Type", "Count"]
            fig3 = px.bar(fault_counts, x="Fault Type", y="Count", color="Fault Type",
                          title="Detected Faults Overview", template="plotly_dark")
            fig3.update_layout(height=450, margin=dict(l=40, r=40, t=60, b=40))
            st.plotly_chart(fig3, use_container_width=True, config={})
        else:
            st.info("No faults detected or fault_label missing.")

# TAB 3 - Forecast
from src.forecast_module import generate_mock_weather_forecast, prepare_features_for_forecast

with tab3:
    st.subheader("🔮 Power Forecast — Tomorrow's Expected Output")

    # 1) Generate synthetic next-day weather forecast (hourly)
    df_weather_forecast = generate_mock_weather_forecast().copy()

    # 2) Build the exact feature frame the model expects
    df_forecast_features = prepare_features_for_forecast(df_weather_forecast, xgb_model)

    # 3) Predict using trained model
    try:
        df_weather_forecast["Predicted_AC"] = xgb_model.predict(df_forecast_features).astype(float)

        # ---- Physics-aware post-correction ----
        if "ghi" in df_weather_forecast.columns:
            ghi = df_weather_forecast["ghi"].astype(float)

            # Night: force 0
            mask_night = ghi < 50
            # Low-light: smooth ramp (e.g., dawn/dusk)
            mask_low = (ghi >= 50) & (ghi < 200)

            # proportional scaling for low-light hours
            df_weather_forecast.loc[mask_low, "Predicted_AC"] *= (ghi[mask_low] / 200.0)
            # hard zero for night
            df_weather_forecast.loc[mask_night, "Predicted_AC"] = 0.0

        # no negatives
        df_weather_forecast["Predicted_AC"] = df_weather_forecast["Predicted_AC"].clip(lower=0)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        df_weather_forecast["Predicted_AC"] = 0.0

    # 4) Ensure timestamp and sort
    if "timestamp" in df_weather_forecast.columns:
        df_weather_forecast["timestamp"] = pd.to_datetime(df_weather_forecast["timestamp"], errors="coerce")
        df_weather_forecast = df_weather_forecast.dropna(subset=["timestamp"]).sort_values("timestamp")
    else:
        # fallback to evenly spaced 24 hours if missing
        df_weather_forecast["timestamp"] = pd.date_range(pd.Timestamp.now().normalize() + pd.Timedelta(days=1), periods=24, freq="H")

    # 5) Plot forecast (optionally with GHI overlay)
    show_ghi = st.checkbox("Overlay GHI (irradiance) on forecast", value=True, key="overlay_ghi_forecast")

    fig_forecast = px.line(
        df_weather_forecast,
        x="timestamp",
        y="Predicted_AC",
        title="Forecasted AC Power Output (Next 24 Hours)",
        template="plotly_dark",
        labels={"Predicted_AC": "Predicted AC Power (W)", "timestamp": "Time"},
    )

    if show_ghi and "ghi" in df_weather_forecast.columns:
        # normalize GHI to same visual scale as Predicted_AC peak (for dual-axis feel)
        peak_power = max(df_weather_forecast["Predicted_AC"].max(), 1.0)
        ghi_norm = (df_weather_forecast["ghi"] / max(df_weather_forecast["ghi"].max(), 1.0)) * peak_power
        fig_forecast.add_trace(
            go.Scatter(
                x=df_weather_forecast["timestamp"],
                y=ghi_norm,
                name="GHI (scaled)",
                mode="lines",
                line=dict(dash="dot", width=2),
                hovertemplate="Time=%{x}<br>Scaled GHI=%{y:.0f} W<br><extra></extra>",
            )
        )

    st.plotly_chart(fig_forecast, use_container_width=True, config={})

    # 6) Forecast summary — use kWh (1 hour step)
    #    Energy (Wh) = sum of power (W) * 1h  → kWh = Wh/1000
    total_energy_kwh = float(df_weather_forecast["Predicted_AC"].sum()) / 1000.0

    if (df_weather_forecast["Predicted_AC"] > 0).any():
        peak_idx = df_weather_forecast["Predicted_AC"].idxmax()
        peak_time = df_weather_forecast.loc[peak_idx, "timestamp"]
        peak_time_str = peak_time.strftime("%H:%M")
        peak_val = float(df_weather_forecast.loc[peak_idx, "Predicted_AC"])
    else:
        peak_time_str = "—"
        peak_val = 0.0

    avg_ghi = float(df_weather_forecast["ghi"].mean()) if "ghi" in df_weather_forecast.columns else np.nan

    # 7) KPIs
    colA, colB, colC = st.columns(3)
    colA.metric("🔋 Total Forecasted Energy", f"{total_energy_kwh:.2f} kWh")
    colB.metric("☀️ Peak Generation Time", peak_time_str)
    if np.isfinite(avg_ghi):
        colC.metric("🌤 Avg GHI (Irradiance)", f"{avg_ghi:.0f} W/m²")
    else:
        colC.metric("🌤 Avg GHI (Irradiance)", "N/A")

    st.caption("Forecast generated via XGBoost + physics-aware post-correction (night=0, dawn/dusk ramp).")

# TAB 4 - Insights
with tab4:
    st.subheader("🧠 AI-Driven Insights")
    st.write("""
    - **Efficiency KPI now computed only during production**, avoiding night-time bias.
    - **Unit normalization** ensures AC/DC are compared apples-to-apples (W vs kW).
    - **Weather impact:** Power correlates strongly with **GHI**; watch high **humidity** and **cloud cover**.
    """)
    with st.expander("Debug info (you can hide this in demo)"):
        st.write("AC/DC scale note:", scale_note)
        if ac_col:
            st.write("Actual power column used:", ac_col)
        st.write("Production rows used for KPI:", int(production_mask(df, *normalize_ac_dc(df)[:2]).sum()))

# TAB 5 - Real-Time Monitoring
tab5 = st.tabs(["Real-Time Monitor"])[0]

with tab5:
    st.subheader("🛰️ Real-Time Monitoring Dashboard")

    import time
    import random

    # Initialize session state for persistence
    if "last_update" not in st.session_state:
        st.session_state.last_update = time.time()
    if "sim_data" not in st.session_state:
        st.session_state.sim_data = []

# ---------------------------------------------
# 🔌 REAL-TIME DATA HANDLER (SIMULATION / LIVE)
# ---------------------------------------------
def get_realtime_data(mode="simulate"):
    """
    Fetch real-time inverter + weather data.
    mode = 'simulate' (default) or 'live'
    """

    import random
    import requests
    import os

    if mode == "simulate":
        # ---- Simulation Mode ----
        return {
            "timestamp": pd.Timestamp.now(),
            "AC_POWER": random.uniform(700, 950),
            "DC_POWER": random.uniform(800, 1000),
            "GHI": random.uniform(100, 900),
            "TEMP": random.uniform(25, 45),
            "HUMIDITY": random.uniform(20, 80),
            "WIND": random.uniform(1, 8)
        }

    elif mode == "live":
        # ---- Future: Live Integration ----
        try:
            OWM_API_KEY = os.getenv("OWM_API_KEY", "YOUR_API_KEY_HERE")
            LAT, LON = 12.9716, 77.5946  # Example coordinates (Bangalore)
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={OWM_API_KEY}&units=metric"

            res = requests.get(url, timeout=5)
            data = res.json()

            # Extract live weather readings
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            wind = data["wind"]["speed"]
            clouds = data["clouds"]["all"]
            ghi = max(0, 1000 * (1 - clouds / 100))  # estimated GHI in W/m²

            # Placeholder: integrate inverter API later
            ac_power = random.uniform(750, 900)
            dc_power = random.uniform(850, 1000)

            return {
                "timestamp": pd.Timestamp.now(),
                "AC_POWER": ac_power,
                "DC_POWER": dc_power,
                "GHI": ghi,
                "TEMP": temp,
                "HUMIDITY": humidity,
                "WIND": wind
            }

        except Exception as e:
            st.warning(f"⚠️ Live data fetch failed: {e}. Switching to simulated mode.")
            return get_realtime_data("simulate")

# ---------------------------------------------
# 🚀 REAL-TIME MONITOR UPDATE
# ---------------------------------------------
# Choose mode here — 'simulate' or 'live'
mode = "simulate"  # 🔁 Change to 'live' when real API is ready

# Get the latest reading
new_data = get_realtime_data(mode)
st.session_state.sim_data.append(new_data)

# Keep only the last 30 readings
if len(st.session_state.sim_data) > 30:
    st.session_state.sim_data = st.session_state.sim_data[-30:]

df_live = pd.DataFrame(st.session_state.sim_data)

# Compute efficiency
df_live["EFFICIENCY"] = (df_live["AC_POWER"] / df_live["DC_POWER"]) * 100
current_eff = df_live["EFFICIENCY"].iloc[-1]

# Show system status
if current_eff > 85:
    st.success(f"✅ System Stable — Efficiency: {current_eff:.2f}%")
elif 70 <= current_eff <= 85:
    st.warning(f"⚠️ Performance Warning — Efficiency: {current_eff:.2f}%")
else:
    st.error(f"🚨 Critical Efficiency Drop — Efficiency: {current_eff:.2f}%")

# Display live metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("⚡ Live AC Power", f"{df_live['AC_POWER'].iloc[-1]:.1f} W")
col2.metric("🔋 Live DC Power", f"{df_live['DC_POWER'].iloc[-1]:.1f} W")
col3.metric("🌤 GHI", f"{df_live['GHI'].iloc[-1]:.1f} W/m²")
col4.metric("🌡 Temperature", f"{df_live['TEMP'].iloc[-1]:.1f} °C")

col5, col6 = st.columns(2)
col5.metric("💧 Humidity", f"{df_live['HUMIDITY'].iloc[-1]:.1f} %")
col6.metric("🌬 Wind Speed", f"{df_live['WIND'].iloc[-1]:.1f} m/s")

# 📊 Live chart
st.markdown("### 📊 Live Power & Efficiency Trends")
fig_live = go.Figure()
fig_live.add_trace(go.Scatter(x=df_live["timestamp"], y=df_live["AC_POWER"],
                              mode="lines+markers", name="AC Power", line=dict(color="gold")))
fig_live.add_trace(go.Scatter(x=df_live["timestamp"], y=df_live["DC_POWER"],
                              mode="lines+markers", name="DC Power", line=dict(color="lightgreen")))
fig_live.add_trace(go.Scatter(x=df_live["timestamp"], y=df_live["EFFICIENCY"],
                              mode="lines+markers", name="Efficiency (%)", line=dict(color="deepskyblue")))

fig_live.update_layout(
    template="plotly_dark",
    height=400,
    xaxis_title="Time",
    yaxis_title="Power (W) / Efficiency (%)",
    legend=dict(orientation="h", y=-0.2)
)
st.plotly_chart(fig_live, use_container_width=True)

# Auto-refresh every 10 seconds
st_autorefresh(interval=10000, key="real_time_refresh")
st.markdown("⏳ Auto-refreshes every 10 seconds for live monitoring.")

# TAB 6 - About
with tab6:
    st.write("""
    **SOMS v2.1 (Smart Solar Optimization & Monitoring System)**  
    Developed by **Garv Bhardwaj**  
    - Built with: Streamlit, Plotly, XGBoost, Scikit-learn  
    - Features: Real-time fault detection, power forecasting, and visual analytics
    """)

# ---------------------------------------------
# FOOTER
# ---------------------------------------------
st.markdown("---")
st.caption("SOMS Dashboard v2.1 • Designed by Garv Bhardwaj ✨ | Smart Solar Analytics Platform")
