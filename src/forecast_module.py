import pandas as pd
import numpy as np
import datetime as dt

def generate_mock_weather_forecast(hours=24, seed=42):
    """Generate realistic next-day solar weather forecast (hourly)."""
    np.random.seed(seed)
    base_time = dt.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + dt.timedelta(days=1)
    hours_list = [base_time + dt.timedelta(hours=i) for i in range(hours)]

    # Define GHI (Global Horizontal Irradiance) as bell-shaped daylight curve
    ghi = []
    for h in range(hours):
        if 6 <= h <= 18:  # approximate daylight hours
            val = max(0, np.random.normal(800 * np.sin((h - 6) * np.pi / 12), 100))
        else:
            val = 0  # no sun at night
        ghi.append(val)

    ghi = np.array(ghi)
    temp = np.clip(20 + 10 * np.sin((np.array(range(hours)) - 6) * np.pi / 12) + np.random.normal(0, 2, hours), 10, 45)
    humidity = np.clip(60 - 15 * np.sin((np.array(range(hours)) - 6) * np.pi / 12) + np.random.normal(0, 5, hours), 20, 90)
    wind_speed = np.clip(np.random.normal(3, 1, hours), 0, 8)
    clouds_all = np.clip(100 - ghi / 10 + np.random.normal(0, 5, hours), 0, 100)

    df_forecast = pd.DataFrame({
        "timestamp": hours_list,
        "ghi": ghi,
        "temp": temp,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "clouds_all": clouds_all
    })

    return df_forecast

def prepare_features_for_forecast(df_weather, model):
    """Match feature structure to model’s expected input, with realistic mock values."""
    try:
        model_features = model.get_booster().feature_names
    except Exception:
        model_features = []

    df_input = df_weather.copy()

    # Time-based features
    df_input["hour"] = df_weather["timestamp"].dt.hour
    df_input["day"] = df_weather["timestamp"].dt.day
    df_input["month"] = df_weather["timestamp"].dt.month
    df_input["dow"] = df_weather["timestamp"].dt.dayofweek

    # Derived mock features
    df_input["sunlighttime"] = np.clip(12 + np.sin((df_input["hour"] - 6) * np.pi / 12) * 4, 0, 16)
    df_input["daylength"] = np.clip(10 + np.sin((df_input["month"] - 6) * np.pi / 12) * 2, 8, 14)
    df_input["efficiency"] = np.clip(np.random.normal(0.9, 0.03, len(df_input)), 0.7, 0.95)
    df_input["ac_lag_1"] = np.random.uniform(1000, 3000, len(df_input))
    df_input["ac_lag_3"] = np.random.uniform(1000, 3000, len(df_input))
    df_input["ac_roll3"] = np.random.uniform(1000, 3000, len(df_input))

    # Drop timestamp
    if "timestamp" in df_input.columns:
        df_input = df_input.drop(columns=["timestamp"])

    # Fill missing model features with 0
    for f in model_features:
        if f not in df_input.columns:
            df_input[f] = 0

    # Keep and order columns as per model
    df_input = df_input.reindex(columns=model_features, fill_value=0)

    return df_input