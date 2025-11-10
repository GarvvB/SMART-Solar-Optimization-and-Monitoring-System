import pandas as pd
import numpy as np
import os

def load_and_merge_data(gen_path, weather_path):
    print("🔹 Loading generation and weather data...")

    # ----------------------------
    # Load Solar Generation Data
    # ----------------------------
    gen = pd.read_csv(gen_path)
    gen.columns = gen.columns.str.lower()
    gen['date_time'] = pd.to_datetime(gen['date_time'], dayfirst=True, errors='coerce')
    gen = gen.dropna(subset=['date_time']).reset_index(drop=True)
    gen['timestamp'] = gen['date_time']

    # ----------------------------
    # Load Weather Data
    # ----------------------------
    weather = pd.read_csv(weather_path)
    weather.columns = weather.columns.str.lower().str.strip()
    
    # Identify timestamp column
    time_col = next((c for c in ['time', 'date_time', 'timestamp'] if c in weather.columns), None)
    if time_col is None:
        raise KeyError("❌ Weather CSV must contain a time/date_time/timestamp column.")
    
    weather['timestamp'] = pd.to_datetime(weather[time_col], dayfirst=True, errors='coerce')
    invalid_count = weather['timestamp'].isna().sum()
    if invalid_count > 0:
        print(f"⚠️ Dropping {invalid_count} weather rows with invalid timestamps...")
        weather = weather.dropna(subset=['timestamp'])

    weather = weather.sort_values('timestamp')

    # ----------------------------
    # Merge with nearest timestamps
    # ----------------------------
    weather_feats = [
        'ghi', 'temp', 'pressure', 'humidity', 'wind_speed', 'rain_1h',
        'clouds_all', 'issun', 'sunlighttime', 'daylength',
        'sunlighttime_daylength', 'weather_type'
    ]
    weather_feats = [f for f in weather_feats if f in weather.columns]

    merged = pd.merge_asof(
        gen.sort_values('timestamp'),
        weather[['timestamp'] + weather_feats],
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('30min')
    )

    if weather_feats:
        merged[weather_feats] = merged[weather_feats].ffill().bfill()

    print(f"✅ Merge successful. Final merged rows: {len(merged)}")

    # ----------------------------
    # Feature Engineering
    # ----------------------------
    merged['efficiency'] = np.where(
        merged['dc_power'] != 0, merged['ac_power'] / merged['dc_power'], 0
    )

    merged['hour'] = merged['timestamp'].dt.hour
    merged['day'] = merged['timestamp'].dt.day
    merged['month'] = merged['timestamp'].dt.month
    merged['dow'] = merged['timestamp'].dt.dayofweek

    merged = merged.sort_values(['source_key', 'timestamp'])
    merged['ac_lag_1'] = merged.groupby('source_key')['ac_power'].shift(1)
    merged['ac_lag_3'] = merged.groupby('source_key')['ac_power'].shift(3)
    merged['ac_roll3'] = (
        merged.groupby('source_key')['ac_power'].shift(1)
        .rolling(window=3, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )

    merged[['ac_lag_1', 'ac_lag_3', 'ac_roll3']] = merged[['ac_lag_1', 'ac_lag_3', 'ac_roll3']].fillna(0)
    merged = merged.dropna(subset=['ac_power']).reset_index(drop=True)

    return merged
