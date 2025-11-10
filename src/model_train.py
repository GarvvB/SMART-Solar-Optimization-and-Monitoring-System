import joblib
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.data_preprocess import load_and_merge_data

# ---------------------- METRIC FUNCTION ----------------------
def evaluate_model(y_true, y_pred, model_name):
    """Compute and return MAE, RMSE, R2 for a model."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"✅ {model_name} -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

# ---------------------- TRAINING FUNCTION ----------------------
def train_and_save_models(gen_path, weather_path):
    print("🔹 Loading generation and weather data...")
    df = load_and_merge_data(gen_path, weather_path)

    # Ensure required columns exist
    if "AC_POWER" not in df.columns:
        if "ac_power" in df.columns:
            df.rename(columns={"ac_power": "AC_POWER"}, inplace=True)
        else:
            raise KeyError("Neither 'AC_POWER' nor 'ac_power' found in dataset.")

    # Define features
    features = [
        "DC_POWER", "efficiency", "hour", "day", "month", "dow",
        "ac_lag_1", "ac_lag_3", "ac_roll3",
        "ghi", "temp", "pressure", "humidity", "wind_speed",
        "rain_1h", "clouds_all", "issun", "sunlighttime", "daylength"
    ]
    features = [f for f in features if f in df.columns]
    target = "AC_POWER"

    # Split data
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    os.makedirs("models", exist_ok=True)
    metrics = {}

    # -------------------- Linear Regression --------------------
    print("\n⚙️ Training Linear Regression...")
    try:
        lr_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LinearRegression())
        ])
        lr_pipeline.fit(X_train, y_train)
        y_pred_lr = lr_pipeline.predict(X_test)
        metrics["Linear Regression"] = evaluate_model(y_test, y_pred_lr, "Linear Regression")
        joblib.dump(lr_pipeline, "models/linear_with_weather.joblib")
        print("💾 Linear Regression model saved successfully.")
    except Exception as e:
        print(f"❌ Error training Linear Regression: {e}")

    # -------------------- XGBoost --------------------
    print("\n⚙️ Training XGBoost Regressor...")
    try:
        xgb_model = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="hist"
        )
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        metrics["XGBoost"] = evaluate_model(y_test, y_pred_xgb, "XGBoost Regressor")
        joblib.dump(xgb_model, "models/xgb_with_weather.joblib")
        print("💾 XGBoost model saved successfully.")
    except Exception as e:
        print(f"❌ Error training XGBoost: {e}")

    # -------------------- Save Metrics JSON --------------------
    try:
        metrics_path = "models/metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"\n📊 Metrics saved successfully at: {metrics_path}")
    except Exception as e:
        print(f"❌ Failed to save metrics: {e}")

    print("\n✅ All models and metrics saved successfully.")


# ---------------------- LOADER FUNCTION ----------------------
# --- at the top of src/model_train.py ---
import os
from pathlib import Path
import joblib
import json

def load_trained_models(model_dir: str | None = None):
    """
    Load LR and XGB models from the models directory.
    Falls back to <repo_root>/models if model_dir is None.
    """
    if model_dir is None:
        ROOT = Path(__file__).resolve().parents[1]  # <repo_root>
        model_dir = os.fspath(ROOT / "models")

    print(f"🔍 Loading models from: {model_dir}")

    lr_path  = os.path.join(model_dir, "linear_with_weather.joblib")
    xgb_path = os.path.join(model_dir, "xgb_with_weather.joblib")
    metrics_path = os.path.join(model_dir, "metrics.json")

    # Load models
    lr_model = joblib.load(lr_path)
    xgb_model = joblib.load(xgb_path)

    # Load metrics if present (optional)
    try:
        with open(metrics_path, "r") as f:
            _ = json.load(f)
    except Exception:
        pass

    return lr_model, xgb_model


# ---------------------- MAIN ENTRY POINT ----------------------
if __name__ == "__main__":
    gen_path = "data/Plant_1_Generation_Data.csv"
    weather_path = "data/solar_weather.csv"
    train_and_save_models(gen_path, weather_path)
