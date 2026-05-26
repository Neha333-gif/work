# -*- coding: utf-8 -*-
"""
Weather Forecasting - Unified ML Pipeline and FastAPI Backend
Trains RandomForestRegressor pipelines on weather and climate data,
generates beautiful visualizations, saves model artifacts, and
serves real-time weather predictions via FastAPI.
"""

import os
import io
import sys
import argparse
import warnings

import pandas as pd
import numpy as np
import joblib

# Use Non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error
)
from imblearn.over_sampling import SMOTE

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# File Paths & Config
# ──────────────────────────────────────────────────────────────────────
DATA_DIR     = "weather_forecasting_data"
DATA_PATH    = os.path.join(DATA_DIR, "weather_forecast.csv")
RESULTS_DIR  = "results"
MODEL_FILE   = "weather_forecasting_model.joblib"
METADATA_FILE = "weather_forecasting_metadata.joblib"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# Dataset Generator Helper
# ──────────────────────────────────────────────────────────────────────
def generate_synthetic_data(file_path):
    print("[INFO] Generating synthetic weather forecast dataset...")
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sequential dates
    dates = pd.date_range(start="2025-01-01", periods=n_samples, freq="D").strftime("%Y-%m-%d")
    
    # Regions
    regions = np.random.choice(["North", "South", "East", "West"], size=n_samples)
    
    # Seasons based on month
    seasons = []
    for d in dates:
        month = int(d.split("-")[1])
        if month in [12, 1, 2]:
            seasons.append("Winter")
        elif month in [3, 4, 5]:
            seasons.append("Spring")
        elif month in [6, 7, 8]:
            seasons.append("Summer")
        else:
            seasons.append("Fall")
            
    # Atmospheric pressure (normally around 1013 hPa)
    pressure = np.random.normal(loc=1012, scale=8, size=n_samples)
    
    # Humidity (percentage)
    humidity = np.random.normal(loc=65, scale=15, size=n_samples)
    humidity = np.clip(humidity, 20, 100)
    
    # Wind speed (km/h)
    wind_speed = np.random.normal(loc=15, scale=7, size=n_samples)
    wind_speed = np.clip(wind_speed, 0, 60)
    
    # Temperature based on season + noise + pressure/humidity
    temperature = []
    for i in range(n_samples):
        season = seasons[i]
        reg = regions[i]
        
        # Base temp by season
        if season == "Summer":
            base = 30
        elif season == "Winter":
            base = 8
        elif season == "Spring":
            base = 18
        else:
            base = 15
            
        # Regional adjustments
        if reg == "South":
            base += 4
        elif reg == "North":
            base -= 4
            
        # Noise + pressure effect
        temp = base + np.random.normal(loc=0, scale=3) + (pressure[i] - 1012) * 0.1
        temperature.append(round(temp, 1))
        
    temperature = np.array(temperature)
    
    # Outlook based on humidity and pressure
    outlooks = []
    rainfall = []
    for i in range(n_samples):
        h = humidity[i]
        p = pressure[i]
        
        rain_prob = 0.1
        if h > 80:
            rain_prob += 0.4
        if p < 1008:
            rain_prob += 0.4
            
        rand = np.random.random()
        if rand < rain_prob:
            outlooks.append("Rainy")
            rainfall.append(round(np.random.exponential(scale=15) + 2, 1))
        elif rand < rain_prob + 0.25:
            outlooks.append("Overcast")
            rainfall.append(round(np.random.uniform(0, 3), 1))
        else:
            outlooks.append("Sunny")
            rainfall.append(0.0)
            
    rainfall = np.array(rainfall)
    
    # Windy status
    windy = ["Strong" if w > 20 else "Weak" for w in wind_speed]
    
    # Play (suitable for outdoor activity)
    play = []
    for i in range(n_samples):
        t = temperature[i]
        h = humidity[i]
        out = outlooks[i]
        w = wind_speed[i]
        
        if (15 <= t <= 32) and (out != "Rainy") and (w < 25) and (40 <= h <= 80):
            play.append("Yes")
        else:
            if np.random.random() < 0.15:
                play.append("Yes")
            else:
                play.append("No")
                
    df = pd.DataFrame({
        "Date": dates,
        "Region": regions,
        "Season": seasons,
        "Temperature": temperature,
        "Humidity": humidity,
        "Wind Speed": wind_speed,
        "Atmospheric Pressure": pressure,
        "Rainfall": rainfall,
        "Outlook": outlooks,
        "Windy": windy,
        "Play": play
    })
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"[INFO] Synthetic weather dataset created at {file_path}")

# ──────────────────────────────────────────────────────────────────────
# 1. ML Pipeline Training & Visualization
# ──────────────────────────────────────────────────────────────────────

def train_weather_model():
    print("[INFO] Starting Weather Forecasting Model Training...")

    # If dataset doesn't exist or is the simple 14-row file, generate the rich synthetic dataset
    if not os.path.exists(DATA_PATH) or (os.path.exists(DATA_PATH) and len(pd.read_csv(DATA_PATH)) < 20):
        if os.path.exists(DATA_PATH):
            orig_backup = os.path.join(DATA_DIR, "weather_forecast_original.csv")
            if not os.path.exists(orig_backup):
                os.rename(DATA_PATH, orig_backup)
                print(f"[INFO] Backed up original 14-row dataset to {orig_backup}")
        generate_synthetic_data(DATA_PATH)

    # ── Load ────────────────────────────────────────────────────────────
    print("[INFO] Loading weather and climate dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # Save a copy of training data for trend plotting
    df_raw = df.copy()

    # ── Preprocessing ───────────────────────────────────────────────────
    print("[INFO] Preprocessing data...")

    # Save target variable y before scaling/encoding features
    df['Play'] = LabelEncoder().fit_transform(df['Play'])
    y = df['Play']
    
    # Feature matrix X
    drop_cols = [c for c in ['Date', 'Play'] if c in df.columns]
    X = df.drop(columns=drop_cols)

    # Handle missing values
    imputer = SimpleImputer(strategy='most_frequent')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Encode categorical columns and scale numerical columns
    le_map = {}
    cat_cols = []
    num_cols = []
    X_processed = X_imputed.copy()

    for col in X_processed.columns:
        if X_processed[col].dtype == 'object' or col in ['Region', 'Season', 'Outlook', 'Windy']:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            le_map[col] = le
            cat_cols.append(col)
        else:
            scaler = StandardScaler()
            X_processed[col] = scaler.fit_transform(X_processed[col].values.reshape(-1, 1))
            le_map[col] = scaler
            num_cols.append(col)

    feature_names = list(X_processed.columns)

    # Apply SMOTE to handle target class imbalance
    print("[INFO] Applying SMOTE to balance play suitability classes...")
    smote = SMOTE(random_state=42, k_neighbors=4)
    X_resampled, y_resampled = smote.fit_resample(X_processed, y)

    # Split train/test
    x_train, x_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )
    print(f"  Training shape: {x_train.shape}")
    print(f"  Testing shape:  {x_test.shape}")

    # ── Model ───────────────────────────────────────────────────────────
    print("[INFO] Training RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=42
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"  R² Score : {r2:.4f}")
    print(f"  MAE      : {mae:.4f}")
    print(f"  MSE      : {mse:.4f}")
    print(f"  RMSE     : {rmse:.4f}")

    # ── Extra Forecast Models (Temperature & Rainfall) ─────────────────────
    print("[INFO] Training extra forecasting models for Temperature & Rainfall...")
    # Temperature Target
    y_temp = df['Temperature']
    X_temp = X_processed.drop(columns=['Temperature'])
    xt_train, xt_test, yt_train, yt_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    model_temp = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model_temp.fit(xt_train, yt_train)
    yt_pred = model_temp.predict(xt_test)
    temp_r2 = r2_score(yt_test, yt_pred)

    # Rainfall Target
    y_rain = df['Rainfall']
    X_rain = X_processed.drop(columns=['Rainfall'])
    xr_train, xr_test, yr_train, yr_test = train_test_split(X_rain, y_rain, test_size=0.2, random_state=42)
    model_rain = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model_rain.fit(xr_train, yr_train)
    yr_pred = model_rain.predict(xr_test)
    rain_r2 = r2_score(yr_test, yr_pred)

    # Save metrics text report
    metrics_text = f"""Weather Forecasting - Model Evaluation Report
=================================================
Play Suitability Index (Weather Fit) Model:
  R² Score : {r2:.4f}   ({r2 * 100:.2f}%)
  MAE      : {mae:.4f}
  MSE      : {mse:.4f}
  RMSE     : {rmse:.4f}

Forecasted Temperature Model:
  R² Score : {temp_r2:.4f}
  MAE      : {mean_absolute_error(yt_test, yt_pred):.4f} °C
  RMSE     : {np.sqrt(mean_squared_error(yt_test, yt_pred)):.4f} °C

Forecasted Rainfall Model:
  R² Score : {rain_r2:.4f}
  MAE      : {mean_absolute_error(yr_test, yr_pred):.4f} mm
  RMSE     : {np.sqrt(mean_squared_error(yr_test, yr_pred)):.4f} mm

Model Type : RandomForestRegressor
Features   : {feature_names}
Samples    : {len(df)} (train: {len(x_train)}, test: {len(x_test)})
"""
    with open(os.path.join(RESULTS_DIR, "accuracy_results.txt"), "w", encoding="utf-8") as f:
        f.write(metrics_text)
    print(f"[INFO] Saved metrics report -> {RESULTS_DIR}/accuracy_results.txt")

    # ── Visualizations ──────────────────────────────────────────────────
    sns.set_theme(style="darkgrid", palette="muted")
    plt.rcParams.update({
        'figure.facecolor': '#070f1e',
        'axes.facecolor':   '#0d1b2e',
        'text.color':       '#f8fafc',
        'axes.labelcolor':  '#94a3b8',
        'xtick.color':      '#94a3b8',
        'ytick.color':      '#94a3b8',
        'axes.edgecolor':   '#1e3a5f',
        'grid.color':       '#1e3a5f',
    })

    # 1. Temperature Trend Analysis
    print("[INFO] Generating temperature trend analysis...")
    fig, ax = plt.subplots(figsize=(12, 5))
    df_sorted = df_raw.sort_values("Date").reset_index()
    ax.plot(df_sorted.index[:180], df_sorted['Temperature'][:180], color='#f43f5e', lw=2, label='Daily Temperature')
    ax.fill_between(df_sorted.index[:180], df_sorted['Temperature'][:180], color='#f43f5e', alpha=0.15)
    ax.set_title("Temperature Trend Analysis (180-Day Timeline)", fontsize=14, color='#f8fafc', pad=15)
    ax.set_xlabel("Timeline Days", fontsize=12)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "temperature_trend.png"), dpi=150, facecolor='#070f1e')
    plt.close()

    # 2. Humidity Distribution Plot
    print("[INFO] Generating humidity distribution plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df_raw['Humidity'], bins=30, kde=True, color='#06b6d4', ax=ax, edgecolor='#0891b2')
    ax.set_title("Humidity Distribution Profile", fontsize=14, color='#f8fafc', pad=15)
    ax.set_xlabel("Relative Humidity (%)", fontsize=12)
    ax.set_ylabel("Density / Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "humidity_distribution.png"), dpi=150, facecolor='#070f1e')
    plt.close()

    # 3. Correlation Heatmap
    print("[INFO] Generating correlation heatmap...")
    fig, ax = plt.subplots(figsize=(10, 8))
    # Select numeric columns for correlation
    corr_cols = ['Temperature', 'Humidity', 'Wind Speed', 'Atmospheric Pressure', 'Rainfall', 'Play']
    corr = df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm", fmt=".2f",
                linewidths=0.5, annot_kws={"size": 10}, ax=ax)
    ax.set_title("Weather & Climate Correlation Heatmap", fontsize=14, color='#f8fafc', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "correlation_heatmap.png"), dpi=150, facecolor='#070f1e')
    plt.close()

    # 4. Rainfall/precipitation visualization
    print("[INFO] Generating rainfall/precipitation visualization...")
    fig, ax = plt.subplots(figsize=(10, 6))
    rain_season = df_raw.groupby('Season')['Rainfall'].mean().reset_index()
    sns.barplot(data=rain_season, x='Season', y='Rainfall', palette=['#3b82f6', '#10b981', '#f59e0b', '#ef4444'], ax=ax)
    ax.set_title("Average Precipitation / Rainfall by Season", fontsize=14, color='#f8fafc', pad=15)
    ax.set_xlabel("Season", fontsize=12)
    ax.set_ylabel("Average Rainfall (mm)", fontsize=12)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f} mm", (p.get_x() + p.get_width() / 2., p.get_height() + 0.2),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', color='#cbd5e1', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "rainfall_visualization.png"), dpi=150, facecolor='#070f1e')
    plt.close()

    # 5. Actual vs Predicted Weather Graph (for Temperature)
    print("[INFO] Generating actual vs predicted weather graph...")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(yt_test, yt_pred, alpha=0.6, color='#3b82f6', s=25, edgecolor='#1d4ed8')
    lims = [min(yt_test.min(), yt_pred.min()), max(yt_test.max(), yt_pred.max())]
    ax.plot(lims, lims, '--', color='#ef4444', lw=2.5, label='Perfect Fit')
    ax.set_title("Actual vs. Predicted Temperatures", fontsize=14, color='#f8fafc', pad=15)
    ax.set_xlabel("Actual Temperature (°C)", fontsize=12)
    ax.set_ylabel("Predicted Temperature (°C)", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "actual_vs_predicted.png"), dpi=150, facecolor='#070f1e')
    plt.close()

    # 6. Residual/Error Distribution Plot
    print("[INFO] Generating residual error distribution plot...")
    residuals = np.array(yt_test) - yt_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(residuals, bins=30, kde=True, color='#f59e0b', edgecolor='#d97706', ax=ax)
    ax.axvline(0, color='#ef4444', linestyle='--', lw=2, label='Zero Error')
    ax.set_title("Temperature Prediction Residuals (Error Distribution)", fontsize=14, color='#f8fafc', pad=15)
    ax.set_xlabel("Residual (Actual – Predicted Temp)", fontsize=12)
    ax.set_ylabel("Density / Count", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "residuals_distribution.png"), dpi=150, facecolor='#070f1e')
    plt.close()

    # 7. Feature Importance Plot
    print("[INFO] Generating feature importance plot...")
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#3b82f6' if v < importances.median() else '#10b981' for v in importances]
    ax.barh(importances.index, importances.values, color=colors)
    ax.set_title("Feature Importance – Drivers of Weather Suitability (Play)", fontsize=14, color='#f8fafc', pad=15)
    ax.set_xlabel("Importance Weight", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=150, facecolor='#070f1e')
    plt.close()

    # 8. Forecast Trend Visualization
    print("[INFO] Generating forecast trend visualization...")
    # Rolling average of daily temperature
    rolling_temp = df_sorted['Temperature'].rolling(window=30, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_sorted.index[:365], df_sorted['Temperature'][:365], color='#f43f5e', alpha=0.3, label='Daily Temp')
    ax.plot(df_sorted.index[:365], rolling_temp[:365], color='#10b981', lw=3, label='30-Day Moving Average')
    ax.set_title("Long-Term Temperature Forecast Trend (1-Year Cycle)", fontsize=14, color='#f8fafc', pad=15)
    ax.set_xlabel("Day of Year", fontsize=12)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "forecast_trend.png"), dpi=150, facecolor='#070f1e')
    plt.close()

    # 9. Model Performance Metrics Visualization
    print("[INFO] Generating model performance metrics visualization...")
    metric_names  = ['Play R²', 'Temp R²', 'Rain R²']
    metric_values = [r2, temp_r2, rain_r2]
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(metric_names, metric_values, color=['#10b981', '#3b82f6', '#8b5cf6'], width=0.45)
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                f"{val:.3f}", ha='center', fontweight='bold', color='#cbd5e1', fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Variance Explained (R² Comparison)", fontsize=14, color='#f8fafc', pad=15)
    ax.set_ylabel("R² Metric Value", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_metrics.png"), dpi=150, facecolor='#070f1e')
    plt.close()

    # ── Save Model Assets ───────────────────────────────────────────────
    print("[INFO] Saving model assets...")
    # Save all models in a dictionary
    models_dict = {
        "model_play": model,
        "model_temp": model_temp,
        "model_rain": model_rain
    }
    joblib.dump(models_dict, MODEL_FILE)
    
    metadata = {
        "feature_names": feature_names,
        "label_encoders": le_map,
        "r2":   r2,
        "mae":  mae,
        "mse":  mse,
        "rmse": rmse,
        "temp_r2": temp_r2,
        "rain_r2": rain_r2,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
    }
    joblib.dump(metadata, METADATA_FILE)
    print(f"  Saved -> {MODEL_FILE}")
    print(f"  Saved -> {METADATA_FILE}")
    print("[DONE] Weather Forecasting training complete!")

# ──────────────────────────────────────────────────────────────────────
# 2. FastAPI Web Server
# ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Weather Forecasting API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global assets
_models    = None
_metadata = None


def load_weather_assets():
    global _models, _metadata
    try:
        if os.path.exists(MODEL_FILE) and os.path.exists(METADATA_FILE):
            _models   = joblib.load(MODEL_FILE)
            _metadata = joblib.load(METADATA_FILE)
            print("[INFO] Weather forecasting model assets loaded successfully.")
        else:
            print("[WARNING] Model assets not found. Run with --train first.")
    except Exception as e:
        print(f"[ERROR] Failed to load weather model assets: {e}")


load_weather_assets()


class WeatherPredictionRequest(BaseModel):
    temperature:          float = Field(22.5, description="Temperature (°C)")
    humidity:             float = Field(60.0, description="Humidity (%)")
    wind_speed:           float = Field(12.5, description="Wind Speed (km/h)")
    atmospheric_pressure: float = Field(1012.0, description="Atmospheric Pressure (hPa)")
    rainfall:             float = Field(0.0, description="Rainfall / Precipitation (mm)")
    region:               str   = Field("North", description="Geographic region")
    date:                 str   = Field("2026-05-26", description="Forecast Date (YYYY-MM-DD)")
    season:               str   = Field("Summer", description="Season")


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serve the main weather forecasting frontend UI."""
    frontend_path = "weather_forecasting_frontend.html"
    if os.path.exists(frontend_path):
        with open(frontend_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h3>Frontend file not found.</h3>", status_code=404)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok" if _models is not None else "no_model_loaded",
        "model": "RandomForestRegressor (Multi-Target)",
        "assets_exist": {
            "model":    os.path.exists(MODEL_FILE),
            "metadata": os.path.exists(METADATA_FILE),
        },
    }


@app.post("/predict")
def predict_weather(req: WeatherPredictionRequest):
    """Predict play suitability, forecasted temperature, and forecasted rainfall."""
    if _models is None or _metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run with --train first.")

    try:
        feature_names  = _metadata["feature_names"]
        le_map         = _metadata["label_encoders"]
        cat_cols       = _metadata["cat_cols"]

        # Derive Windy variable from wind speed
        derived_windy = "Strong" if req.wind_speed >= 20.0 else "Weak"
        
        # Derive Outlook variable from rainfall and humidity
        if req.rainfall > 5.0:
            derived_outlook = "Rainy"
        elif req.humidity > 80.0:
            derived_outlook = "Overcast"
        else:
            derived_outlook = "Sunny"

        # Build raw feature row
        raw = {
            "Region":               req.region,
            "Season":               req.season,
            "Temperature":          req.temperature,
            "Humidity":             req.humidity,
            "Wind Speed":           req.wind_speed,
            "Atmospheric Pressure": req.atmospheric_pressure,
            "Rainfall":             req.rainfall,
            "Outlook":              derived_outlook,
            "Windy":                derived_windy
        }

        # Encode and scale features using saved encoders
        processed = {}
        for col in feature_names:
            val = raw[col]
            if col in cat_cols:
                le = le_map[col]
                # LabelEncoder
                if str(val) in le.classes_:
                    processed[col] = float(le.transform([str(val)])[0])
                else:
                    processed[col] = float(len(le.classes_) // 2)
            else:
                # StandardScaler
                scaler = le_map[col]
                processed[col] = float(scaler.transform([[float(val)]])[0][0])

        # Build feature vector in correct order
        row = [processed[f] for f in feature_names]
        X_input = np.array(row, dtype=float).reshape(1, -1)

        # 1. Predict Play suitability (0 or 1 continuous)
        model_play = _models["model_play"]
        pred_play_score = float(model_play.predict(X_input)[0])
        pred_play_label = "Yes" if pred_play_score >= 0.5 else "No"

        # Confidence bounds for play index
        tree_preds = np.array([t.predict(X_input)[0] for t in model_play.estimators_])
        lower = max(0.0, round(float(tree_preds.mean() - tree_preds.std()), 2))
        upper = min(1.0, round(float(tree_preds.mean() + tree_preds.std()), 2))

        # 2. Predict Forecasted Temperature
        model_temp = _models["model_temp"]
        # Feature row excluding Temperature
        row_temp = [processed[f] for f in feature_names if f != 'Temperature']
        pred_temp = float(model_temp.predict(np.array(row_temp).reshape(1, -1))[0])

        # 3. Predict Forecasted Rainfall
        model_rain = _models["model_rain"]
        # Feature row excluding Rainfall
        row_rain = [processed[f] for f in feature_names if f != 'Rainfall']
        pred_rain = float(model_rain.predict(np.array(row_rain).reshape(1, -1))[0])
        pred_rain = max(0.0, pred_rain)

        # Map derived weather condition
        return {
            "predicted_play":           pred_play_label,
            "play_suitability_score":   round(pred_play_score * 100, 1),
            "play_lower_bound":         round(lower * 100, 1),
            "play_upper_bound":         round(upper * 100, 1),
            "predicted_outlook":        derived_outlook,
            "forecasted_temperature":   round(pred_temp, 1),
            "forecasted_rainfall":      round(pred_rain, 1),
            "model_r2":                 round(_metadata["r2"], 4),
            "model_rmse":               round(_metadata["rmse"], 4),
            "region":                   req.region,
            "season":                   req.season,
            "date":                     req.date
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/results/metrics")
def get_metrics():
    """Return metrics report and plot paths."""
    metrics_file = os.path.join(RESULTS_DIR, "accuracy_results.txt")
    if not os.path.exists(metrics_file):
        raise HTTPException(status_code=404, detail="Metrics not found. Train the model first.")
    with open(metrics_file, "r") as f:
        content = f.read()
    return {
        "text_report": content,
        "plots": {
            "temperature_trend":   "/results/image/temperature_trend.png",
            "humidity_distribution": "/results/image/humidity_distribution.png",
            "correlation_heatmap":  "/results/image/correlation_heatmap.png",
            "rainfall_visualization": "/results/image/rainfall_visualization.png",
            "actual_vs_predicted":  "/results/image/actual_vs_predicted.png",
            "residuals_distribution": "/results/image/residuals_distribution.png",
            "feature_importance":   "/results/image/feature_importance.png",
            "forecast_trend":       "/results/image/forecast_trend.png",
            "model_metrics":        "/results/image/model_metrics.png",
        },
    }


@app.get("/results/image/{image_name}")
def serve_plot(image_name: str):
    """Serve a generated plot image."""
    path = os.path.join(RESULTS_DIR, image_name)
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Image not found.")


def _background_retrain():
    print("[RETRAIN] Starting background model retraining...")
    try:
        train_weather_model()
        load_weather_assets()
        print("[RETRAIN] Model retrained successfully.")
    except Exception as e:
        print(f"[RETRAIN] Error: {e}")


@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    """Retrain the weather forecasting model in the background."""
    background_tasks.add_task(_background_retrain)
    return {"status": "accepted", "message": "Model retraining initiated in the background."}


# ──────────────────────────────────────────────────────────────────────
# 3. Main Execution Block
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weather Forecasting unified ML pipeline and backend")
    parser.add_argument("--train", action="store_true", help="Force train the model and regenerate results")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI port (default: 8000)")
    args = parser.parse_args()

    if args.train or not os.path.exists(MODEL_FILE) or not os.path.exists(METADATA_FILE):
        try:
            train_weather_model()
            load_weather_assets()
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            sys.exit(1)

    if "--train" not in sys.argv:
        import uvicorn
        print(f"[INFO] Starting Weather Forecasting API server on port {args.port}...")
        uvicorn.run("weather_forecasting:app", host="127.0.0.1", port=args.port, reload=False)
