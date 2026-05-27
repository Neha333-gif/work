# -*- coding: utf-8 -*-
"""
Energy Consumption Prediction - Unified ML Pipeline and FastAPI Backend
Trains a RandomForestRegressor on Energy_consumption_dataset.csv,
generates visualizations, saves model artifacts, and serves
real-time energy consumption predictions via FastAPI.

Run:
    python energy_consumption_prediction.py --train   # train + start server
    python energy_consumption_prediction.py           # start server only
"""

import argparse
import os
import sys
import warnings

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# ──────────────────────────────────────────────────────────────────────────────
# File Paths & Config
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR = "energy_consumption_prediction_data"
DATA_PATH = os.path.join(DATA_DIR, "Energy_consumption_dataset.csv")
RESULTS_DIR = "results"
MODEL_FILE = "energy_consumption_prediction_model.joblib"
METADATA_FILE = "energy_consumption_prediction_metadata.joblib"
FRONTEND_FILE = "energy_consumption_prediction_frontend.html"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Plot theme constants
# ──────────────────────────────────────────────────────────────────────────────

DARK_BG = '#070f1e'
SURFACE = '#0d1b2e'
COLOR_P = '#6366f1'
COLOR_A = '#10b981'
COLOR_W = '#f59e0b'
COLOR_D = '#f43f5e'
TEXT_COL = '#f8fafc'
MUTED_COL = '#94a3b8'


def _set_plot_theme():
    plt.rcParams.update({
        'figure.facecolor': DARK_BG,
        'axes.facecolor': SURFACE,
        'text.color': TEXT_COL,
        'axes.labelcolor': MUTED_COL,
        'xtick.color': MUTED_COL,
        'ytick.color': MUTED_COL,
        'axes.edgecolor': '#1e3a5f',
        'grid.color': '#1e3a5f',
        'axes.grid': True,
        'grid.alpha': 0.25,
    })


# ──────────────────────────────────────────────────────────────────────────────
# 1. ML Training Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def train_energy_model():
    print("[INFO] Starting Energy Consumption Model Training...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_PATH}'. Please ensure Energy_consumption_dataset.csv is in {DATA_DIR}/"
        )

    print("[INFO] Loading energy consumption dataset...")
    df_raw = pd.read_csv(DATA_PATH)
    print(f"  Dataset shape        : {df_raw.shape}")
    print(f"  Columns available    : {list(df_raw.columns)}")

    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]

    # Map common categorical markers to numeric values
    mapping = {
        "Holiday": {"No": 0, "Yes": 1, "False": 0, "True": 1},
        "HVACUsage": {"Off": 0, "On": 1},
        "LightingUsage": {"Off": 0, "On": 1},
    }
    for col, map_values in mapping.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(map_values).fillna(df[col])

    if 'DayOfWeek' in df.columns:
        df['DayOfWeek'] = df['DayOfWeek'].astype(str)

    if 'EnergyConsumption' not in df.columns:
        raise KeyError("Required target column 'EnergyConsumption' not found in the dataset.")

    # Clean and prepare features
    target = 'EnergyConsumption'
    X = df.drop(columns=[target])
    y = df[target].astype(float).copy()

    # Encode non-numeric categorical features
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    X = X.apply(pd.to_numeric, errors='coerce')
    X.fillna(X.median(numeric_only=True), inplace=True)
    y.fillna(y.mean(), inplace=True)
    X.drop_duplicates(inplace=True)

    feature_names = list(X.columns)
    print(f"  Processed features   : {feature_names}")
    print(f"  Training rows        : {len(X)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"  Train/test split     : {x_train.shape[0]}/{x_test.shape[0]}")

    print("[INFO] Training RandomForestRegressor (n_estimators=120, max_depth=12)...")
    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n  MAE  : {mae:.3f}")
    print(f"  MSE  : {mse:.3f}")
    print(f"  RMSE : {rmse:.3f}")
    print(f"  R2   : {r2:.3f}\n")

    report_text = f"""Energy Consumption Prediction - Model Evaluation Report
=========================================================
Model     : RandomForestRegressor (n_estimators=120, max_depth=12)
Dataset   : Energy_consumption_dataset.csv ({len(X)} samples)
Features  : {feature_names}

Performance Metrics
-------------------
MAE       : {mae:.3f}
MSE       : {mse:.3f}
RMSE      : {rmse:.3f}
R2 Score  : {r2:.3f}

Top 5 Predictions
------------------
"""
    for actual, predicted in list(zip(y_test.tolist(), y_pred.tolist()))[:5]:
        report_text += f"Actual={actual:.3f}, Predicted={predicted:.3f}\n"

    with open(os.path.join(RESULTS_DIR, "accuracy_results.txt"), "w", encoding="utf-8") as fh:
        fh.write(report_text)
    print(f"[INFO] Saved -> {RESULTS_DIR}/accuracy_results.txt")

    prediction_report = pd.DataFrame({
        'actual_consumption': y_test.values,
        'predicted_consumption': y_pred,
        'error': (y_test.values - y_pred),
    })
    prediction_report.to_csv(os.path.join(RESULTS_DIR, 'predictions_report.csv'), index=False)
    print(f"[INFO] Saved -> {RESULTS_DIR}/predictions_report.csv")

    _set_plot_theme()
    print("[INFO] Generating energy consumption visualizations...")

    index_series = np.arange(len(y_test))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(index_series, y_test.values, label='Actual', color=COLOR_A, alpha=0.88)
    ax.plot(index_series, y_pred, label='Predicted', color=COLOR_P, alpha=0.88)
    ax.set_title('Actual vs Predicted Energy Consumption', fontsize=14, color=TEXT_COL, pad=15)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Energy Consumption', fontsize=12)
    ax.legend(facecolor=SURFACE, edgecolor='#1e3a5f', labelcolor=TEXT_COL)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'actual_vs_predicted.png'), dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"  Saved -> {RESULTS_DIR}/actual_vs_predicted.png")

    corr = X.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(11, 10))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5,
                annot_kws={'size': 8}, ax=ax)
    ax.set_title('Feature Correlation Heatmap – Energy Usage Data', fontsize=14, color=TEXT_COL, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'correlation_heatmap.png'), dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"  Saved -> {RESULTS_DIR}/correlation_heatmap.png")

    residuals = y_test.values - y_pred
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(residuals, kde=True, color=COLOR_W, edgecolor=None, ax=ax)
    ax.set_title('Residual Distribution – Prediction Errors', fontsize=14, color=TEXT_COL, pad=15)
    ax.set_xlabel('Error (Actual - Predicted)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'residual_distribution.png'), dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"  Saved -> {RESULTS_DIR}/residual_distribution.png")

    feature_importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
    colors_fi = [COLOR_P if v < feature_importances.median() else COLOR_A for v in feature_importances]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(feature_importances.index, feature_importances.values, color=colors_fi, edgecolor='none')
    ax.set_title('Feature Importance – Energy Consumption Drivers', fontsize=14, color=TEXT_COL, pad=15)
    ax.set_xlabel('Importance Score', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'), dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"  Saved -> {RESULTS_DIR}/feature_importance.png")

    if 'DayOfWeek' in df.columns:
        day_avg = df.groupby('DayOfWeek')[target].mean().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
    else:
        day_avg = pd.Series(dtype=float)
    if not day_avg.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(day_avg.index, day_avg.values, marker='o', color=COLOR_A)
        ax.set_title('Average Energy Consumption by Day of Week', fontsize=14, color=TEXT_COL, pad=15)
        ax.set_xlabel('Day of Week', fontsize=12)
        ax.set_ylabel('Average Energy Consumption', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'daily_energy_usage.png'), dpi=150, facecolor=DARK_BG)
        plt.close()
        print(f"  Saved -> {RESULTS_DIR}/daily_energy_usage.png")

    if 'Hour' in df.columns:
        hour_avg = df.groupby('Hour')[target].mean()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hour_avg.index, hour_avg.values, marker='o', color=COLOR_P)
        ax.set_title('Peak Consumption by Hour of Day', fontsize=14, color=TEXT_COL, pad=15)
        ax.set_xlabel('Hour', fontsize=12)
        ax.set_ylabel('Average Energy Consumption', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'peak_consumption.png'), dpi=150, facecolor=DARK_BG)
        plt.close()
        print(f"  Saved -> {RESULTS_DIR}/peak_consumption.png")

    metric_names = ['MAE', 'MSE', 'RMSE', 'R2']
    metric_values = [mae, mse, rmse, r2]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metric_names, metric_values, color=[COLOR_A, COLOR_P, COLOR_W, COLOR_D], edgecolor='none')
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(metric_values) * 0.03,
                f'{val:.3f}', ha='center', fontweight='bold', color=TEXT_COL, fontsize=11)
    ax.set_title('Energy Consumption Prediction – Model Performance Metrics', fontsize=14, color=TEXT_COL, pad=15)
    ax.set_ylabel('Score', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_metrics.png'), dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"  Saved -> {RESULTS_DIR}/model_metrics.png")

    print("[INFO] Saving model artifacts...")
    joblib.dump(model, MODEL_FILE)
    metadata = {
        'feature_names': feature_names,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'cat_cols': cat_cols,
        'metrics': {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
        },
        'feature_importance': feature_importances.to_dict(),
    }
    joblib.dump(metadata, METADATA_FILE)
    print(f"  Saved model    -> {MODEL_FILE}")
    print(f"  Saved metadata -> {METADATA_FILE}")
    print("[DONE] Energy Consumption model training complete!\n")


# ──────────────────────────────────────────────────────────────────────────────
# 2. FastAPI Web Server
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Energy Consumption Prediction API",
    description="AI-powered energy consumption prediction using Random Forest Regression",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_metadata = None


def load_energy_assets():
    global _model, _metadata
    try:
        if os.path.exists(MODEL_FILE) and os.path.exists(METADATA_FILE):
            _model = joblib.load(MODEL_FILE)
            _metadata = joblib.load(METADATA_FILE)
            print("[INFO] Energy model assets loaded successfully.")
        else:
            print("[WARNING] Model assets not found. Run with --train first.")
    except Exception as e:
        print(f"[ERROR] Failed to load model assets: {e}")


load_energy_assets()


class EnergyPredictionRequest(BaseModel):
    Month: int = Field(1, ge=1, le=12, description="Month of the year")
    Hour: int = Field(0, ge=0, le=23, description="Hour of the day")
    DayOfWeek: str = Field("Monday", description="Day of week")
    Holiday: str = Field("No", description="Holiday status (Yes/No)")
    Temperature: float = Field(25.0, description="Temperature in degrees Celsius")
    Humidity: float = Field(45.0, ge=0, le=100, description="Relative humidity percentage")
    SquareFootage: float = Field(1500.0, ge=0, description="Building square footage")
    Occupancy: int = Field(5, ge=0, description="Number of occupants")
    HVACUsage: str = Field("On", description="HVAC usage state")
    LightingUsage: str = Field("Off", description="Lighting usage state")
    RenewableEnergy: float = Field(5.0, ge=0, description="Renewable energy contribution")


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    if os.path.exists(FRONTEND_FILE):
        with open(FRONTEND_FILE, "r", encoding="utf-8") as fh:
            return HTMLResponse(content=fh.read())
    return HTMLResponse(content="<h3>Frontend file not found.</h3>", status_code=404)


@app.get("/health")
def health():
    return {
        "status": "ok" if _model is not None else "no_model_loaded",
        "model": "RandomForestRegressor",
        "domain": "Energy Consumption Prediction",
        "assets_exist": {
            "model": os.path.exists(MODEL_FILE),
            "metadata": os.path.exists(METADATA_FILE),
        },
        "metrics": _metadata.get('metrics') if _metadata else {},
    }


def _encode_row(raw, feature_names, label_encoders, cat_cols):
    mapping = {
        'Holiday':       {'No': 0, 'Yes': 1, 'False': 0, 'True': 1},
        'HVACUsage':     {'Off': 0, 'On': 1},
        'LightingUsage': {'Off': 0, 'On': 1},
    }
    for col, mapper in mapping.items():
        if col in raw:
            raw[col] = mapper.get(str(raw[col]).strip(), raw[col])

    for col in cat_cols:
        if col in raw:
            le = label_encoders.get(col)
            if le is not None:
                value = str(raw[col])
                if value in le.classes_:
                    raw[col] = int(le.transform([value])[0])
                else:
                    raw[col] = int(len(le.classes_) // 2)

    ordered = [raw.get(name, 0) for name in feature_names]
    return np.array(ordered, dtype=float).reshape(1, -1)


@app.post("/predict")
def predict_energy(req: EnergyPredictionRequest):
    if _model is None or _metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run with --train first.")

    try:
        feature_names = _metadata['feature_names']
        scaler = _metadata['scaler']
        label_encoders = _metadata['label_encoders']
        cat_cols = _metadata['cat_cols']

        raw = {
            'Month': req.Month,
            'Hour': req.Hour,
            'DayOfWeek': req.DayOfWeek,
            'Holiday': req.Holiday,
            'Temperature': req.Temperature,
            'Humidity': req.Humidity,
            'SquareFootage': req.SquareFootage,
            'Occupancy': req.Occupancy,
            'HVACUsage': req.HVACUsage,
            'LightingUsage': req.LightingUsage,
            'RenewableEnergy': req.RenewableEnergy,
        }

        X_input = _encode_row(raw, feature_names, label_encoders, cat_cols)
        X_scaled = scaler.transform(X_input)
        predicted = float(_model.predict(X_scaled)[0])

        average_usage = float(np.mean(_model.predict(scaler.transform(scaler.inverse_transform(X_scaled))))) if _model is not None else None
        trend = "High usage expected" if predicted > 80 else "Moderate usage" if predicted > 40 else "Low usage"

        return {
            'predicted_energy_consumption': round(predicted, 3),
            'trend_estimate': trend,
            'prediction_units': 'Energy Consumption Units',
            'metrics': _metadata['metrics'],
            'plots': {
                'actual_vs_predicted': '/results/image/actual_vs_predicted.png',
                'correlation_heatmap': '/results/image/correlation_heatmap.png',
                'feature_importance': '/results/image/feature_importance.png',
                'residual_distribution': '/results/image/residual_distribution.png',
                'daily_energy_usage': '/results/image/daily_energy_usage.png',
                'peak_consumption': '/results/image/peak_consumption.png',
                'model_metrics': '/results/image/model_metrics.png',
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/results/metrics")
def get_metrics():
    metrics_file = os.path.join(RESULTS_DIR, "accuracy_results.txt")
    if not os.path.exists(metrics_file):
        raise HTTPException(status_code=404, detail="Metrics not found. Train the model first.")
    with open(metrics_file, "r", encoding="utf-8") as fh:
        content = fh.read()
    return {
        'text_report': content,
        'plots': {
            'actual_vs_predicted': '/results/image/actual_vs_predicted.png',
            'correlation_heatmap': '/results/image/correlation_heatmap.png',
            'residual_distribution': '/results/image/residual_distribution.png',
            'feature_importance': '/results/image/feature_importance.png',
            'daily_energy_usage': '/results/image/daily_energy_usage.png',
            'peak_consumption': '/results/image/peak_consumption.png',
            'model_metrics': '/results/image/model_metrics.png',
        },
    }


@app.get("/results/image/{image_name}")
def serve_plot(image_name: str):
    path = os.path.join(RESULTS_DIR, image_name)
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail=f"Image '{image_name}' not found.")


def _background_retrain():
    print("[RETRAIN] Starting background energy model retraining...")
    try:
        train_energy_model()
        load_energy_assets()
        print("[RETRAIN] Energy model retrained successfully.")
    except Exception as e:
        print(f"[RETRAIN] Error during retraining: {e}")


@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(_background_retrain)
    return {
        'status': 'accepted',
        'message': 'Energy consumption model retraining initiated in the background.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Energy Consumption Prediction – unified ML pipeline and FastAPI backend'
    )
    parser.add_argument('--train', action='store_true', help='Train the model and regenerate all results')
    parser.add_argument('--port', type=int, default=8000, help='FastAPI server port (default: 8000)')
    args = parser.parse_args()

    if args.train or not os.path.exists(MODEL_FILE) or not os.path.exists(METADATA_FILE):
        try:
            train_energy_model()
            load_energy_assets()
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            sys.exit(1)

    import uvicorn
    print(f"[INFO] Starting Energy Consumption Prediction API server on port {args.port}...")
    print(f"[INFO] Open your browser at: http://127.0.0.1:{args.port}")
    uvicorn.run(
        'energy_consumption_prediction:app',
        host='127.0.0.1',
        port=args.port,
        reload=False,
    )
