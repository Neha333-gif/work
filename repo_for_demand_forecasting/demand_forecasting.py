# -*- coding: utf-8 -*-
"""
Demand Forecasting - Unified ML Pipeline and FastAPI Backend
Trains a RandomForestRegressor pipeline on retail demand data,
generates beautiful visualizations, saves model artifacts, and
serves real-time demand predictions via FastAPI.
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

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# File Paths & Config
# ──────────────────────────────────────────────────────────────────────
DATA_DIR     = "demand_forecasting_data"
DATA_PATH    = os.path.join(DATA_DIR, "demand_forecasting.csv")
RESULTS_DIR  = "results"
MODEL_FILE   = "demand_forecasting_model.joblib"
METADATA_FILE = "demand_forecasting_metadata.joblib"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# 1. ML Pipeline Training & Visualization
# ──────────────────────────────────────────────────────────────────────

def train_demand_model():
    print("[INFO] Starting Demand Forecasting Model Training...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Please ensure demand_forecasting.csv is in demand_forecasting_data/."
        )

    # ── Load ────────────────────────────────────────────────────────────
    print("[INFO] Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # Sample 20 000 rows so training stays fast without losing coverage
    if len(df) > 20000:
        df = df.sample(n=20000, random_state=42).reset_index(drop=True)
    print(f"  Dataset shape (sampled): {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # ── Preprocessing ───────────────────────────────────────────────────
    print("[INFO] Preprocessing data...")

    # Drop high-cardinality / leakage columns
    drop_cols = [c for c in ['Date', 'Units Sold', 'Units Ordered'] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # Encode categorical columns
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    le_map = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_map[col] = le

    # Impute any remaining nulls
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # ── Feature / Target Split ──────────────────────────────────────────
    TARGET = 'Demand'
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    feature_names = list(X.columns)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
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

    # Save metrics text report
    metrics_text = f"""Demand Forecasting - Model Evaluation Report
=================================================
R² Score : {r2:.4f}   ({r2 * 100:.2f}%)
MAE      : {mae:.4f}
MSE      : {mse:.4f}
RMSE     : {rmse:.4f}

Model    : RandomForestRegressor (n_estimators=150, max_depth=12)
Features : {feature_names}
Samples  : {len(df)} (training: {len(x_train)}, test: {len(x_test)})
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

    # 1. Demand Distribution
    print("[INFO] Generating demand distribution plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['Demand'], bins=40, color='#6366f1', edgecolor='#4f46e5', alpha=0.85)
    ax.set_title("Demand Distribution Across All Records", fontsize=14, color='#f8fafc', pad=15)
    ax.set_xlabel("Demand (Units)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sales_distribution.png"), dpi=150, facecolor='#070f1e')
    plt.close()

    # 2. Correlation Heatmap
    print("[INFO] Generating correlation heatmap...")
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm", fmt=".2f",
                linewidths=0.5, annot_kws={"size": 9}, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, color='#f8fafc', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "correlation_heatmap.png"), dpi=150, facecolor='#070f1e')
    plt.close()

    # 3. Feature Importance
    print("[INFO] Generating feature importance plot...")
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#6366f1' if v < importances.median() else '#10b981' for v in importances]
    ax.barh(importances.index, importances.values, color=colors)
    ax.set_title("Feature Importance – Drivers of Demand", fontsize=14, color='#f8fafc', pad=15)
    ax.set_xlabel("Importance Score", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=150, facecolor='#070f1e')
    plt.close()

    # 4. Actual vs Predicted
    print("[INFO] Generating actual vs predicted plot...")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(y_test, y_pred, alpha=0.35, color='#6366f1', s=18)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, '--', color='#f43f5e', lw=2, label='Perfect Forecast')
    ax.set_title("Actual vs Predicted Demand", fontsize=14, color='#f8fafc', pad=15)
    ax.set_xlabel("Actual Demand", fontsize=12)
    ax.set_ylabel("Predicted Demand", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "actual_vs_predicted.png"), dpi=150, facecolor='#070f1e')
    plt.close()

    # 5. Residuals Distribution
    print("[INFO] Generating residuals distribution plot...")
    residuals = np.array(y_test) - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins=40, color='#f59e0b', edgecolor='#d97706', alpha=0.85)
    ax.axvline(0, color='#f43f5e', linestyle='--', lw=2, label='Zero Error')
    ax.set_title("Residual (Error) Distribution", fontsize=14, color='#f8fafc', pad=15)
    ax.set_xlabel("Residual (Actual – Predicted)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "residuals_distribution.png"), dpi=150, facecolor='#070f1e')
    plt.close()

    # 6. Model Metrics Bar Chart
    print("[INFO] Generating model metrics visualization...")
    metric_names  = ['R² Score', 'MAE', 'RMSE']
    metric_values = [r2, mae, rmse]
    norm_values   = [r2, 1 - (mae / max(mae, 1)), 1 - (rmse / max(rmse, 1))]
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(metric_names, metric_values, color=['#10b981', '#6366f1', '#f59e0b'], width=0.5)
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                f"{val:.3f}", ha='center', fontweight='bold', color='#cbd5e1', fontsize=11)
    ax.set_title("Demand Forecasting – Model Performance Metrics", fontsize=14, color='#f8fafc', pad=15)
    ax.set_ylabel("Metric Value", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_metrics.png"), dpi=150, facecolor='#070f1e')
    plt.close()

    # 7. Demand Trend (rolling average over sorted index as proxy)
    print("[INFO] Generating demand trend plot...")
    demand_sample = df['Demand'].reset_index(drop=True)
    rolling = demand_sample.rolling(window=200, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(demand_sample.index[:3000], demand_sample[:3000], color='#6366f1',
            alpha=0.25, lw=1, label='Raw Demand')
    ax.plot(rolling.index[:3000], rolling[:3000], color='#10b981',
            lw=2.5, label='Rolling Average (200)')
    ax.set_title("Demand Trend (Rolling Average)", fontsize=14, color='#f8fafc', pad=15)
    ax.set_xlabel("Record Index", fontsize=12)
    ax.set_ylabel("Demand (Units)", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "demand_trend.png"), dpi=150, facecolor='#070f1e')
    plt.close()

    # ── Save Model Assets ───────────────────────────────────────────────
    print("[INFO] Saving model assets...")
    joblib.dump(model, MODEL_FILE)
    metadata = {
        "feature_names": feature_names,
        "scaler": scaler,
        "label_encoders": le_map,
        "r2":   r2,
        "mae":  mae,
        "mse":  mse,
        "rmse": rmse,
        "cat_cols": cat_cols,
        "drop_cols": drop_cols,
    }
    joblib.dump(metadata, METADATA_FILE)
    print(f"  Saved -> {MODEL_FILE}")
    print(f"  Saved -> {METADATA_FILE}")
    print("[DONE] Demand Forecasting training complete!")

# ──────────────────────────────────────────────────────────────────────
# 2. FastAPI Web Server
# ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Demand Forecasting API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global assets
_model    = None
_metadata = None


def load_demand_assets():
    global _model, _metadata
    try:
        if os.path.exists(MODEL_FILE) and os.path.exists(METADATA_FILE):
            _model    = joblib.load(MODEL_FILE)
            _metadata = joblib.load(METADATA_FILE)
            print("[INFO] Demand forecasting model assets loaded successfully.")
        else:
            print("[WARNING] Model assets not found. Run with --train first.")
    except Exception as e:
        print(f"[ERROR] Failed to load demand model assets: {e}")


load_demand_assets()


class DemandPredictionRequest(BaseModel):
    store_id:         str   = Field("S001", description="Store ID (e.g. S001)")
    product_id:       str   = Field("P0001", description="Product ID (e.g. P0001)")
    category:         str   = Field("Electronics", description="Product category")
    region:           str   = Field("North", description="Geographic region")
    inventory_level:  float = Field(150.0, description="Current inventory level (units)")
    price:            float = Field(50.0, description="Selling price (USD)", ge=0)
    discount:         float = Field(10.0, description="Discount percentage", ge=0, le=100)
    weather_condition: str  = Field("Sunny", description="Weather condition")
    promotion:        int   = Field(0, description="Promotion active (0=No, 1=Yes)", ge=0, le=1)
    competitor_pricing: float = Field(55.0, description="Competitor price (USD)", ge=0)
    seasonality:      str   = Field("Summer", description="Season (Winter/Spring/Summer/Fall)")
    epidemic:         int   = Field(0, description="Epidemic indicator (0=No, 1=Yes)", ge=0, le=1)


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serve the main demand forecasting frontend UI."""
    frontend_path = "demand_forecasting_frontend.html"
    if os.path.exists(frontend_path):
        with open(frontend_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h3>Frontend file not found.</h3>", status_code=404)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok" if _model is not None else "no_model_loaded",
        "model": "RandomForestRegressor",
        "assets_exist": {
            "model":    os.path.exists(MODEL_FILE),
            "metadata": os.path.exists(METADATA_FILE),
        },
    }


@app.post("/predict")
def predict_demand(req: DemandPredictionRequest):
    """Predict demand for the given product / store context."""
    if _model is None or _metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run with --train first.")

    try:
        feature_names  = _metadata["feature_names"]
        scaler         = _metadata["scaler"]
        le_map         = _metadata["label_encoders"]
        cat_cols       = _metadata["cat_cols"]

        # Build a raw dict matching training columns (excluding dropped cols and target)
        raw = {
            "Store ID":           req.store_id,
            "Product ID":         req.product_id,
            "Category":           req.category,
            "Region":             req.region,
            "Inventory Level":    req.inventory_level,
            "Price":              req.price,
            "Discount":           req.discount,
            "Weather Condition":  req.weather_condition,
            "Promotion":          req.promotion,
            "Competitor Pricing": req.competitor_pricing,
            "Seasonality":        req.seasonality,
            "Epidemic":           req.epidemic,
        }

        # Encode categoricals with saved LabelEncoders
        for col in cat_cols:
            if col in raw:
                le = le_map[col]
                val = str(raw[col])
                if val in le.classes_:
                    raw[col] = int(le.transform([val])[0])
                else:
                    # Unseen label – use median encoded class
                    raw[col] = int(len(le.classes_) // 2)

        # Build feature row in the exact order used during training
        row = [raw.get(f, 0) for f in feature_names]
        X_input = np.array(row, dtype=float).reshape(1, -1)
        X_scaled = scaler.transform(X_input)

        predicted_demand = float(_model.predict(X_scaled)[0])
        predicted_demand = max(0.0, round(predicted_demand, 2))

        # Confidence interval (±1 std of tree predictions)
        tree_preds = np.array([t.predict(X_scaled)[0] for t in _model.estimators_])
        lower = max(0.0, round(float(tree_preds.mean() - tree_preds.std()), 2))
        upper = round(float(tree_preds.mean() + tree_preds.std()), 2)

        return {
            "predicted_demand":   predicted_demand,
            "demand_lower_bound": lower,
            "demand_upper_bound": upper,
            "confidence_range":   round(upper - lower, 2),
            "model_r2":           round(_metadata["r2"], 4),
            "model_rmse":         round(_metadata["rmse"], 4),
            "category":           req.category,
            "region":             req.region,
            "store_id":           req.store_id,
            "product_id":         req.product_id,
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
            "demand_trend":        "/results/image/demand_trend.png",
            "sales_distribution":  "/results/image/sales_distribution.png",
            "correlation_heatmap": "/results/image/correlation_heatmap.png",
            "feature_importance":  "/results/image/feature_importance.png",
            "actual_vs_predicted": "/results/image/actual_vs_predicted.png",
            "residuals_distribution": "/results/image/residuals_distribution.png",
            "model_metrics":       "/results/image/model_metrics.png",
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
        train_demand_model()
        load_demand_assets()
        print("[RETRAIN] Model retrained successfully.")
    except Exception as e:
        print(f"[RETRAIN] Error: {e}")


@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    """Retrain the demand forecasting model in the background."""
    background_tasks.add_task(_background_retrain)
    return {"status": "accepted", "message": "Model retraining initiated in the background."}


# ──────────────────────────────────────────────────────────────────────
# 3. Main Execution Block
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demand Forecasting unified ML pipeline and backend")
    parser.add_argument("--train", action="store_true", help="Force train the model and regenerate results")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI port (default: 8000)")
    args = parser.parse_args()

    if args.train or not os.path.exists(MODEL_FILE) or not os.path.exists(METADATA_FILE):
        try:
            train_demand_model()
            load_demand_assets()
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            sys.exit(1)

    if "--train" not in sys.argv:
        import uvicorn
        print(f"[INFO] Starting Demand Forecasting API server on port {args.port}...")
        uvicorn.run("demand_forecasting:app", host="127.0.0.1", port=args.port, reload=False)
