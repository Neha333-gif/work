# -*- coding: utf-8 -*-
"""
Retail Sales Forecasting - Unified ML Pipeline and FastAPI Backend
Trains lightweight regression models on retail sales data,
generates visualization artifacts, saves model assets, and
serves real-time retail sales forecasts via FastAPI.
"""

import os
import sys
import argparse
import warnings

import pandas as pd
import numpy as np
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

DATA_PATH = r"C:\Users\peepl\Downloads\cust_segmentation_zip\repo_for_demand_forecasting\repo_for_demand_forecasting\retail_sales_dataset.csv"
RESULTS_DIR = "results"
MODEL_FILE = "retail_sales_forecasting_model.joblib"
METADATA_FILE = "retail_sales_forecasting_metadata.joblib"
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, "retail_sales_forecasts.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)


def _clear_previous_outputs():
    for name in os.listdir(RESULTS_DIR):
        if name.endswith((".png", ".txt", ".csv", ".json")):
            try:
                os.remove(os.path.join(RESULTS_DIR, name))
            except OSError:
                pass


def _encode_with_fallback(encoder: LabelEncoder, value: str):
    value = str(value).strip().lower()
    classes = [str(c).strip().lower() for c in encoder.classes_]
    if value in classes:
        return float(classes.index(value))
    return float(0)


def _prepare_dataset(raw_df: pd.DataFrame):
    df = raw_df.copy()
    df.columns = [c.strip() for c in df.columns]

    expected_cols = [
        "Date",
        "Customer ID",
        "Gender",
        "Age",
        "Product Category",
        "Quantity",
        "Price per Unit",
        "Total Amount",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")

    if "Transaction ID" in df.columns:
        df.drop(columns=["Transaction ID"], inplace=True)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Month"] = df["Date"].dt.month.fillna(1).astype(int)

    regions = ["North", "South", "East", "West"]
    df["Store_Location"] = df["Customer ID"].astype(str).apply(
        lambda x: regions[abs(hash(x)) % len(regions)]
    )

    promo_map = {
        "Beauty": "Seasonal",
        "Clothing": "Clearance",
        "Electronics": "Flash Sale",
    }
    df["Promotion_Type"] = df["Product Category"].map(promo_map).fillna("Standard")

    for col in ["Quantity", "Price per Unit", "Total Amount", "Age"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    df["Marketing_Spend"] = df["Quantity"] * df["Price per Unit"] * 0.08
    df["Seasonal_Demand_Index"] = df["Month"] / 12.0
    df["Number_of_Customers"] = 1
    df["Discount_Percentage"] = np.clip((500 - df["Price per Unit"]) / 500 * 100, 0, 80)
    df["Inventory_Level"] = df["Quantity"] * 15

    monthly_avg = df.groupby(["Product Category", "Month"])["Total Amount"].transform("mean")
    df["Previous_Month_Sales"] = monthly_avg * 0.85

    encoders = {}
    for col in ["Gender", "Product Category", "Store_Location", "Promotion_Type"]:
        enc = LabelEncoder()
        df[col] = enc.fit_transform(df[col].astype(str).str.strip())
        encoders[col] = enc

    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    feature_cols = [
        "Product Category",
        "Store_Location",
        "Previous_Month_Sales",
        "Marketing_Spend",
        "Seasonal_Demand_Index",
        "Number_of_Customers",
        "Discount_Percentage",
        "Inventory_Level",
        "Promotion_Type",
        "Quantity",
        "Price per Unit",
        "Age",
        "Gender",
    ]
    X = df[feature_cols].copy()
    y = df["Total Amount"].copy()

    return df, X, y, encoders, feature_cols


def _build_retail_row(req, encoders):
    qty = max(1.0, float(req.inventory_level) / 15.0)
    price = max(1.0, 500.0 * (1.0 - float(req.discount_percentage) / 100.0))
    return {
        "Product Category": _encode_with_fallback(encoders["Product Category"], req.product_category),
        "Store_Location": _encode_with_fallback(encoders["Store_Location"], req.store_location),
        "Previous_Month_Sales": float(req.previous_month_sales),
        "Marketing_Spend": float(req.marketing_spend),
        "Seasonal_Demand_Index": float(req.seasonal_demand_index),
        "Number_of_Customers": float(req.number_of_customers),
        "Discount_Percentage": float(req.discount_percentage),
        "Inventory_Level": float(req.inventory_level),
        "Promotion_Type": _encode_with_fallback(encoders["Promotion_Type"], req.promotion_type),
        "Quantity": qty,
        "Price per Unit": price,
        "Age": 35.0,
        "Gender": _encode_with_fallback(encoders["Gender"], "Male"),
    }


def _sales_trend_indicator(forecast: float, previous_sales: float) -> str:
    if forecast > previous_sales * 1.05:
        return "Growing"
    if forecast < previous_sales * 0.95:
        return "Declining"
    return "Stable"


def train_retail_sales_model():
    print("[INFO] Starting Retail Sales Forecasting model training...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    _clear_previous_outputs()

    df = pd.read_csv(DATA_PATH)
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42).reset_index(drop=True)

    df, X, y, encoders, feature_names = _prepare_dataset(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=220, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
    }

    model_metrics = {}
    trained_models = {}

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, pred)
        model_metrics[model_name] = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}
        trained_models[model_name] = model

    best_model_name = min(model_metrics.keys(), key=lambda k: model_metrics[k]["rmse"])
    model = trained_models[best_model_name]

    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    _, X_test_raw, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    prediction_frame = X_test_raw.copy()
    prediction_frame["actual_total_amount"] = y_test.values
    prediction_frame["forecasted_total_amount"] = y_pred
    prediction_frame["forecast_error"] = y_test.values - y_pred
    prediction_frame.to_csv(PREDICTIONS_FILE, index=False)

    metrics_text = f"""Retail Sales Forecasting - Model Evaluation Report
=================================================
Selected Model           : {best_model_name}
MAE                      : {mae:.4f}
MSE                      : {mse:.4f}
RMSE                     : {rmse:.4f}
R2 Score                 : {r2:.4f}
Dataset Path             : {DATA_PATH}
Total Samples            : {len(df)}
Training Samples         : {len(x_train)}
Testing Samples          : {len(x_test)}

Model Comparison
----------------
{os.linesep.join([f"{k}: MAE={v['mae']:.4f}, RMSE={v['rmse']:.4f}, R2={v['r2']:.4f}" for k, v in model_metrics.items()])}
"""

    with open(os.path.join(RESULTS_DIR, "accuracy_results.txt"), "w", encoding="utf-8") as f:
        f.write(metrics_text)
    with open(os.path.join(RESULTS_DIR, "evaluation_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR2: {r2:.4f}\n")

    sns.set_theme(style="darkgrid")

    fig, ax = plt.subplots(figsize=(10, 5))
    monthly_sales = df.groupby("Month")["Total Amount"].sum()
    ax.plot(monthly_sales.index, monthly_sales.values, marker="o", color="#2563eb", linewidth=2)
    ax.set_title("Retail Sales Trend Graph")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sales_trend.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        df[feature_names + ["Total Amount"]].corr(numeric_only=True),
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        ax=ax,
    )
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    elif hasattr(model, "coef_"):
        importances = pd.Series(np.abs(model.coef_), index=feature_names).sort_values()
    else:
        importances = pd.Series(np.zeros(len(feature_names)), index=feature_names).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind="barh", color="#14b8a6", ax=ax)
    ax.set_title("Feature Importance Plot")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.4, color="#8b5cf6")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "--", color="#ef4444")
    ax.set_title("Actual vs Forecasted Retail Sales")
    ax.set_xlabel("Actual Total Amount")
    ax.set_ylabel("Forecasted Total Amount")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "actual_vs_forecasted.png"), dpi=150)
    plt.close()

    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, bins=30, kde=True, color="#f59e0b", ax=ax)
    ax.axvline(0, linestyle="--", color="#ef4444")
    ax.set_title("Residual Error Distribution")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "residual_distribution.png"), dpi=150)
    plt.close()

    cat_labels = encoders["Product Category"].classes_
    cat_map = {i: str(label) for i, label in enumerate(cat_labels)}
    fig, ax = plt.subplots(figsize=(8, 5))
    df.groupby(df["Product Category"].map(cat_map))["Total Amount"].mean().plot(kind="bar", color="#6366f1", ax=ax)
    ax.set_title("Product Category Sales Comparison")
    ax.set_xlabel("Product Category")
    ax.set_ylabel("Average Total Amount")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "product_category_comparison.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    monthly_sales.plot(kind="bar", color="#22c55e", ax=ax)
    ax.set_title("Monthly Sales Analysis")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "monthly_sales_analysis.png"), dpi=150)
    plt.close()

    loc_labels = encoders["Store_Location"].classes_
    loc_map = {i: str(label) for i, label in enumerate(loc_labels)}
    fig, ax = plt.subplots(figsize=(8, 5))
    df.groupby(df["Store_Location"].map(loc_map))["Total Amount"].sum().plot(kind="bar", color="#f59e0b", ax=ax)
    ax.set_title("Regional Sales Performance Analysis")
    ax.set_xlabel("Store Location")
    ax.set_ylabel("Total Sales")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "regional_sales_performance.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    metric_names = ["MAE", "RMSE", "R2"]
    metric_values = [mae, rmse, r2]
    bars = ax.bar(metric_names, metric_values, color=["#2563eb", "#f59e0b", "#22c55e"])
    for b, v in zip(bars, metric_values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Model Performance Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_performance_metrics.png"), dpi=150)
    plt.close()

    metadata = {
        "feature_names": feature_names,
        "scaler": scaler,
        "encoders": encoders,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "best_model_name": best_model_name,
    }
    joblib.dump(model, MODEL_FILE)
    joblib.dump(metadata, METADATA_FILE)
    print("[DONE] Retail Sales Forecasting training complete.")


app = FastAPI(title="Retail Sales Forecasting API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_metadata = None


def load_retail_sales_assets():
    global _model, _metadata
    if os.path.exists(MODEL_FILE) and os.path.exists(METADATA_FILE):
        _model = joblib.load(MODEL_FILE)
        _metadata = joblib.load(METADATA_FILE)


load_retail_sales_assets()


class RetailSalesForecastingRequest(BaseModel):
    product_category: str = Field(..., min_length=2)
    store_location: str = Field(..., min_length=2)
    previous_month_sales: float = Field(..., ge=0)
    marketing_spend: float = Field(..., ge=0)
    seasonal_demand_index: float = Field(..., ge=0, le=1)
    number_of_customers: int = Field(..., ge=1)
    discount_percentage: float = Field(..., ge=0, le=100)
    inventory_level: float = Field(..., ge=0)
    promotion_type: str = Field(..., min_length=2)


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    path = "retail_sales_forecasting_frontend.html"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h3>Frontend file not found.</h3>", status_code=404)


@app.get("/health")
def health():
    return {
        "status": "ok" if _model is not None else "no_model_loaded",
        "model": _metadata["best_model_name"] if _metadata else "not_loaded",
        "assets_exist": {"model": os.path.exists(MODEL_FILE), "metadata": os.path.exists(METADATA_FILE)},
    }


@app.post("/predict")
def predict_retail_sales(req: RetailSalesForecastingRequest):
    if _model is None or _metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run with --train first.")

    row = _build_retail_row(req, _metadata["encoders"])
    feature_names = _metadata["feature_names"]
    X_input = np.array([[row.get(f, 0.0) for f in feature_names]], dtype=float)
    X_scaled = _metadata["scaler"].transform(X_input)

    forecast = float(_model.predict(X_scaled)[0])
    forecast = max(0.0, round(forecast, 2))

    spread = max(50.0, float(_metadata["rmse"]) * 0.75)
    lower = round(max(0.0, forecast - spread), 2)
    upper = round(forecast + spread, 2)
    trend = _sales_trend_indicator(forecast, float(req.previous_month_sales))

    return {
        "forecasted_retail_sales": forecast,
        "estimated_revenue_range": {"lower": lower, "upper": upper},
        "sales_trend_indicator": trend,
        "selected_model": _metadata["best_model_name"],
        "model_rmse": round(float(_metadata["rmse"]), 2),
        "model_r2": round(float(_metadata["r2"]), 4),
    }


@app.get("/results/metrics")
def get_metrics():
    metrics_file = os.path.join(RESULTS_DIR, "accuracy_results.txt")
    if not os.path.exists(metrics_file):
        raise HTTPException(status_code=404, detail="Metrics not found. Train the model first.")
    with open(metrics_file, "r", encoding="utf-8") as f:
        content = f.read()
    return {
        "text_report": content,
        "plots": {
            "sales_trend": "/results/image/sales_trend.png",
            "correlation_heatmap": "/results/image/correlation_heatmap.png",
            "feature_importance": "/results/image/feature_importance.png",
            "actual_vs_forecasted": "/results/image/actual_vs_forecasted.png",
            "residual_distribution": "/results/image/residual_distribution.png",
            "monthly_sales_analysis": "/results/image/monthly_sales_analysis.png",
            "product_category_comparison": "/results/image/product_category_comparison.png",
            "regional_sales_performance": "/results/image/regional_sales_performance.png",
            "model_performance_metrics": "/results/image/model_performance_metrics.png",
        },
    }


@app.get("/results/image/{image_name}")
def serve_plot(image_name: str):
    path = os.path.join(RESULTS_DIR, image_name)
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Image not found.")


def _background_retrain():
    train_retail_sales_model()
    load_retail_sales_assets()


@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(_background_retrain)
    return {"status": "accepted", "message": "Retail sales forecasting model retraining started."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retail Sales Forecasting unified ML pipeline and backend")
    parser.add_argument("--train", action="store_true", help="Force train the model and regenerate results")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI port (default: 8000)")
    args = parser.parse_args()

    if args.train or not os.path.exists(MODEL_FILE) or not os.path.exists(METADATA_FILE):
        try:
            train_retail_sales_model()
            load_retail_sales_assets()
        except Exception as exc:
            print(f"[ERROR] Training failed: {exc}")
            sys.exit(1)

    if "--train" not in sys.argv:
        import uvicorn
        print(f"[INFO] Starting Retail Sales Forecasting API server on port {args.port}...")
        uvicorn.run("retail_sales_forecasting:app", host="127.0.0.1", port=args.port, reload=False)
