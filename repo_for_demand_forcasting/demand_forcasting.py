"""Demand Forecasting System - Unified ML pipeline and FastAPI backend."""

import argparse
import os
import sys
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

DATA_PATH = "/content/demand_forecasting.csv"
LOCAL_DATA_FALLBACK = os.path.join("..", "demand forcasting system", "train_0irEZ2H.csv")
RESULTS_DIR = "results"
MODEL_FILE = "demand_forcasting_model.joblib"
METADATA_FILE = "demand_forcasting_metadata.joblib"
OUTPUTS_FILE = os.path.join(RESULTS_DIR, "demand_forecasting_outputs.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)


def _resolve_data_path() -> str:
    if os.path.exists(DATA_PATH):
        return DATA_PATH
    if os.path.exists(LOCAL_DATA_FALLBACK):
        return LOCAL_DATA_FALLBACK
    raise FileNotFoundError(f"Dataset not found. Tried: {DATA_PATH} and {LOCAL_DATA_FALLBACK}")


def _clear_previous_outputs():
    for name in os.listdir(RESULTS_DIR):
        if name.endswith((".png", ".txt", ".csv", ".json")):
            try:
                os.remove(os.path.join(RESULTS_DIR, name))
            except OSError:
                pass


def _prepare_dataset(df: pd.DataFrame):
    needed = [
        "week",
        "store_id",
        "sku_id",
        "total_price",
        "base_price",
        "is_featured_sku",
        "is_display_sku",
        "units_sold",
    ]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    clean = df.copy()
    clean["week"] = pd.to_datetime(clean["week"], dayfirst=True, errors="coerce")
    clean.dropna(subset=["week", "units_sold"], inplace=True)
    clean["units_sold"] = pd.to_numeric(clean["units_sold"], errors="coerce")
    clean.dropna(subset=["units_sold"], inplace=True)

    clean["month"] = clean["week"].dt.month
    clean["year"] = clean["week"].dt.year
    clean["discount_amount"] = (clean["base_price"] - clean["total_price"]).clip(lower=0)
    clean["discount_ratio"] = np.where(clean["base_price"] > 0, clean["discount_amount"] / clean["base_price"], 0)
    clean["seasonal_index"] = np.sin((2 * np.pi * clean["month"]) / 12.0)
    clean["inventory_level"] = clean["base_price"]
    clean["historical_sales"] = clean.groupby(["store_id", "sku_id"])["units_sold"].transform("mean")
    clean["marketing_spend"] = clean["is_featured_sku"] * 100 + clean["is_display_sku"] * 50
    clean["region_code"] = clean["store_id"] % 10
    clean["promotional_offers"] = clean["is_featured_sku"]
    clean["customer_demand_trends"] = clean.groupby("sku_id")["units_sold"].transform("mean")
    clean["product_category"] = pd.qcut(
        clean["sku_id"].rank(method="first"),
        q=5,
        labels=[0, 1, 2, 3, 4],
    ).astype(int)

    feature_cols = [
        "store_id",
        "sku_id",
        "total_price",
        "base_price",
        "is_featured_sku",
        "is_display_sku",
        "month",
        "year",
        "discount_amount",
        "discount_ratio",
        "seasonal_index",
        "inventory_level",
        "historical_sales",
        "marketing_spend",
        "region_code",
        "promotional_offers",
        "customer_demand_trends",
        "product_category",
    ]
    x = clean[feature_cols].copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True))
    y = clean["units_sold"].copy()
    return clean, x, y, feature_cols


def train_demand_forcasting_model():
    print("[INFO] Starting Demand Forecasting System training...")
    data_path = _resolve_data_path()
    _clear_previous_outputs()

    df = pd.read_csv(data_path)
    clean, x, y, feature_names = _prepare_dataset(df)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=12, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=220, max_depth=14, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
    }

    model_metrics = {}
    trained_models = {}
    preds = {}
    for name, model in models.items():
        if name == "Linear Regression":
            model.fit(x_train_scaled, y_train)
            pred = model.predict(x_test_scaled)
        else:
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        rmse = float(np.sqrt(mse))
        model_metrics[name] = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2_score(y_test, pred)}
        trained_models[name] = model
        preds[name] = pred

    best_model_name = min(model_metrics.keys(), key=lambda n: model_metrics[n]["rmse"])
    model = trained_models[best_model_name]
    y_pred = preds[best_model_name]

    forecast_df = x_test.copy()
    forecast_df["actual_units_sold"] = y_test.values
    forecast_df["forecasted_units_sold"] = y_pred
    forecast_df["residual_error"] = forecast_df["actual_units_sold"] - forecast_df["forecasted_units_sold"]
    forecast_df.to_csv(OUTPUTS_FILE, index=False)

    best = model_metrics[best_model_name]
    report_text = f"""Demand Forecasting System - Model Evaluation Report
=================================================
Selected Model           : {best_model_name}
MAE                      : {best['mae']:.4f}
MSE                      : {best['mse']:.4f}
RMSE                     : {best['rmse']:.4f}
R2 Score                 : {best['r2']:.4f}
Dataset Path             : {data_path}
Total Samples            : {len(clean)}
Training Samples         : {len(x_train)}
Testing Samples          : {len(x_test)}

Model Comparison
----------------
{os.linesep.join([f"{k}: mae={v['mae']:.4f}, mse={v['mse']:.4f}, rmse={v['rmse']:.4f}, r2={v['r2']:.4f}" for k, v in model_metrics.items()])}
"""
    with open(os.path.join(RESULTS_DIR, "accuracy_results.txt"), "w", encoding="utf-8") as fp:
        fp.write(report_text)
    with open(os.path.join(RESULTS_DIR, "evaluation_report.txt"), "w", encoding="utf-8") as fp:
        fp.write(
            f"MAE: {best['mae']:.4f}\nMSE: {best['mse']:.4f}\nRMSE: {best['rmse']:.4f}\nR2: {best['r2']:.4f}\n"
        )

    sns.set_theme(style="darkgrid")

    fig, ax = plt.subplots(figsize=(10, 5))
    trend = clean.groupby("week")["units_sold"].sum().reset_index().sort_values("week")
    ax.plot(trend["week"].tail(100), trend["units_sold"].tail(100), color="#2563eb")
    ax.set_title("Demand Trend Graph")
    ax.set_xlabel("Week")
    ax.set_ylabel("Units Sold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "demand_trend.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 6))
    corr_cols = feature_names + ["units_sold"]
    sns.heatmap(clean[corr_cols].corr(numeric_only=True), cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    else:
        importances = pd.Series(np.abs(np.ravel(getattr(model, "coef_", np.ones(len(feature_names))))), index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind="barh", color="#14b8a6", ax=ax)
    ax.set_title("Feature Importance Plot")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.4, color="#f59e0b")
    lo = min(float(y_test.min()), float(np.min(y_pred)))
    hi = max(float(y_test.max()), float(np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "--", color="red")
    ax.set_title("Actual vs Forecasted Demand Graph")
    ax.set_xlabel("Actual Units Sold")
    ax.set_ylabel("Forecasted Units Sold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "actual_vs_forecasted_demand.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(forecast_df["residual_error"], bins=30, kde=True, color="#22c55e", ax=ax)
    ax.set_title("Residual/Error Distribution Plot")
    ax.set_xlabel("Residual Error")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "residual_error_distribution.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    seasonal = clean.groupby("month")["units_sold"].mean().reindex(range(1, 13), fill_value=0)
    ax.plot(seasonal.index, seasonal.values, marker="o", color="#ef4444")
    ax.set_title("Seasonal Demand Analysis")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Units Sold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "seasonal_demand_analysis.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    cat = clean.copy()
    cat["product_category_label"] = cat["product_category"].astype(str)
    cat.groupby("product_category_label")["units_sold"].mean().plot(kind="bar", color="#8b5cf6", ax=ax)
    ax.set_title("Product Category Demand Comparison Chart")
    ax.set_xlabel("Product Category")
    ax.set_ylabel("Average Units Sold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "product_category_demand_comparison.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(clean["inventory_level"], clean["units_sold"], alpha=0.3, color="#6366f1")
    ax.set_title("Inventory vs Demand Analysis")
    ax.set_xlabel("Inventory Level")
    ax.set_ylabel("Units Sold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "inventory_vs_demand_analysis.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(model_metrics.keys())
    rmse_vals = [model_metrics[n]["rmse"] for n in names]
    mae_vals = [model_metrics[n]["mae"] for n in names]
    bars1 = ax.bar(np.arange(len(names)) - 0.2, rmse_vals, width=0.4, label="RMSE", color="#2563eb")
    bars2 = ax.bar(np.arange(len(names)) + 0.2, mae_vals, width=0.4, label="MAE", color="#22c55e")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=10, ha="right")
    ax.set_title("Model Performance Metrics Visualization")
    ax.legend()
    for b in list(bars1) + list(bars2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_performance_metrics_visualization.png"), dpi=150)
    plt.close()

    summary = pd.DataFrame(
        [{"model": k, "mae": v["mae"], "mse": v["mse"], "rmse": v["rmse"], "r2": v["r2"]} for k, v in model_metrics.items()]
    )
    summary.to_csv(os.path.join(RESULTS_DIR, "model_metrics_summary.csv"), index=False)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(
        {
            "feature_names": feature_names,
            "scaler": scaler,
            "best_model_name": best_model_name,
            "metrics": best,
        },
        METADATA_FILE,
    )
    print("[DONE] Demand Forecasting System training complete.")


app = FastAPI(title="Demand Forecasting System API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_metadata = None


def load_demand_forcasting_assets():
    global _model, _metadata
    if os.path.exists(MODEL_FILE) and os.path.exists(METADATA_FILE):
        _model = joblib.load(MODEL_FILE)
        _metadata = joblib.load(METADATA_FILE)


load_demand_forcasting_assets()


class DemandForecastRequest(BaseModel):
    product_category: str = Field(..., min_length=1)
    historical_sales: float = Field(..., ge=0)
    inventory_level: float = Field(..., ge=0)
    seasonal_index: float = Field(..., ge=0)
    marketing_spend: float = Field(..., ge=0)
    region: str = Field(..., min_length=1)
    month: int = Field(..., ge=1, le=12)
    promotional_offers: int = Field(..., ge=0, le=1)
    customer_demand_trends: float = Field(..., ge=0)


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    ui_path = "demand_forcasting_frontend.html"
    if os.path.exists(ui_path):
        with open(ui_path, "r", encoding="utf-8") as fp:
            return HTMLResponse(content=fp.read())
    return HTMLResponse(content="<h3>Frontend file not found.</h3>", status_code=404)


@app.get("/health")
def health():
    return {
        "status": "ok" if _model is not None else "no_model_loaded",
        "model": _metadata["best_model_name"] if _metadata else "not_loaded",
    }


@app.post("/forecast")
def forecast_demand(req: DemandForecastRequest):
    if _model is None or _metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run with --train first.")

    region_code = float(sum(ord(ch) for ch in req.region) % 10)
    category_code = float(sum(ord(ch) for ch in req.product_category) % 5)
    base_price = max(1.0, req.inventory_level)
    total_price = max(1.0, base_price - req.marketing_spend * 0.01)
    discount_amount = max(0.0, base_price - total_price)
    discount_ratio = discount_amount / base_price

    row = {
        "store_id": region_code * 1000 + 8000,
        "sku_id": category_code * 10000 + 200000,
        "total_price": total_price,
        "base_price": base_price,
        "is_featured_sku": float(req.promotional_offers),
        "is_display_sku": float(1 if req.customer_demand_trends >= 0.5 else 0),
        "month": float(req.month),
        "year": 2013.0,
        "discount_amount": discount_amount,
        "discount_ratio": discount_ratio,
        "seasonal_index": float(req.seasonal_index),
        "inventory_level": float(req.inventory_level),
        "historical_sales": float(req.historical_sales),
        "marketing_spend": float(req.marketing_spend),
        "region_code": region_code,
        "promotional_offers": float(req.promotional_offers),
        "customer_demand_trends": float(req.customer_demand_trends),
        "product_category": category_code,
    }

    x_input = np.array([[row.get(col, 0.0) for col in _metadata["feature_names"]]], dtype=float)
    if _metadata["best_model_name"] == "Linear Regression":
        x_input = _metadata["scaler"].transform(x_input)
    forecast = float(_model.predict(x_input)[0])
    forecast = max(0.0, forecast)
    low = max(0.0, forecast * 0.9)
    high = forecast * 1.1
    trend = "Rising" if forecast >= req.historical_sales else "Stable/Declining"

    return {
        "forecasted_product_demand": round(forecast, 2),
        "estimated_sales_range": f"{low:.2f} - {high:.2f}",
        "demand_trend_indicator": trend,
        "selected_model": _metadata["best_model_name"],
        "mae": round(float(_metadata["metrics"]["mae"]), 4),
        "mse": round(float(_metadata["metrics"]["mse"]), 4),
        "rmse": round(float(_metadata["metrics"]["rmse"]), 4),
    }


@app.get("/results/metrics")
def get_metrics():
    path = os.path.join(RESULTS_DIR, "accuracy_results.txt")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Metrics not found. Train model first.")
    with open(path, "r", encoding="utf-8") as fp:
        return {"text_report": fp.read()}


@app.get("/results/image/{image_name}")
def serve_image(image_name: str):
    path = os.path.join(RESULTS_DIR, image_name)
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Image not found.")


def _background_retrain():
    train_demand_forcasting_model()
    load_demand_forcasting_assets()


@app.post("/retrain")
def retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(_background_retrain)
    return {"status": "accepted", "message": "Demand forecasting retraining started."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demand Forecasting System unified ML pipeline and backend")
    parser.add_argument("--train", action="store_true", help="Train model and regenerate results")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI port")
    args = parser.parse_args()

    if args.train or not os.path.exists(MODEL_FILE) or not os.path.exists(METADATA_FILE):
        try:
            train_demand_forcasting_model()
            load_demand_forcasting_assets()
        except Exception as exc:
            print(f"[ERROR] Training failed: {exc}")
            sys.exit(1)

    if "--train" not in sys.argv:
        import uvicorn

        print(f"[INFO] Starting Demand Forecasting System API on port {args.port}...")
        uvicorn.run("demand_forcasting:app", host="127.0.0.1", port=args.port, reload=False)
