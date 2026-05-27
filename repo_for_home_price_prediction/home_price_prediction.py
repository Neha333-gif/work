# -*- coding: utf-8 -*-
"""
Home Price Prediction - Unified ML Pipeline and FastAPI Backend
Trains lightweight regression models on real estate property data,
generates visualization artifacts, saves model assets, and
serves real-time property price predictions via FastAPI.
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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

DATA_PATH = r"C:\Users\peepl\Downloads\cust_segmentation_zip\repo_for_demand_forecasting\repo_for_demand_forecasting\House Price India.csv"
RESULTS_DIR = "results"
MODEL_FILE = "home_price_prediction_model.joblib"
METADATA_FILE = "home_price_prediction_metadata.joblib"
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, "home_price_predictions.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)


def _clean_results_directory():
    for file_name in os.listdir(RESULTS_DIR):
        if file_name.endswith((".png", ".txt", ".csv")):
            try:
                os.remove(os.path.join(RESULTS_DIR, file_name))
            except OSError:
                pass


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "number of bedrooms": "bedrooms",
        "number of bathrooms": "bathrooms",
        "living area": "living_area",
        "lot area": "lot_area",
        "number of floors": "floors",
        "waterfront present": "garage_available",
        "number of views": "views",
        "condition of the house": "condition_score",
        "grade of the house": "grade_score",
        "Area of the house(excluding basement)": "area_without_basement",
        "Area of the basement": "basement_area",
        "Built Year": "year_built",
        "Renovation Year": "renovation_year",
        "Postal Code": "postal_code",
        "Lattitude": "latitude",
        "Longitude": "longitude",
        "living_area_renov": "living_area_renov",
        "lot_area_renov": "lot_area_renov",
        "Number of schools nearby": "schools_nearby",
        "Distance from the airport": "distance_from_airport",
        "Price": "price",
    }
    return df.rename(columns=rename_map)


def _build_property_row(req):
    property_type_map = {
        "Apartment": 6,
        "Condo": 7,
        "Townhouse": 8,
        "Single Family": 9,
        "Luxury Villa": 10,
    }
    grade_score = property_type_map.get(req.property_type, 8)
    condition_score = 4 if req.property_type in {"Single Family", "Luxury Villa"} else 3

    return {
        "bedrooms": float(req.bedrooms),
        "bathrooms": float(req.bathrooms),
        "living_area": float(req.square_footage),
        "lot_area": float(req.lot_size),
        "floors": 2.0 if req.property_type in {"Single Family", "Luxury Villa"} else 1.0,
        "garage_available": float(req.garage_availability),
        "views": 2.0,
        "condition_score": float(condition_score),
        "grade_score": float(grade_score),
        "area_without_basement": float(max(400, req.square_footage - 250)),
        "basement_area": float(max(0, req.square_footage - max(400, req.square_footage - 250))),
        "year_built": float(req.year_built),
        "renovation_year": 0.0,
        "postal_code": float(req.location),
        "latitude": 52.8,
        "longitude": -114.4,
        "living_area_renov": float(max(300, req.square_footage - 120)),
        "lot_area_renov": float(req.lot_size),
        "schools_nearby": 2.0,
        "distance_from_airport": 60.0,
    }


def train_home_price_model():
    print("[INFO] Starting Home Price Prediction model training...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    _clean_results_directory()

    df = pd.read_csv(DATA_PATH)
    df = _normalize_columns(df)
    if len(df) > 25000:
        df = df.sample(n=25000, random_state=42).reset_index(drop=True)

    required_cols = [
        "bedrooms", "bathrooms", "living_area", "lot_area", "floors",
        "garage_available", "views", "condition_score", "grade_score",
        "area_without_basement", "basement_area", "year_built", "renovation_year",
        "postal_code", "latitude", "longitude", "living_area_renov", "lot_area_renov",
        "schools_nearby", "distance_from_airport", "price"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")

    df = df[required_cols].copy()
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    y = df["price"].astype(float)
    X = df.drop(columns=["price"]).astype(float)
    feature_names = list(X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=14, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=180, max_depth=14, min_samples_leaf=2, n_jobs=-1, random_state=42
        ),
    }

    scored_models = {}
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        scored_models[model_name] = {
            "model": model,
            "mae": mean_absolute_error(y_test, pred),
            "mse": mean_squared_error(y_test, pred),
            "rmse": np.sqrt(mean_squared_error(y_test, pred)),
            "r2": r2_score(y_test, pred),
        }

    best_model_name = max(scored_models.keys(), key=lambda k: scored_models[k]["r2"])
    model = scored_models[best_model_name]["model"]
    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    prediction_frame = pd.DataFrame({
        "actual_price": y_test.values,
        "predicted_price": y_pred,
        "residual_error": y_test.values - y_pred,
    })
    prediction_frame.to_csv(PREDICTIONS_FILE, index=False)

    metrics_text = f"""Home Price Prediction - Model Evaluation Report
=================================================
Selected Model           : {best_model_name}
MAE                      : {mae:,.2f}
MSE                      : {mse:,.2f}
RMSE                     : {rmse:,.2f}
R2 Score                 : {r2:.4f}
Dataset Path             : {DATA_PATH}
Total Samples            : {len(df)}
Training Samples         : {len(x_train)}
Testing Samples          : {len(x_test)}

Model Performance Comparison (by R2)
------------------------------------
{os.linesep.join([f"{k}: R2={v['r2']:.4f}, RMSE={v['rmse']:,.2f}" for k, v in scored_models.items()])}
"""
    with open(os.path.join(RESULTS_DIR, "accuracy_results.txt"), "w", encoding="utf-8") as f:
        f.write(metrics_text)

    with open(os.path.join(RESULTS_DIR, "evaluation_report.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"MAE: {mae:,.2f}\nMSE: {mse:,.2f}\nRMSE: {rmse:,.2f}\nR2: {r2:.4f}\nModel: {best_model_name}\n"
        )

    sns.set_theme(style="darkgrid")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["price"], bins=35, kde=True, color="#4f46e5", ax=ax)
    ax.set_title("House Price Distribution")
    ax.set_xlabel("Property Price")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "price_distribution.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 7))
    corr = df[["price", "bedrooms", "bathrooms", "living_area", "lot_area", "grade_score", "year_built", "schools_nearby"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    else:
        importances = pd.Series(np.abs(model.coef_), index=feature_names).sort_values()

    fig, ax = plt.subplots(figsize=(9, 6))
    importances.tail(12).plot(kind="barh", color="#14b8a6", ax=ax)
    ax.set_title("Feature Importance Plot")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.35, color="#2563eb")
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#ef4444")
    ax.set_title("Actual vs Predicted Home Price")
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "actual_vs_predicted.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    residuals = y_test - y_pred
    sns.histplot(residuals, bins=35, kde=True, color="#f59e0b", ax=ax)
    ax.axvline(0, linestyle="--", color="#ef4444")
    ax.set_title("Residual Error Distribution")
    ax.set_xlabel("Actual - Predicted")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "residual_distribution.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    location_prices = df.groupby("postal_code")["price"].mean().sort_values(ascending=False).head(10)
    location_prices.plot(kind="bar", color="#8b5cf6", ax=ax)
    ax.set_title("Location-wise Price Analysis")
    ax.set_xlabel("Postal Code")
    ax.set_ylabel("Average Price")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "location_wise_price_analysis.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    comparison = df[["bedrooms", "bathrooms", "living_area", "lot_area"]].mean()
    comparison.plot(kind="bar", color=["#2563eb", "#10b981", "#f59e0b", "#a855f7"], ax=ax)
    ax.set_title("Property Feature Comparison")
    ax.set_xlabel("Property Feature")
    ax.set_ylabel("Average Value")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "property_feature_comparison.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    metric_labels = ["R2", "MAE", "RMSE"]
    metric_values = [r2, mae, rmse]
    ax.bar(metric_labels, metric_values, color=["#22c55e", "#2563eb", "#f97316"])
    ax.set_title("Model Performance Metrics")
    ax.set_ylabel("Metric Value")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_performance_metrics.png"), dpi=150)
    plt.close()

    metadata = {
        "feature_names": feature_names,
        "scaler": scaler,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "best_model_name": best_model_name,
    }
    joblib.dump(model, MODEL_FILE)
    joblib.dump(metadata, METADATA_FILE)
    print("[DONE] Home Price Prediction training complete.")


app = FastAPI(title="Home Price Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_metadata = None


def load_home_price_assets():
    global _model, _metadata
    if os.path.exists(MODEL_FILE) and os.path.exists(METADATA_FILE):
        _model = joblib.load(MODEL_FILE)
        _metadata = joblib.load(METADATA_FILE)


load_home_price_assets()


class HomePricePredictionRequest(BaseModel):
    bedrooms: int = Field(..., ge=1, le=15)
    bathrooms: float = Field(..., ge=1, le=10)
    square_footage: int = Field(..., ge=300, le=20000)
    location: int = Field(..., description="Postal code")
    year_built: int = Field(..., ge=1800, le=2030)
    garage_availability: int = Field(..., ge=0, le=1)
    lot_size: int = Field(..., ge=500, le=200000)
    property_type: str = Field(..., min_length=3)


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    path = "home_price_prediction_frontend.html"
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
def predict_home_price(req: HomePricePredictionRequest):
    if _model is None or _metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run with --train first.")

    row = _build_property_row(req)
    features = _metadata["feature_names"]
    X_input = np.array([[row.get(f, 0.0) for f in features]], dtype=float)
    X_scaled = _metadata["scaler"].transform(X_input)
    predicted_price = float(_model.predict(X_scaled)[0])
    predicted_price = max(0.0, round(predicted_price, 2))

    spread = max(5000.0, _metadata["rmse"] * 0.8)
    lower = round(max(0.0, predicted_price - spread), 2)
    upper = round(predicted_price + spread, 2)

    return {
        "predicted_home_price": predicted_price,
        "estimated_value_range": {"lower": lower, "upper": upper},
        "selected_model": _metadata["best_model_name"],
        "model_r2": round(float(_metadata["r2"]), 4),
        "model_rmse": round(float(_metadata["rmse"]), 2),
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
            "price_distribution": "/results/image/price_distribution.png",
            "correlation_heatmap": "/results/image/correlation_heatmap.png",
            "feature_importance": "/results/image/feature_importance.png",
            "actual_vs_predicted": "/results/image/actual_vs_predicted.png",
            "residual_distribution": "/results/image/residual_distribution.png",
            "location_wise_price_analysis": "/results/image/location_wise_price_analysis.png",
            "property_feature_comparison": "/results/image/property_feature_comparison.png",
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
    train_home_price_model()
    load_home_price_assets()


@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(_background_retrain)
    return {"status": "accepted", "message": "Home price model retraining started."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Home Price Prediction unified ML pipeline and backend")
    parser.add_argument("--train", action="store_true", help="Force train the model and regenerate results")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI port (default: 8000)")
    args = parser.parse_args()

    if args.train or not os.path.exists(MODEL_FILE) or not os.path.exists(METADATA_FILE):
        try:
            train_home_price_model()
            load_home_price_assets()
        except Exception as exc:
            print(f"[ERROR] Training failed: {exc}")
            sys.exit(1)

    if "--train" not in sys.argv:
        import uvicorn
        print(f"[INFO] Starting Home Price Prediction API server on port {args.port}...")
        uvicorn.run("home_price_prediction:app", host="127.0.0.1", port=args.port, reload=False)
