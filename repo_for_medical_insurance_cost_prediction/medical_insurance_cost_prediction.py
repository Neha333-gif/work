# -*- coding: utf-8 -*-
"""
Medical Insurance Cost Prediction - Unified ML Pipeline and FastAPI Backend
Trains lightweight regression models on patient insurance data,
generates visualization artifacts, saves model assets, and
serves real-time insurance cost predictions via FastAPI.
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

DATA_PATH = r"C:\Users\peepl\Downloads\cust_segmentation_zip\repo_for_demand_forecasting\repo_for_demand_forecasting\medical insurance\medical_insurance.csv"
RESULTS_DIR = "results"
MODEL_FILE = "medical_insurance_cost_prediction_model.joblib"
METADATA_FILE = "medical_insurance_cost_prediction_metadata.joblib"
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, "medical_insurance_cost_predictions.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)


def _clear_previous_outputs():
    for name in os.listdir(RESULTS_DIR):
        if name.endswith((".png", ".txt", ".csv", ".json")):
            try:
                os.remove(os.path.join(RESULTS_DIR, name))
            except OSError:
                pass


def _prepare_dataset(raw_df: pd.DataFrame):
    df = raw_df.copy()
    df.columns = [c.strip() for c in df.columns]

    expected_cols = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")

    encoders = {}
    for col in ["sex", "smoker", "region"]:
        enc = LabelEncoder()
        df[col] = enc.fit_transform(df[col].astype(str).str.strip())
        encoders[col] = enc

    for col in expected_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    X = df.drop(columns=["charges"]).copy()
    y = df["charges"].copy()

    return df, X, y, encoders, list(X.columns)


def _encode_with_fallback(encoder: LabelEncoder, value: str):
    value = str(value).strip().lower()
    classes = [str(c).strip().lower() for c in encoder.classes_]
    if value in classes:
        idx = classes.index(value)
        return float(idx)
    return float(0)


def _build_patient_row(req, encoders):
    return {
        "age": float(req.age),
        "sex": _encode_with_fallback(encoders["sex"], req.gender),
        "bmi": float(req.bmi),
        "children": float(req.number_of_children),
        "smoker": _encode_with_fallback(encoders["smoker"], req.smoking_status),
        "region": _encode_with_fallback(encoders["region"], req.region),
    }


def train_medical_insurance_cost_model():
    print("[INFO] Starting Medical Insurance Cost Prediction model training...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    _clear_previous_outputs()

    df = pd.read_csv(DATA_PATH)
    if len(df) > 20000:
        df = df.sample(n=20000, random_state=42).reset_index(drop=True)

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

    prediction_frame = pd.DataFrame(X.iloc[x_test.shape[0] * -1 :].copy())
    prediction_frame["actual_charges"] = y_test.values
    prediction_frame["predicted_charges"] = y_pred
    prediction_frame["prediction_error"] = y_test.values - y_pred
    prediction_frame.to_csv(PREDICTIONS_FILE, index=False)

    metrics_text = f"""Medical Insurance Cost Prediction - Model Evaluation Report
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

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["charges"], bins=30, kde=True, color="#2563eb", ax=ax)
    ax.set_title("Insurance Cost Distribution")
    ax.set_xlabel("Insurance Charges")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "insurance_cost_distribution.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[feature_names + ["charges"]].corr(), cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
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
    ax.set_title("Actual vs Predicted Insurance Cost")
    ax.set_xlabel("Actual Charges")
    ax.set_ylabel("Predicted Charges")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "actual_vs_predicted.png"), dpi=150)
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

    fig, ax = plt.subplots(figsize=(8, 5))
    df.groupby(pd.cut(df["age"], bins=[17, 25, 35, 45, 55, 70]))["charges"].mean().plot(kind="bar", color="#6366f1", ax=ax)
    ax.set_title("Age-wise Insurance Cost Analysis")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Average Charges")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "age_wise_insurance_cost_analysis.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=df["bmi"], y=df["charges"], hue=df["smoker"], palette="cool", alpha=0.6, ax=ax)
    ax.set_title("BMI vs Insurance Charges Comparison")
    ax.set_xlabel("BMI")
    ax.set_ylabel("Charges")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "bmi_vs_insurance_charges.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    smoker_map = {0: "No", 1: "Yes"}
    pd.DataFrame({"smoker": df["smoker"].map(smoker_map), "charges": df["charges"]}).groupby("smoker")["charges"].mean().plot(kind="bar", color=["#22c55e", "#ef4444"], ax=ax)
    ax.set_title("Smoking Status Impact Analysis")
    ax.set_xlabel("Smoker")
    ax.set_ylabel("Average Charges")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "smoking_status_impact.png"), dpi=150)
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
    print("[DONE] Medical Insurance Cost Prediction training complete.")


app = FastAPI(title="Medical Insurance Cost Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_metadata = None


def load_medical_insurance_assets():
    global _model, _metadata
    if os.path.exists(MODEL_FILE) and os.path.exists(METADATA_FILE):
        _model = joblib.load(MODEL_FILE)
        _metadata = joblib.load(METADATA_FILE)


load_medical_insurance_assets()


class MedicalInsuranceCostPredictionRequest(BaseModel):
    age: int = Field(..., ge=18, le=100)
    gender: str = Field(..., min_length=1)
    bmi: float = Field(..., ge=10, le=80)
    number_of_children: int = Field(..., ge=0, le=10)
    smoking_status: str = Field(..., min_length=1)
    region: str = Field(..., min_length=2)
    annual_income: float = Field(..., ge=1000)
    existing_medical_conditions: str = Field(..., min_length=2)


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    path = "medical_insurance_cost_prediction_frontend.html"
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
def predict_insurance_cost(req: MedicalInsuranceCostPredictionRequest):
    if _model is None or _metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run with --train first.")

    row = _build_patient_row(req, _metadata["encoders"])
    feature_names = _metadata["feature_names"]
    X_input = np.array([[row.get(f, 0.0) for f in feature_names]], dtype=float)
    X_scaled = _metadata["scaler"].transform(X_input)

    pred_cost = float(_model.predict(X_scaled)[0])
    pred_cost = max(0.0, round(pred_cost, 2))

    spread = max(500.0, float(_metadata["rmse"]) * 0.75)
    lower = round(max(0.0, pred_cost - spread), 2)
    upper = round(pred_cost + spread, 2)

    return {
        "predicted_medical_insurance_cost": pred_cost,
        "estimated_insurance_premium_range": {"lower": lower, "upper": upper},
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
            "insurance_cost_distribution": "/results/image/insurance_cost_distribution.png",
            "correlation_heatmap": "/results/image/correlation_heatmap.png",
            "feature_importance": "/results/image/feature_importance.png",
            "actual_vs_predicted": "/results/image/actual_vs_predicted.png",
            "residual_distribution": "/results/image/residual_distribution.png",
            "age_wise_insurance_cost_analysis": "/results/image/age_wise_insurance_cost_analysis.png",
            "bmi_vs_insurance_charges": "/results/image/bmi_vs_insurance_charges.png",
            "smoking_status_impact": "/results/image/smoking_status_impact.png",
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
    train_medical_insurance_cost_model()
    load_medical_insurance_assets()


@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(_background_retrain)
    return {"status": "accepted", "message": "Medical insurance model retraining started."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Insurance Cost Prediction unified ML pipeline and backend")
    parser.add_argument("--train", action="store_true", help="Force train the model and regenerate results")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI port (default: 8000)")
    args = parser.parse_args()

    if args.train or not os.path.exists(MODEL_FILE) or not os.path.exists(METADATA_FILE):
        try:
            train_medical_insurance_cost_model()
            load_medical_insurance_assets()
        except Exception as exc:
            print(f"[ERROR] Training failed: {exc}")
            sys.exit(1)

    if "--train" not in sys.argv:
        import uvicorn
        print(f"[INFO] Starting Medical Insurance Cost Prediction API server on port {args.port}...")
        uvicorn.run("medical_insurance_cost_prediction:app", host="127.0.0.1", port=args.port, reload=False)
