# -*- coding: utf-8 -*-
"""
Student Performance Prediction - Unified ML Pipeline and FastAPI Backend
Trains lightweight regression models on student academic data,
generates visualization artifacts, saves model assets, and
serves real-time student performance predictions via FastAPI.
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

DATA_PATH = r"C:\Users\peepl\Downloads\cust_segmentation_zip\repo_for_demand_forecasting\repo_for_demand_forecasting\student performance\student_performance_dataset.csv"
RESULTS_DIR = "results"
MODEL_FILE = "student_performance_prediction_model.joblib"
METADATA_FILE = "student_performance_prediction_metadata.joblib"
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, "student_performance_predictions.csv")

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

    expected_cols = [
        "Gender",
        "Study_Hours_per_Week",
        "Attendance_Rate",
        "Past_Exam_Scores",
        "Parental_Education_Level",
        "Internet_Access_at_Home",
        "Extracurricular_Activities",
        "Final_Exam_Score",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")

    if "Student_ID" in df.columns:
        df.drop(columns=["Student_ID"], inplace=True)
    if "Pass_Fail" in df.columns:
        df.drop(columns=["Pass_Fail"], inplace=True)

    encoders = {}
    for col in [
        "Gender",
        "Parental_Education_Level",
        "Internet_Access_at_Home",
        "Extracurricular_Activities",
    ]:
        enc = LabelEncoder()
        df[col] = enc.fit_transform(df[col].astype(str).str.strip())
        encoders[col] = enc

    numeric_cols = [
        "Study_Hours_per_Week",
        "Attendance_Rate",
        "Past_Exam_Scores",
        "Final_Exam_Score",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    X = df.drop(columns=["Final_Exam_Score"]).copy()
    y = df["Final_Exam_Score"].copy()

    return df, X, y, encoders, list(X.columns)


def _encode_with_fallback(encoder: LabelEncoder, value: str):
    value = str(value).strip().lower()
    classes = [str(c).strip().lower() for c in encoder.classes_]
    if value in classes:
        idx = classes.index(value)
        return float(idx)
    return float(0)


def _participation_to_extracurricular(participation_level: str) -> str:
    level = str(participation_level).strip().lower()
    if level in {"high", "active", "yes"}:
        return "Yes"
    if level in {"low", "inactive", "no"}:
        return "No"
    return "Yes"


def _build_student_row(req, encoders):
    adjusted_past_score = (
        float(req.previous_exam_scores) * 0.7
        + float(req.assignment_completion_rate) * 0.3
    )
    return {
        "Gender": _encode_with_fallback(encoders["Gender"], req.gender),
        "Study_Hours_per_Week": float(req.study_hours),
        "Attendance_Rate": float(req.attendance_percentage),
        "Past_Exam_Scores": adjusted_past_score,
        "Parental_Education_Level": _encode_with_fallback(
            encoders["Parental_Education_Level"], req.subject
        ),
        "Internet_Access_at_Home": _encode_with_fallback(
            encoders["Internet_Access_at_Home"], "Yes"
        ),
        "Extracurricular_Activities": _encode_with_fallback(
            encoders["Extracurricular_Activities"],
            _participation_to_extracurricular(req.participation_level),
        ),
    }


def train_student_performance_model():
    print("[INFO] Starting Student Performance Prediction model training...")
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

    _, X_test_raw, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    prediction_frame = X_test_raw.copy()
    prediction_frame["actual_final_exam_score"] = y_test.values
    prediction_frame["predicted_final_exam_score"] = y_pred
    prediction_frame["prediction_error"] = y_test.values - y_pred
    prediction_frame.to_csv(PREDICTIONS_FILE, index=False)

    metrics_text = f"""Student Performance Prediction - Model Evaluation Report
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
    sns.histplot(df["Final_Exam_Score"], bins=30, kde=True, color="#2563eb", ax=ax)
    ax.set_title("Student Score Distribution")
    ax.set_xlabel("Final Exam Score")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "score_distribution.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        df[feature_names + ["Final_Exam_Score"]].corr(),
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
    ax.set_title("Actual vs Predicted Student Performance")
    ax.set_xlabel("Actual Final Exam Score")
    ax.set_ylabel("Predicted Final Exam Score")
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

    edu_labels = encoders["Parental_Education_Level"].classes_
    edu_map = {i: str(label) for i, label in enumerate(edu_labels)}
    fig, ax = plt.subplots(figsize=(8, 5))
    df.groupby(df["Parental_Education_Level"].map(edu_map))["Final_Exam_Score"].mean().plot(
        kind="bar", color="#6366f1", ax=ax
    )
    ax.set_title("Subject-wise Performance Analysis (Parental Education Groups)")
    ax.set_xlabel("Education Level Group")
    ax.set_ylabel("Average Final Exam Score")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "subject_wise_performance.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        x=df["Attendance_Rate"],
        y=df["Final_Exam_Score"],
        hue=df["Gender"].map({0: "Female", 1: "Male"}),
        palette="cool",
        alpha=0.6,
        ax=ax,
    )
    ax.set_title("Attendance vs Score Comparison")
    ax.set_xlabel("Attendance Rate (%)")
    ax.set_ylabel("Final Exam Score")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "attendance_vs_score.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(
        x=df["Study_Hours_per_Week"],
        y=df["Final_Exam_Score"],
        scatter_kws={"alpha": 0.5, "color": "#22c55e"},
        line_kws={"color": "#ef4444"},
        ax=ax,
    )
    ax.set_title("Study Hours Impact Analysis")
    ax.set_xlabel("Study Hours per Week")
    ax.set_ylabel("Average Final Exam Score")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "study_hours_impact.png"), dpi=150)
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
    print("[DONE] Student Performance Prediction training complete.")


app = FastAPI(title="Student Performance Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_metadata = None


def load_student_performance_assets():
    global _model, _metadata
    if os.path.exists(MODEL_FILE) and os.path.exists(METADATA_FILE):
        _model = joblib.load(MODEL_FILE)
        _metadata = joblib.load(METADATA_FILE)


load_student_performance_assets()


class StudentPerformancePredictionRequest(BaseModel):
    student_age: int = Field(..., ge=14, le=30)
    gender: str = Field(..., min_length=1)
    attendance_percentage: float = Field(..., ge=0, le=100)
    study_hours: float = Field(..., ge=0, le=60)
    previous_exam_scores: float = Field(..., ge=0, le=100)
    assignment_completion_rate: float = Field(..., ge=0, le=100)
    participation_level: str = Field(..., min_length=2)
    subject: str = Field(..., min_length=2)
    sleep_hours: float = Field(..., ge=0, le=16)


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    path = "student_performance_prediction_frontend.html"
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
def predict_student_performance(req: StudentPerformancePredictionRequest):
    if _model is None or _metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run with --train first.")

    row = _build_student_row(req, _metadata["encoders"])
    feature_names = _metadata["feature_names"]
    X_input = np.array([[row.get(f, 0.0) for f in feature_names]], dtype=float)
    X_scaled = _metadata["scaler"].transform(X_input)

    pred_score = float(_model.predict(X_scaled)[0])
    pred_score = max(0.0, min(100.0, round(pred_score, 2)))

    spread = max(3.0, float(_metadata["rmse"]) * 0.75)
    lower = round(max(0.0, pred_score - spread), 2)
    upper = round(min(100.0, pred_score + spread), 2)

    if pred_score >= 85:
        grade = "A"
    elif pred_score >= 75:
        grade = "B"
    elif pred_score >= 65:
        grade = "C"
    elif pred_score >= 50:
        grade = "D"
    else:
        grade = "F"

    return {
        "predicted_student_performance": pred_score,
        "estimated_grade_score_range": {"lower": lower, "upper": upper, "grade": grade},
        "selected_model": _metadata["best_model_name"],
        "model_rmse": round(float(_metadata["rmse"]), 2),
        "model_r2": round(float(_metadata["r2"]), 4),
        "student_age": req.student_age,
        "sleep_hours": req.sleep_hours,
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
            "score_distribution": "/results/image/score_distribution.png",
            "correlation_heatmap": "/results/image/correlation_heatmap.png",
            "feature_importance": "/results/image/feature_importance.png",
            "actual_vs_predicted": "/results/image/actual_vs_predicted.png",
            "residual_distribution": "/results/image/residual_distribution.png",
            "subject_wise_performance": "/results/image/subject_wise_performance.png",
            "attendance_vs_score": "/results/image/attendance_vs_score.png",
            "study_hours_impact": "/results/image/study_hours_impact.png",
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
    train_student_performance_model()
    load_student_performance_assets()


@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(_background_retrain)
    return {"status": "accepted", "message": "Student performance model retraining started."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Student Performance Prediction unified ML pipeline and backend")
    parser.add_argument("--train", action="store_true", help="Force train the model and regenerate results")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI port (default: 8000)")
    args = parser.parse_args()

    if args.train or not os.path.exists(MODEL_FILE) or not os.path.exists(METADATA_FILE):
        try:
            train_student_performance_model()
            load_student_performance_assets()
        except Exception as exc:
            print(f"[ERROR] Training failed: {exc}")
            sys.exit(1)

    if "--train" not in sys.argv:
        import uvicorn
        print(f"[INFO] Starting Student Performance Prediction API server on port {args.port}...")
        uvicorn.run("student_performance_prediction:app", host="127.0.0.1", port=args.port, reload=False)
