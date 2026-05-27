# -*- coding: utf-8 -*-
"""
Loan Attrition Prediction - Unified ML Pipeline and FastAPI Backend
Trains lightweight classification models on customer loan account data,
generates visualization artifacts, saves model assets, and
serves real-time loan attrition risk predictions via FastAPI.
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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

DATA_PATH = r"C:\Users\peepl\Downloads\cust_segmentation_zip\repo_for_demand_forecasting\repo_for_demand_forecasting\loan_prediction_system\loan_approval_dataset.csv"
RESULTS_DIR = "results"
MODEL_FILE = "loan_attrition_prediction_model.joblib"
METADATA_FILE = "loan_attrition_prediction_metadata.joblib"
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, "loan_attrition_predictions.csv")

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
        "loan_id",
        "no_of_dependents",
        "education",
        "self_employed",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value",
        "loan_status",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")

    df["loan_status"] = df["loan_status"].astype(str).str.strip()

    encoders = {}
    for col in ["education", "self_employed", "loan_status"]:
        enc = LabelEncoder()
        df[col] = enc.fit_transform(df[col].astype(str).str.strip())
        encoders[col] = enc

    numeric_cols = [c for c in expected_cols if c != "loan_status"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    feature_cols = [c for c in expected_cols if c not in ["loan_status", "loan_id"]]

    X = df[feature_cols].copy()
    y = df["loan_status"].copy()

    return df, X, y, encoders, feature_cols


def _build_loan_row(req, encoders):
    education_text = "Graduate" if req.credit_score >= 650 else "Not Graduate"
    self_emp_text = "Yes" if req.employment_status.lower() in {"self employed", "business owner", "freelancer"} else "No"

    education_encoder = encoders["education"]
    self_emp_encoder = encoders["self_employed"]

    if education_text not in education_encoder.classes_:
        education_text = str(education_encoder.classes_[0])
    if self_emp_text not in self_emp_encoder.classes_:
        self_emp_text = str(self_emp_encoder.classes_[0])

    no_of_dependents = min(5, max(0, int(req.existing_debt // 300000)))

    residential_assets_value = max(100000, int(req.annual_income * 0.40))
    commercial_assets_value = max(50000, int(req.existing_debt * 0.50))
    luxury_assets_value = max(100000, int(req.loan_amount * 0.70))
    bank_asset_value = max(50000, int(req.annual_income * 0.25))

    return {
        "no_of_dependents": float(no_of_dependents),
        "education": float(education_encoder.transform([education_text])[0]),
        "self_employed": float(self_emp_encoder.transform([self_emp_text])[0]),
        "income_annum": float(req.annual_income),
        "loan_amount": float(req.loan_amount),
        "loan_term": float(req.loan_tenure),
        "cibil_score": float(req.credit_score),
        "residential_assets_value": float(residential_assets_value),
        "commercial_assets_value": float(commercial_assets_value),
        "luxury_assets_value": float(luxury_assets_value),
        "bank_asset_value": float(bank_asset_value),
    }


def train_loan_attrition_model():
    print("[INFO] Starting Loan Attrition Prediction model training...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    _clear_previous_outputs()

    df = pd.read_csv(DATA_PATH)
    if len(df) > 30000:
        df = df.sample(n=30000, random_state=42).reset_index(drop=True)

    df, X, y, encoders, feature_names = _prepare_dataset(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1200, random_state=42),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=220, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
    }

    model_metrics = {}
    trained_models = {}

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        model_metrics[model_name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
        }
        trained_models[model_name] = model

    best_model_name = max(model_metrics.keys(), key=lambda k: model_metrics[k]["f1"])
    model = trained_models[best_model_name]

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    target_encoder = encoders["loan_status"]
    class_names = [str(c) for c in target_encoder.classes_]
    report = classification_report(y_test, y_pred, target_names=class_names)
    cm = confusion_matrix(y_test, y_pred)

    prediction_frame = pd.DataFrame(X.iloc[x_test.shape[0] * -1 :].copy())
    prediction_frame["actual_loan_status"] = target_encoder.inverse_transform(y_test)
    prediction_frame["predicted_loan_status"] = target_encoder.inverse_transform(y_pred)
    if y_prob is not None:
        prediction_frame["attrition_probability"] = y_prob
    prediction_frame.to_csv(PREDICTIONS_FILE, index=False)

    metrics_text = f"""Loan Attrition Prediction - Model Evaluation Report
=================================================
Selected Model           : {best_model_name}
Accuracy                 : {accuracy:.4f}
Precision                : {precision:.4f}
Recall                   : {recall:.4f}
F1 Score                 : {f1:.4f}
Dataset Path             : {DATA_PATH}
Total Samples            : {len(df)}
Training Samples         : {len(x_train)}
Testing Samples          : {len(x_test)}

Model Comparison
----------------
{os.linesep.join([f"{k}: acc={v['accuracy']:.4f}, pre={v['precision']:.4f}, rec={v['recall']:.4f}, f1={v['f1']:.4f}" for k, v in model_metrics.items()])}

Classification Report
---------------------
{report}
"""

    with open(os.path.join(RESULTS_DIR, "accuracy_results.txt"), "w", encoding="utf-8") as f:
        f.write(metrics_text)
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    sns.set_theme(style="darkgrid")

    fig, ax = plt.subplots(figsize=(8, 5))
    status_series = pd.Series(target_encoder.inverse_transform(y), name="loan_status")
    status_series.value_counts().plot(kind="bar", color=["#2563eb", "#ef4444"], ax=ax)
    ax.set_title("Loan Attrition Distribution")
    ax.set_xlabel("Loan Status")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "attrition_distribution.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 7))
    corr_cols = feature_names + ["loan_status"]
    sns.heatmap(df[corr_cols].corr(), cmap="coolwarm", fmt=".2f", annot=False, ax=ax)
    ax.set_title("Loan Attrition Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    elif hasattr(model, "coef_"):
        importances = pd.Series(np.abs(model.coef_[0]), index=feature_names).sort_values()
    else:
        importances = pd.Series(np.zeros(len(feature_names)), index=feature_names).sort_values()

    fig, ax = plt.subplots(figsize=(9, 6))
    importances.plot(kind="barh", color="#14b8a6", ax=ax)
    ax.set_title("Feature Importance Plot")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, color="#10b981", lw=2, label=f"AUC={roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#ef4444")
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=150)
        plt.close()

        precision_pts, recall_pts, _ = precision_recall_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(recall_pts, precision_pts, color="#8b5cf6", lw=2)
        ax.set_title("Precision vs Recall")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "precision_vs_recall.png"), dpi=150)
        plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    edu = pd.DataFrame({
        "education": encoders["education"].inverse_transform(df["education"]),
        "loan_status": target_encoder.inverse_transform(df["loan_status"]),
    })
    edu.groupby("education")["loan_status"].count().plot(kind="bar", color="#f59e0b", ax=ax)
    ax.set_title("Customer Demographic Analysis")
    ax.set_xlabel("Education Segment")
    ax.set_ylabel("Customer Count")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "customer_demographic_analysis.png"), dpi=150)
    plt.close()

    loan_type_df = df.copy()
    loan_type_df["loan_type"] = pd.cut(
        loan_type_df["loan_term"],
        bins=[0, 6, 12, 30],
        labels=["Short Term", "Medium Term", "Long Term"],
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    loan_type_df.groupby("loan_type")["loan_amount"].mean().plot(kind="bar", color="#6366f1", ax=ax)
    ax.set_title("Loan Type Comparison")
    ax.set_xlabel("Loan Type")
    ax.set_ylabel("Average Loan Amount")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "loan_type_comparison.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    metric_names = ["Accuracy", "Precision", "Recall", "F1"]
    metric_values = [accuracy, precision, recall, f1]
    bars = ax.bar(metric_names, metric_values, color=["#22c55e", "#2563eb", "#f59e0b", "#a855f7"])
    for bar, value in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_performance_metrics.png"), dpi=150)
    plt.close()

    metadata = {
        "feature_names": feature_names,
        "scaler": scaler,
        "encoders": encoders,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_model_name": best_model_name,
        "model_metrics": model_metrics,
    }
    joblib.dump(model, MODEL_FILE)
    joblib.dump(metadata, METADATA_FILE)
    print("[DONE] Loan Attrition Prediction training complete.")


app = FastAPI(title="Loan Attrition Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_metadata = None


def load_loan_attrition_assets():
    global _model, _metadata
    if os.path.exists(MODEL_FILE) and os.path.exists(METADATA_FILE):
        _model = joblib.load(MODEL_FILE)
        _metadata = joblib.load(METADATA_FILE)


load_loan_attrition_assets()


class LoanAttritionPredictionRequest(BaseModel):
    customer_age: int = Field(..., ge=18, le=85)
    annual_income: float = Field(..., ge=100000)
    loan_amount: float = Field(..., ge=100000)
    credit_score: int = Field(..., ge=300, le=900)
    employment_status: str = Field(..., min_length=2)
    loan_type: str = Field(..., min_length=2)
    loan_tenure: int = Field(..., ge=2, le=30)
    emi_amount: float = Field(..., ge=1000)
    existing_debt: float = Field(..., ge=0)
    payment_history: str = Field(..., min_length=2)


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    path = "loan_attrition_prediction_frontend.html"
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
def predict_loan_attrition(req: LoanAttritionPredictionRequest):
    if _model is None or _metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run with --train first.")

    row = _build_loan_row(req, _metadata["encoders"])
    feature_names = _metadata["feature_names"]
    X_input = np.array([[row.get(f, 0.0) for f in feature_names]], dtype=float)
    X_scaled = _metadata["scaler"].transform(X_input)

    pred_idx = int(_model.predict(X_scaled)[0])
    target_encoder = _metadata["encoders"]["loan_status"]
    pred_label = str(target_encoder.inverse_transform([pred_idx])[0]).strip()

    probability = float(_model.predict_proba(X_scaled)[0][pred_idx]) if hasattr(_model, "predict_proba") else 0.5
    probability = round(probability * 100, 2)

    if pred_label.lower() == "rejected":
        risk_label = "High"
    elif probability >= 55:
        risk_label = "Medium"
    else:
        risk_label = "Low"

    return {
        "loan_attrition_risk": pred_label,
        "attrition_probability_score": probability,
        "risk_category": risk_label,
        "selected_model": _metadata["best_model_name"],
        "model_accuracy": round(float(_metadata["accuracy"]) * 100, 2),
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
            "attrition_distribution": "/results/image/attrition_distribution.png",
            "correlation_heatmap": "/results/image/correlation_heatmap.png",
            "feature_importance": "/results/image/feature_importance.png",
            "confusion_matrix": "/results/image/confusion_matrix.png",
            "roc_curve": "/results/image/roc_curve.png",
            "precision_vs_recall": "/results/image/precision_vs_recall.png",
            "customer_demographic_analysis": "/results/image/customer_demographic_analysis.png",
            "loan_type_comparison": "/results/image/loan_type_comparison.png",
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
    train_loan_attrition_model()
    load_loan_attrition_assets()


@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(_background_retrain)
    return {"status": "accepted", "message": "Loan attrition model retraining started."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loan Attrition Prediction unified ML pipeline and backend")
    parser.add_argument("--train", action="store_true", help="Force train the model and regenerate results")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI port (default: 8000)")
    args = parser.parse_args()

    if args.train or not os.path.exists(MODEL_FILE) or not os.path.exists(METADATA_FILE):
        try:
            train_loan_attrition_model()
            load_loan_attrition_assets()
        except Exception as exc:
            print(f"[ERROR] Training failed: {exc}")
            sys.exit(1)

    if "--train" not in sys.argv:
        import uvicorn
        print(f"[INFO] Starting Loan Attrition Prediction API server on port {args.port}...")
        uvicorn.run("loan_attrition_prediction:app", host="127.0.0.1", port=args.port, reload=False)
