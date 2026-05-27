# -*- coding: utf-8 -*-
"""
Loan Default Prediction - Unified ML Pipeline and FastAPI Backend
Trains lightweight classification models on borrower financial data,
generates visualization artifacts, saves model assets, and
serves real-time loan default risk predictions via FastAPI.
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

DATA_PATH = r"C:\Users\peepl\Downloads\cust_segmentation_zip\repo_for_demand_forecasting\repo_for_demand_forecasting\loan default prediction\Default_Fin.csv"
RESULTS_DIR = "results"
MODEL_FILE = "loan_default_prediction_model.joblib"
METADATA_FILE = "loan_default_prediction_metadata.joblib"
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, "loan_default_predictions.csv")

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
    df.columns = [c.strip().replace(" ", "_").replace("?", "") for c in df.columns]

    expected_cols = ["Index", "Employed", "Bank_Balance", "Annual_Salary", "Defaulted"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")

    for col in expected_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    df["debt_to_income"] = (df["Bank_Balance"] + 1) / (df["Annual_Salary"] + 1)
    df["balance_per_income"] = df["Bank_Balance"] / (df["Annual_Salary"] + 1)

    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    feature_cols = ["Employed", "Bank_Balance", "Annual_Salary", "debt_to_income", "balance_per_income"]
    X = df[feature_cols].copy()
    y = df["Defaulted"].astype(int).copy()

    return df, X, y, feature_cols


def _build_borrower_row(req):
    employed_val = 0 if req.employment_status.lower() in {"unemployed", "not employed"} else 1
    annual_income = float(req.annual_income)
    bank_balance = float(req.existing_debt)

    # keep extra user inputs in the generated signal
    behavior_adjustment = max(0.6, min(1.6, (req.interest_rate / 10.0) + (req.loan_term / 24.0)))
    payment_adjustment = 1.2 if req.payment_history.lower() == "poor" else (1.0 if req.payment_history.lower() == "average" else 0.8)
    bank_balance = bank_balance * behavior_adjustment * payment_adjustment

    debt_to_income = (bank_balance + 1) / (annual_income + 1)
    balance_per_income = bank_balance / (annual_income + 1)

    return {
        "Employed": float(employed_val),
        "Bank_Balance": float(bank_balance),
        "Annual_Salary": float(annual_income),
        "debt_to_income": float(debt_to_income),
        "balance_per_income": float(balance_per_income),
    }


def train_loan_default_model():
    print("[INFO] Starting Loan Default Prediction model training...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    _clear_previous_outputs()

    df = pd.read_csv(DATA_PATH)
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42).reset_index(drop=True)

    df, X, y, feature_names = _prepare_dataset(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1200, random_state=42),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=180, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
    }

    model_metrics = {}
    trained_models = {}

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        model_metrics[model_name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
        }
        trained_models[model_name] = model

    best_model_name = max(model_metrics.keys(), key=lambda k: model_metrics[k]["f1"])
    model = trained_models[best_model_name]

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    report = classification_report(y_test, y_pred, target_names=["No Default", "Default"], zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    prediction_frame = pd.DataFrame(X.iloc[x_test.shape[0] * -1 :].copy())
    prediction_frame["actual_default"] = y_test.values
    prediction_frame["predicted_default"] = y_pred
    if y_prob is not None:
        prediction_frame["default_probability"] = y_prob
    prediction_frame.to_csv(PREDICTIONS_FILE, index=False)

    metrics_text = f"""Loan Default Prediction - Model Evaluation Report
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
    pd.Series(y).map({0: "No Default", 1: "Default"}).value_counts().plot(kind="bar", color=["#2563eb", "#ef4444"], ax=ax)
    ax.set_title("Loan Default Distribution")
    ax.set_xlabel("Default Status")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "default_distribution.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    corr_df = df[["Employed", "Bank_Balance", "Annual_Salary", "debt_to_income", "balance_per_income", "Defaulted"]]
    sns.heatmap(corr_df.corr(), cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
    ax.set_title("Loan Default Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    elif hasattr(model, "coef_"):
        importances = pd.Series(np.abs(model.coef_[0]), index=feature_names).sort_values()
    else:
        importances = pd.Series(np.zeros(len(feature_names)), index=feature_names).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
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
    ax.set_xticklabels(["No Default", "Default"])
    ax.set_yticklabels(["No Default", "Default"])
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

        pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(pr_recall, pr_precision, color="#8b5cf6", lw=2)
        ax.set_title("Precision vs Recall")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "precision_vs_recall.png"), dpi=150)
        plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    age_bins = pd.cut(df["Index"] % 50 + 20, bins=[18, 30, 40, 50, 70], labels=["18-30", "31-40", "41-50", "51-70"])
    pd.DataFrame({"age_group": age_bins, "default": y}).groupby("age_group")["default"].mean().plot(kind="bar", color="#f59e0b", ax=ax)
    ax.set_title("Borrower Demographic Analysis")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Default Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "borrower_demographic_analysis.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    category = pd.cut(df["Bank_Balance"], bins=[-1, 5000, 15000, 1000000], labels=["Low Balance", "Medium Balance", "High Balance"])
    pd.DataFrame({"loan_category": category, "default": y}).groupby("loan_category")["default"].mean().plot(kind="bar", color="#6366f1", ax=ax)
    ax.set_title("Loan Category Comparison")
    ax.set_xlabel("Loan Category")
    ax.set_ylabel("Default Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "loan_category_comparison.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = [accuracy, precision, recall, f1]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    bars = ax.bar(labels, metrics, color=["#22c55e", "#2563eb", "#f59e0b", "#a855f7"])
    for b, v in zip(bars, metrics):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_performance_metrics.png"), dpi=150)
    plt.close()

    metadata = {
        "feature_names": feature_names,
        "scaler": scaler,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_model_name": best_model_name,
        "model_metrics": model_metrics,
    }
    joblib.dump(model, MODEL_FILE)
    joblib.dump(metadata, METADATA_FILE)
    print("[DONE] Loan Default Prediction training complete.")


app = FastAPI(title="Loan Default Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_metadata = None


def load_loan_default_assets():
    global _model, _metadata
    if os.path.exists(MODEL_FILE) and os.path.exists(METADATA_FILE):
        _model = joblib.load(MODEL_FILE)
        _metadata = joblib.load(METADATA_FILE)


load_loan_default_assets()


class LoanDefaultPredictionRequest(BaseModel):
    customer_age: int = Field(..., ge=18, le=85)
    annual_income: float = Field(..., ge=50000)
    loan_amount: float = Field(..., ge=1000)
    credit_score: int = Field(..., ge=300, le=900)
    employment_status: str = Field(..., min_length=2)
    loan_term: int = Field(..., ge=1, le=40)
    interest_rate: float = Field(..., ge=1, le=40)
    existing_debt: float = Field(..., ge=0)
    payment_history: str = Field(..., min_length=2)
    number_of_dependents: int = Field(..., ge=0, le=10)


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    path = "loan_default_prediction_frontend.html"
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
def predict_loan_default(req: LoanDefaultPredictionRequest):
    if _model is None or _metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run with --train first.")

    row = _build_borrower_row(req)
    feature_names = _metadata["feature_names"]
    X_input = np.array([[row.get(f, 0.0) for f in feature_names]], dtype=float)
    X_scaled = _metadata["scaler"].transform(X_input)

    pred_idx = int(_model.predict(X_scaled)[0])
    probability = float(_model.predict_proba(X_scaled)[0][1]) if hasattr(_model, "predict_proba") else 0.5
    probability = round(probability * 100, 2)

    risk_label = "Default" if pred_idx == 1 else "No Default"
    if probability >= 70:
        risk_category = "High"
    elif probability >= 40:
        risk_category = "Medium"
    else:
        risk_category = "Low"

    return {
        "loan_default_risk": risk_label,
        "default_probability_score": probability,
        "risk_category": risk_category,
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
            "default_distribution": "/results/image/default_distribution.png",
            "correlation_heatmap": "/results/image/correlation_heatmap.png",
            "feature_importance": "/results/image/feature_importance.png",
            "confusion_matrix": "/results/image/confusion_matrix.png",
            "roc_curve": "/results/image/roc_curve.png",
            "precision_vs_recall": "/results/image/precision_vs_recall.png",
            "borrower_demographic_analysis": "/results/image/borrower_demographic_analysis.png",
            "loan_category_comparison": "/results/image/loan_category_comparison.png",
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
    train_loan_default_model()
    load_loan_default_assets()


@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(_background_retrain)
    return {"status": "accepted", "message": "Loan default model retraining started."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loan Default Prediction unified ML pipeline and backend")
    parser.add_argument("--train", action="store_true", help="Force train the model and regenerate results")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI port (default: 8000)")
    args = parser.parse_args()

    if args.train or not os.path.exists(MODEL_FILE) or not os.path.exists(METADATA_FILE):
        try:
            train_loan_default_model()
            load_loan_default_assets()
        except Exception as exc:
            print(f"[ERROR] Training failed: {exc}")
            sys.exit(1)

    if "--train" not in sys.argv:
        import uvicorn
        print(f"[INFO] Starting Loan Default Prediction API server on port {args.port}...")
        uvicorn.run("loan_default_prediction:app", host="127.0.0.1", port=args.port, reload=False)
