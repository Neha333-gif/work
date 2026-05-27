# -*- coding: utf-8 -*-
"""
Sales Conversion Prediction - Unified ML Pipeline and FastAPI Backend
Trains lightweight classification models on customer lead data,
generates visualization artifacts, saves model assets, and
serves real-time sales conversion predictions via FastAPI.
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

DATA_PATH = r"C:\Users\peepl\Downloads\cust_segmentation_zip\repo_for_demand_forecasting\repo_for_demand_forecasting\sales conversion\KAG_conversion_data.csv"
RESULTS_DIR = "results"
MODEL_FILE = "sales_conversion_prediction_model.joblib"
METADATA_FILE = "sales_conversion_prediction_metadata.joblib"
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, "sales_conversion_predictions.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)


def _clear_previous_outputs():
    for name in os.listdir(RESULTS_DIR):
        if name.endswith((".png", ".txt", ".csv", ".json")):
            try:
                os.remove(os.path.join(RESULTS_DIR, name))
            except OSError:
                pass


def _age_bucket(age_value):
    age_value = int(age_value)
    if age_value <= 34:
        return "30-34"
    if age_value <= 39:
        return "35-39"
    if age_value <= 44:
        return "40-44"
    return "45-49"


def _encode_with_fallback(encoder: LabelEncoder, value: str):
    value = str(value).strip().lower()
    classes = [str(c).strip().lower() for c in encoder.classes_]
    if value in classes:
        return float(classes.index(value))
    return float(0)


def _prepare_dataset(raw_df: pd.DataFrame):
    df = raw_df.copy()
    df.columns = [c.strip() for c in df.columns]

    if "ad_id" in df.columns:
        df.drop(columns=["ad_id"], inplace=True)

    expected_cols = [
        "xyz_campaign_id",
        "fb_campaign_id",
        "age",
        "gender",
        "interest",
        "Impressions",
        "Clicks",
        "Spent",
        "Total_Conversion",
        "Approved_Conversion",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")

    df["Converted"] = (df["Approved_Conversion"] > 0).astype(int)

    encoders = {}
    for col in ["age", "gender"]:
        enc = LabelEncoder()
        df[col] = enc.fit_transform(df[col].astype(str).str.strip())
        encoders[col] = enc

    df["Lead_Source"] = pd.cut(
        df["fb_campaign_id"],
        bins=[-1, df["fb_campaign_id"].quantile(0.25), df["fb_campaign_id"].quantile(0.5),
              df["fb_campaign_id"].quantile(0.75), df["fb_campaign_id"].max() + 1],
        labels=["Facebook", "Google", "Email", "Referral"],
    )
    enc = LabelEncoder()
    df["Lead_Source"] = enc.fit_transform(df["Lead_Source"].astype(str))
    encoders["Lead_Source"] = enc

    df["Customer_Segment"] = df["age"].astype(str) + "_" + df["gender"].astype(str)
    enc = LabelEncoder()
    df["Customer_Segment"] = enc.fit_transform(df["Customer_Segment"])
    encoders["Customer_Segment"] = enc

    numeric_cols = ["interest", "Impressions", "Clicks", "Spent", "Total_Conversion"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    feature_cols = [
        "age",
        "gender",
        "interest",
        "Impressions",
        "Clicks",
        "Spent",
        "Total_Conversion",
        "Lead_Source",
        "Customer_Segment",
    ]
    X = df[feature_cols].copy()
    y = df["Converted"].astype(int).copy()

    return df, X, y, encoders, feature_cols


def _interest_from_level(level: str) -> float:
    level = str(level).strip().lower()
    if level in {"high", "strong"}:
        return 25.0
    if level in {"medium", "moderate"}:
        return 15.0
    return 8.0


def _campaign_id_from_type(campaign_type: str) -> int:
    mapping = {
        "awareness": 916,
        "retargeting": 936,
        "promotion": 1178,
    }
    key = str(campaign_type).strip().lower()
    return mapping.get(key, 936)


def _lead_source_id(source: str) -> int:
    mapping = {"facebook": 0, "google": 1, "email": 2, "referral": 3}
    return float(mapping.get(str(source).strip().lower(), 0))


def _build_lead_row(req, encoders):
    age_bucket = _age_bucket(req.customer_age)
    gender = "M" if req.region.lower() in {"north", "south", "east"} else "F"
    age_enc = int(_encode_with_fallback(encoders["age"], age_bucket))
    gender_enc = int(_encode_with_fallback(encoders["gender"], gender))
    segment_key = f"{age_enc}_{gender_enc}"
    return {
        "age": float(age_enc),
        "gender": float(gender_enc),
        "interest": _interest_from_level(req.product_interest_level),
        "Impressions": float(req.website_visit_duration) * 1000.0,
        "Clicks": float(req.number_of_interactions),
        "Spent": float(req.annual_income) / 10000.0,
        "Total_Conversion": float(req.previous_purchase_history),
        "Lead_Source": _lead_source_id(req.lead_source),
        "Customer_Segment": _encode_with_fallback(encoders["Customer_Segment"], segment_key),
    }


def _conversion_category(probability: float) -> str:
    if probability >= 70:
        return "High Conversion"
    if probability >= 40:
        return "Medium Conversion"
    return "Low Conversion"


def train_sales_conversion_model():
    print("[INFO] Starting Sales Conversion Prediction model training...")
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

    report = classification_report(
        y_test, y_pred, target_names=["No Conversion", "Converted"], zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    _, X_test_raw, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    prediction_frame = X_test_raw.copy()
    prediction_frame["actual_conversion"] = y_test.values
    prediction_frame["predicted_conversion"] = y_pred
    if y_prob is not None:
        prediction_frame["conversion_probability"] = y_prob
    prediction_frame.to_csv(PREDICTIONS_FILE, index=False)

    metrics_text = f"""Sales Conversion Prediction - Model Evaluation Report
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
    with open(os.path.join(RESULTS_DIR, "evaluation_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}\n")

    sns.set_theme(style="darkgrid")

    fig, ax = plt.subplots(figsize=(8, 5))
    pd.Series(y).map({0: "No Conversion", 1: "Converted"}).value_counts().plot(
        kind="bar", color=["#2563eb", "#22c55e"], ax=ax
    )
    ax.set_title("Sales Conversion Distribution")
    ax.set_xlabel("Conversion Status")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "conversion_distribution.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    corr_df = df[feature_names + ["Converted"]].copy()
    for col in ["age", "gender", "Lead_Source", "Customer_Segment"]:
        corr_df[col] = pd.to_numeric(corr_df[col], errors="coerce")
    sns.heatmap(corr_df.corr(), cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
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
    ax.set_xticklabels(["No Conversion", "Converted"])
    ax.set_yticklabels(["No Conversion", "Converted"])
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

    lead_labels = encoders["Lead_Source"].classes_
    lead_map = {i: str(label) for i, label in enumerate(lead_labels)}
    fig, ax = plt.subplots(figsize=(8, 5))
    df.groupby(df["Lead_Source"].map(lead_map))["Converted"].mean().plot(kind="bar", color="#6366f1", ax=ax)
    ax.set_title("Lead Source Analysis")
    ax.set_xlabel("Lead Source")
    ax.set_ylabel("Conversion Rate")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "lead_source_analysis.png"), dpi=150)
    plt.close()

    seg_labels = encoders["Customer_Segment"].classes_
    seg_map = {i: str(label) for i, label in enumerate(seg_labels)}
    top_segments = df["Customer_Segment"].value_counts().head(6).index
    seg_df = df[df["Customer_Segment"].isin(top_segments)]
    fig, ax = plt.subplots(figsize=(9, 5))
    seg_df.groupby(seg_df["Customer_Segment"].map(seg_map))["Converted"].mean().plot(kind="bar", color="#f59e0b", ax=ax)
    ax.set_title("Customer Segmentation Comparison")
    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Conversion Rate")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "customer_segmentation_comparison.png"), dpi=150)
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
        "encoders": encoders,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_model_name": best_model_name,
    }
    joblib.dump(model, MODEL_FILE)
    joblib.dump(metadata, METADATA_FILE)
    print("[DONE] Sales Conversion Prediction training complete.")


app = FastAPI(title="Sales Conversion Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_metadata = None


def load_sales_conversion_assets():
    global _model, _metadata
    if os.path.exists(MODEL_FILE) and os.path.exists(METADATA_FILE):
        _model = joblib.load(MODEL_FILE)
        _metadata = joblib.load(METADATA_FILE)


load_sales_conversion_assets()


class SalesConversionPredictionRequest(BaseModel):
    customer_age: int = Field(..., ge=18, le=70)
    annual_income: float = Field(..., ge=1000)
    lead_source: str = Field(..., min_length=2)
    number_of_interactions: int = Field(..., ge=0)
    product_interest_level: str = Field(..., min_length=2)
    marketing_campaign_type: str = Field(..., min_length=2)
    website_visit_duration: float = Field(..., ge=0)
    previous_purchase_history: int = Field(..., ge=0)
    region: str = Field(..., min_length=2)


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    path = "sales_conversion_prediction_frontend.html"
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
def predict_sales_conversion(req: SalesConversionPredictionRequest):
    if _model is None or _metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run with --train first.")

    row = _build_lead_row(req, _metadata["encoders"])
    feature_names = _metadata["feature_names"]
    X_input = np.array([[row.get(f, 0.0) for f in feature_names]], dtype=float)
    X_scaled = _metadata["scaler"].transform(X_input)

    pred_idx = int(_model.predict(X_scaled)[0])
    probability = float(_model.predict_proba(X_scaled)[0][1]) if hasattr(_model, "predict_proba") else 0.5
    probability_pct = round(probability * 100, 2)

    outcome = "Converted" if pred_idx == 1 else "Not Converted"
    category = _conversion_category(probability_pct)

    return {
        "sales_conversion_prediction": outcome,
        "conversion_probability_score": probability_pct,
        "customer_conversion_category": category,
        "selected_model": _metadata["best_model_name"],
        "model_accuracy": round(float(_metadata["accuracy"]) * 100, 2),
        "marketing_campaign_type": req.marketing_campaign_type,
        "region": req.region,
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
            "conversion_distribution": "/results/image/conversion_distribution.png",
            "correlation_heatmap": "/results/image/correlation_heatmap.png",
            "feature_importance": "/results/image/feature_importance.png",
            "confusion_matrix": "/results/image/confusion_matrix.png",
            "roc_curve": "/results/image/roc_curve.png",
            "precision_vs_recall": "/results/image/precision_vs_recall.png",
            "lead_source_analysis": "/results/image/lead_source_analysis.png",
            "customer_segmentation_comparison": "/results/image/customer_segmentation_comparison.png",
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
    train_sales_conversion_model()
    load_sales_conversion_assets()


@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(_background_retrain)
    return {"status": "accepted", "message": "Sales conversion model retraining started."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sales Conversion Prediction unified ML pipeline and backend")
    parser.add_argument("--train", action="store_true", help="Force train the model and regenerate results")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI port (default: 8000)")
    args = parser.parse_args()

    if args.train or not os.path.exists(MODEL_FILE) or not os.path.exists(METADATA_FILE):
        try:
            train_sales_conversion_model()
            load_sales_conversion_assets()
        except Exception as exc:
            print(f"[ERROR] Training failed: {exc}")
            sys.exit(1)

    if "--train" not in sys.argv:
        import uvicorn
        print(f"[INFO] Starting Sales Conversion Prediction API server on port {args.port}...")
        uvicorn.run("sales_conversion_prediction:app", host="127.0.0.1", port=args.port, reload=False)
