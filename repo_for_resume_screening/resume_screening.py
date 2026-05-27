# -*- coding: utf-8 -*-
"""
AI Resume Screening - Unified ML Pipeline and FastAPI Backend
Trains lightweight classification models on candidate resume data,
generates visualization artifacts, saves model assets, and
serves real-time screening predictions via FastAPI.
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
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

DATA_PATH = r"C:\Users\peepl\Downloads\cust_segmentation_zip\repo_for_demand_forecasting\repo_for_demand_forecasting\resume_screening\ai_resume_screening.csv"
RESULTS_DIR = "results"
MODEL_FILE = "resume_screening_model.joblib"
METADATA_FILE = "resume_screening_metadata.joblib"
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, "resume_screening_predictions.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)


def _build_candidate_row(req, education_encoder):
    education_value = req.education if req.education else "Bachelors"
    if education_value in education_encoder.classes_:
        education_level = int(education_encoder.transform([education_value])[0])
    else:
        education_level = int(np.median(np.arange(len(education_encoder.classes_))))

    inferred_resume_length = req.resume_length if req.resume_length is not None else max(120, len(req.resume_text or ""))
    inferred_project_count = req.project_count if req.project_count is not None else max(1, req.certifications + 2)
    inferred_github_activity = req.github_activity if req.github_activity is not None else max(10, len(req.candidate_skills.split(",")) * 25)

    return {
        "years_experience": float(req.experience_level),
        "skills_match_score": float(req.skills_match_score),
        "education_level": float(education_level),
        "project_count": float(inferred_project_count),
        "resume_length": float(inferred_resume_length),
        "github_activity": float(inferred_github_activity),
    }


def train_resume_model():
    print("[INFO] Starting AI Resume Screening model training...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if len(df) > 25000:
        df = df.sample(n=25000, random_state=42).reset_index(drop=True)

    expected_cols = [
        "years_experience",
        "skills_match_score",
        "education_level",
        "project_count",
        "resume_length",
        "github_activity",
        "shortlisted",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")

    df = df[expected_cols].copy()
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    education_encoder = LabelEncoder()
    df["education_level"] = education_encoder.fit_transform(df["education_level"].astype(str))

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df["shortlisted"].astype(str))

    X = df.drop(columns=["shortlisted"])
    feature_names = list(X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=180, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
        "Naive Bayes": GaussianNB(),
    }

    model_scores = {}
    trained_models = {}
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        model_scores[model_name] = accuracy_score(y_test, pred)
        trained_models[model_name] = model

    best_model_name = max(model_scores, key=model_scores.get)
    model = trained_models[best_model_name]

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=list(target_encoder.classes_))
    cm = confusion_matrix(y_test, y_pred)

    prediction_frame = X.iloc[x_test.shape[0] * -1 :].copy()
    prediction_frame["actual_label"] = target_encoder.inverse_transform(y_test)
    prediction_frame["predicted_label"] = target_encoder.inverse_transform(y_pred)
    prediction_frame.to_csv(PREDICTIONS_FILE, index=False)

    metrics_text = f"""AI Resume Screening - Model Evaluation Report
=================================================
Selected Model           : {best_model_name}
Accuracy                 : {accuracy:.4f} ({accuracy * 100:.2f}%)
Dataset Path             : {DATA_PATH}
Total Samples            : {len(df)}
Training Samples         : {len(x_train)}
Testing Samples          : {len(x_test)}

Model Accuracy Comparison
-------------------------
{os.linesep.join([f"{k}: {v:.4f}" for k, v in model_scores.items()])}

Classification Report
---------------------
{report}
"""
    with open(os.path.join(RESULTS_DIR, "accuracy_results.txt"), "w", encoding="utf-8") as f:
        f.write(metrics_text)
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    sns.set_theme(style="darkgrid")

    target_names = list(target_encoder.classes_)
    fig, ax = plt.subplots(figsize=(8, 5))
    pd.Series(df["shortlisted"]).value_counts().plot(kind="bar", color=["#6366f1", "#10b981"], ax=ax)
    ax.set_title("Resume Category Distribution")
    ax.set_xlabel("Candidate Category")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "resume_category_distribution.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.drop(columns=["shortlisted"]).corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Candidate Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["skills_match_score"], bins=30, kde=True, color="#8b5cf6", ax=ax)
    ax.set_title("Skill Frequency Visualization")
    ax.set_xlabel("Skills Match Score")
    ax.set_ylabel("Candidate Count")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "skill_frequency.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    if y_prob is not None and len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, color="#10b981", lw=2, label=f"ROC Curve (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#ef4444")
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=150)
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

    fig, ax = plt.subplots(figsize=(8, 5))
    pd.Series(model_scores).sort_values(ascending=False).plot(kind="bar", color="#6366f1", ax=ax)
    ax.set_title("Model Accuracy Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_comparison.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    pd.Series(target_encoder.inverse_transform(y_pred)).value_counts().plot(
        kind="bar", color=["#0ea5e9", "#22c55e"], ax=ax
    )
    ax.set_title("Candidate Category Prediction Chart")
    ax.set_xlabel("Predicted Category")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "candidate_prediction_chart.png"), dpi=150)
    plt.close()

    metadata = {
        "feature_names": feature_names,
        "scaler": scaler,
        "education_encoder": education_encoder,
        "target_encoder": target_encoder,
        "accuracy": accuracy,
        "best_model_name": best_model_name,
        "model_scores": model_scores,
    }
    joblib.dump(model, MODEL_FILE)
    joblib.dump(metadata, METADATA_FILE)
    print("[DONE] AI Resume Screening training complete.")


app = FastAPI(title="AI Resume Screening API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_metadata = None


def load_resume_assets():
    global _model, _metadata
    if os.path.exists(MODEL_FILE) and os.path.exists(METADATA_FILE):
        _model = joblib.load(MODEL_FILE)
        _metadata = joblib.load(METADATA_FILE)


load_resume_assets()


class ResumeScreeningRequest(BaseModel):
    resume_text: str = Field(..., min_length=30, description="Candidate resume summary text")
    candidate_skills: str = Field(..., min_length=2, description="Comma-separated skills")
    experience_level: float = Field(..., ge=0, le=40, description="Years of candidate experience")
    education: str = Field("Bachelors", description="Candidate education level")
    certifications: int = Field(0, ge=0, le=30, description="Number of certifications")
    job_role_applied: str = Field(..., min_length=2, description="Target job role")
    skills_match_score: float = Field(70.0, ge=0, le=100, description="Skills match percentage")
    project_count: int | None = Field(default=None, ge=0, le=100)
    resume_length: int | None = Field(default=None, ge=50, le=5000)
    github_activity: int | None = Field(default=None, ge=0, le=5000)


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    path = "resume_screening_frontend.html"
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
def predict_candidate(req: ResumeScreeningRequest):
    if _model is None or _metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run with --train first.")

    row = _build_candidate_row(req, _metadata["education_encoder"])
    feature_names = _metadata["feature_names"]
    X_input = np.array([[row.get(f, 0.0) for f in feature_names]], dtype=float)
    X_scaled = _metadata["scaler"].transform(X_input)

    pred_idx = int(_model.predict(X_scaled)[0])
    pred_label = _metadata["target_encoder"].inverse_transform([pred_idx])[0]

    if hasattr(_model, "predict_proba"):
        probabilities = _model.predict_proba(X_scaled)[0]
        suitability_score = round(float(np.max(probabilities) * 100), 2)
    else:
        suitability_score = 75.0

    screening_result = "Shortlist Recommended" if str(pred_label).lower() == "yes" else "Needs Manual Review"
    return {
        "predicted_candidate_category": str(pred_label),
        "resume_screening_result": screening_result,
        "candidate_suitability_score": suitability_score,
        "applied_role": req.job_role_applied,
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
            "resume_category_distribution": "/results/image/resume_category_distribution.png",
            "correlation_heatmap": "/results/image/correlation_heatmap.png",
            "skill_frequency": "/results/image/skill_frequency.png",
            "confusion_matrix": "/results/image/confusion_matrix.png",
            "roc_curve": "/results/image/roc_curve.png",
            "feature_importance": "/results/image/feature_importance.png",
            "accuracy_comparison": "/results/image/accuracy_comparison.png",
            "candidate_prediction_chart": "/results/image/candidate_prediction_chart.png",
        },
    }


@app.get("/results/image/{image_name}")
def serve_plot(image_name: str):
    path = os.path.join(RESULTS_DIR, image_name)
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Image not found.")


def _background_retrain():
    train_resume_model()
    load_resume_assets()


@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(_background_retrain)
    return {"status": "accepted", "message": "AI Resume Screening model retraining started."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Resume Screening unified ML pipeline and backend")
    parser.add_argument("--train", action="store_true", help="Force train the model and regenerate results")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI port (default: 8000)")
    args = parser.parse_args()

    if args.train or not os.path.exists(MODEL_FILE) or not os.path.exists(METADATA_FILE):
        try:
            train_resume_model()
            load_resume_assets()
        except Exception as exc:
            print(f"[ERROR] Training failed: {exc}")
            sys.exit(1)

    if "--train" not in sys.argv:
        import uvicorn
        print(f"[INFO] Starting AI Resume Screening API server on port {args.port}...")
        uvicorn.run("resume_screening:app", host="127.0.0.1", port=args.port, reload=False)
