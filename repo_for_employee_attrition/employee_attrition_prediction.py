# -*- coding: utf-8 -*-
"""
Employee Attrition Prediction - Unified ML Pipeline and FastAPI Backend
Trains a RandomForestClassifier on MFG10YearTerminationData.csv,
generates rich visualizations, saves model artifacts, and serves
real-time attrition predictions via FastAPI.

Run:
    python employee_attrition_prediction.py --train   # train + start server
    python employee_attrition_prediction.py           # start server only
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, auc,
)
from imblearn.over_sampling import SMOTE

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# File Paths & Config
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR       = "employee_attrition_prediction_data"
DATA_PATH      = os.path.join(DATA_DIR, "MFG10YearTerminationData.csv")
RESULTS_DIR    = "results"
MODEL_FILE     = "employee_attrition_prediction_model.joblib"
METADATA_FILE  = "employee_attrition_prediction_metadata.joblib"
FRONTEND_FILE  = "employee_attrition_prediction_frontend.html"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Plot theme constants
# ──────────────────────────────────────────────────────────────────────────────

DARK_BG   = '#070f1e'
SURFACE   = '#0d1b2e'
COLOR_P   = '#6366f1'
COLOR_A   = '#10b981'
COLOR_W   = '#f59e0b'
COLOR_D   = '#f43f5e'
TEXT_COL  = '#f8fafc'
MUTED_COL = '#94a3b8'

def _set_plot_theme():
    plt.rcParams.update({
        'figure.facecolor': DARK_BG,
        'axes.facecolor':   SURFACE,
        'text.color':       TEXT_COL,
        'axes.labelcolor':  MUTED_COL,
        'xtick.color':      MUTED_COL,
        'ytick.color':      MUTED_COL,
        'axes.edgecolor':   '#1e3a5f',
        'grid.color':       '#1e3a5f',
        'axes.grid':        True,
        'grid.alpha':       0.3,
    })

# ──────────────────────────────────────────────────────────────────────────────
# 1. ML Training Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def train_attrition_model():
    print("[INFO] Starting Employee Attrition Model Training...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_PATH}'. "
            "Please ensure MFG10YearTerminationData.csv is in employee_attrition_prediction_data/."
        )

    # ── Load ──────────────────────────────────────────────────────────────────
    print("[INFO] Loading employee workforce dataset...")
    df_raw = pd.read_csv(DATA_PATH)
    print(f"  Dataset shape        : {df_raw.shape}")
    print(f"  Target distribution  :\n{df_raw['STATUS'].value_counts()}\n")

    # ── Preprocessing ─────────────────────────────────────────────────────────
    print("[INFO] Preprocessing workforce data...")
    df = df_raw.copy()

    # Drop leakage columns and non-informative identifiers
    drop_cols = [
        'EmployeeID',
        'recorddate_key',
        'birthdate_key',
        'orighiredate_key',
        'terminationdate_key',  # only populated on termination → leakage
        'termreason_desc',      # only populated on termination → leakage
        'termtype_desc',        # only populated on termination → leakage
        'gender_full',          # redundant with gender_short
        'STATUS_YEAR',          # temporal metadata, not predictive feature
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Binary encode target: ACTIVE=0, TERMINATED=1
    df['STATUS'] = (df['STATUS'] == 'TERMINATED').astype(int)

    # Encode categorical columns
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    le_map = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_map[col] = le

    df.fillna(df.median(numeric_only=True), inplace=True)
    df.drop_duplicates(inplace=True)

    print(f"  Processed shape      : {df.shape}")

    # ── Feature / Target split ────────────────────────────────────────────────
    TARGET = 'STATUS'
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    feature_names = list(X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Training samples     : {x_train.shape[0]}")
    print(f"  Test samples         : {x_test.shape[0]}")

    # ── SMOTE to balance classes ──────────────────────────────────────────────
    print("[INFO] Applying SMOTE to balance attrition classes...")
    smote = SMOTE(random_state=42)
    x_train_res, y_train_res = smote.fit_resample(x_train, y_train)
    print(f"  After SMOTE shape    : {x_train_res.shape}")

    # ── Train RandomForestClassifier ──────────────────────────────────────────
    print("[INFO] Training RandomForestClassifier (n=150, depth=10)...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train_res, y_train_res)

    y_pred      = model.predict(x_test)
    y_pred_prob = model.predict_proba(x_test)[:, 1]

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_pred_prob)

    print(f"""
  Accuracy  : {accuracy:.4f}
  Precision : {precision:.4f}
  Recall    : {recall:.4f}
  F1 Score  : {f1:.4f}
  ROC-AUC   : {roc_auc:.4f}
""")

    # ── Save accuracy report ──────────────────────────────────────────────────
    report_text = f"""Employee Attrition Prediction - Model Evaluation Report
=========================================================
Model     : RandomForestClassifier (n_estimators=150, max_depth=10)
Dataset   : MFG10YearTerminationData.csv  ({df.shape[0]} samples)
Features  : {feature_names}

Performance Metrics
-------------------
Accuracy  : {accuracy:.4f}  ({accuracy * 100:.2f}%)
Precision : {precision:.4f}
Recall    : {recall:.4f}
F1 Score  : {f1:.4f}
ROC-AUC   : {roc_auc:.4f}

Classification Report
---------------------
{classification_report(y_test, y_pred, target_names=['Active', 'Terminated'])}

Confusion Matrix
----------------
{confusion_matrix(y_test, y_pred)}
"""
    with open(os.path.join(RESULTS_DIR, "accuracy_results.txt"), "w", encoding="utf-8") as fh:
        fh.write(report_text)
    print(f"[INFO] Saved -> {RESULTS_DIR}/accuracy_results.txt")

    # ── Visualizations ────────────────────────────────────────────────────────
    _set_plot_theme()
    print("[INFO] Generating employee attrition visualizations...")

    # 1. Attrition Class Distribution
    counts = df_raw['STATUS'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = [COLOR_A, COLOR_D]
    bars = ax.bar(counts.index, counts.values, color=bar_colors, width=0.45, edgecolor='none')
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f'{val:,}', ha='center', fontsize=13, fontweight='bold', color=TEXT_COL)
    ax.set_title('Employee Attrition Class Distribution', fontsize=14, color=TEXT_COL, pad=15)
    ax.set_xlabel('Employment Status', fontsize=12)
    ax.set_ylabel('Employee Count', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'attrition_distribution.png'), dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"  Saved -> {RESULTS_DIR}/attrition_distribution.png")

    # 2. Correlation Heatmap
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt='.2f',
                linewidths=0.5, annot_kws={'size': 8}, ax=ax)
    ax.set_title('Feature Correlation Heatmap – Employee Workforce Data', fontsize=14, color=TEXT_COL, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'correlation_heatmap.png'), dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"  Saved -> {RESULTS_DIR}/correlation_heatmap.png")

    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color=COLOR_P, lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], '--', color=COLOR_D, lw=1.5, label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.12, color=COLOR_P)
    ax.set_title('ROC Curve – Employee Attrition Prediction', fontsize=14, color=TEXT_COL, pad=15)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.legend(loc='lower right', facecolor=SURFACE, edgecolor='#1e3a5f', labelcolor=TEXT_COL)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'), dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"  Saved -> {RESULTS_DIR}/roc_curve.png")

    # 4. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Active', 'Terminated'],
                yticklabels=['Active', 'Terminated'], ax=ax,
                linewidths=0.5, linecolor='#1e3a5f', cbar_kws={'shrink': 0.8})
    ax.set_title('Confusion Matrix – Employee Attrition', fontsize=14, color=TEXT_COL, pad=15)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"  Saved -> {RESULTS_DIR}/confusion_matrix.png")

    # 5. Feature Importance
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
    colors_fi   = [COLOR_P if v < importances.median() else COLOR_A for v in importances]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(importances.index, importances.values, color=colors_fi, edgecolor='none')
    ax.set_title('Feature Importance – Drivers of Employee Attrition', fontsize=14, color=TEXT_COL, pad=15)
    ax.set_xlabel('Importance Score', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'), dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"  Saved -> {RESULTS_DIR}/feature_importance.png")

    # 6. Precision-Recall Curve
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(rec_vals, prec_vals)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec_vals, prec_vals, color=COLOR_A, lw=2.5, label=f'PR Curve (AUC = {pr_auc:.3f})')
    ax.fill_between(rec_vals, prec_vals, alpha=0.12, color=COLOR_A)
    ax.set_title('Precision-Recall Curve – Employee Attrition', fontsize=14, color=TEXT_COL, pad=15)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.legend(loc='upper right', facecolor=SURFACE, edgecolor='#1e3a5f', labelcolor=TEXT_COL)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'precision_recall_curve.png'), dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"  Saved -> {RESULTS_DIR}/precision_recall_curve.png")

    # 7. Retention vs Attrition by Department (top 8)
    top_depts  = df_raw['department_name'].value_counts().head(8).index
    dept_df    = df_raw[df_raw['department_name'].isin(top_depts)]
    pivot      = dept_df.groupby(['department_name', 'STATUS']).size().unstack(fill_value=0)
    pivot      = pivot.reindex(columns=['ACTIVE', 'TERMINATED'], fill_value=0)
    x_pos      = np.arange(len(pivot))
    w          = 0.38
    fig, ax    = plt.subplots(figsize=(12, 6))
    ax.bar(x_pos - w/2, pivot['ACTIVE'],     width=w, color=COLOR_A, label='Active',     alpha=0.9, edgecolor='none')
    ax.bar(x_pos + w/2, pivot['TERMINATED'], width=w, color=COLOR_D, label='Terminated', alpha=0.9, edgecolor='none')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(pivot.index, rotation=30, ha='right', fontsize=10)
    ax.set_title('Employee Retention vs Attrition by Department', fontsize=14, color=TEXT_COL, pad=15)
    ax.set_ylabel('Employee Count', fontsize=12)
    ax.legend(facecolor=SURFACE, edgecolor='#1e3a5f', labelcolor=TEXT_COL)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'retention_vs_attrition.png'), dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"  Saved -> {RESULTS_DIR}/retention_vs_attrition.png")

    # 8. Model Metrics Bar Chart
    metric_names  = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    metric_values = [accuracy, precision, recall, f1, roc_auc]
    colors_met    = [COLOR_A, COLOR_P, COLOR_W, COLOR_D, '#a855f7']
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metric_names, metric_values, color=colors_met, width=0.5, edgecolor='none')
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f'{val:.3f}', ha='center', fontweight='bold', color=TEXT_COL, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title('Employee Attrition Prediction – Model Performance Metrics', fontsize=14, color=TEXT_COL, pad=15)
    ax.set_ylabel('Score (0–1)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_metrics.png'), dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"  Saved -> {RESULTS_DIR}/model_metrics.png")

    # ── Save model assets ─────────────────────────────────────────────────────
    print("[INFO] Saving model artifacts...")
    joblib.dump(model, MODEL_FILE)
    metadata = {
        "feature_names":   feature_names,
        "scaler":          scaler,
        "label_encoders":  le_map,
        "cat_cols":        cat_cols,
        "accuracy":        accuracy,
        "precision":       precision,
        "recall":          recall,
        "f1":              f1,
        "roc_auc":         roc_auc,
    }
    joblib.dump(metadata, METADATA_FILE)
    print(f"  Saved model    -> {MODEL_FILE}")
    print(f"  Saved metadata -> {METADATA_FILE}")
    print("[DONE] Employee Attrition model training complete!\n")

# ──────────────────────────────────────────────────────────────────────────────
# 2. FastAPI Web Server
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Employee Attrition Prediction API",
    description="AI-powered employee attrition prediction using Random Forest Classifier",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model assets
_model    = None
_metadata = None


def load_attrition_assets():
    global _model, _metadata
    try:
        if os.path.exists(MODEL_FILE) and os.path.exists(METADATA_FILE):
            _model    = joblib.load(MODEL_FILE)
            _metadata = joblib.load(METADATA_FILE)
            print("[INFO] Employee attrition model assets loaded successfully.")
        else:
            print("[WARNING] Model assets not found. Run with --train first.")
    except Exception as e:
        print(f"[ERROR] Failed to load model assets: {e}")


load_attrition_assets()


class AttritionPredictionRequest(BaseModel):
    age:               float = Field(35.0,   description="Employee age in years", ge=18, le=80)
    length_of_service: float = Field(5.0,    description="Years at company", ge=0)
    gender_short:      str   = Field("M",    description="Gender (M/F)")
    city_name:         str   = Field("Vancouver", description="City of work location")
    department_name:   str   = Field("Sales", description="Department name")
    job_title:         str   = Field("Sales Associate", description="Job title / role")
    store_name:        int   = Field(1,      description="Store number identifier", ge=0)
    business_unit:     str   = Field("STORES", description="Business unit (STORES/HEADOFFICE)")


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serve the Employee Attrition Prediction frontend UI."""
    if os.path.exists(FRONTEND_FILE):
        with open(FRONTEND_FILE, "r", encoding="utf-8") as fh:
            return HTMLResponse(content=fh.read())
    return HTMLResponse(content="<h3>Frontend file not found.</h3>", status_code=404)


@app.get("/health")
def health():
    """API health check endpoint."""
    return {
        "status":        "ok" if _model is not None else "no_model_loaded",
        "model":         "RandomForestClassifier",
        "domain":        "Employee Attrition Prediction",
        "assets_exist":  {
            "model":    os.path.exists(MODEL_FILE),
            "metadata": os.path.exists(METADATA_FILE),
        },
        "metrics": {
            "accuracy": round(_metadata["accuracy"], 4) if _metadata else None,
            "roc_auc":  round(_metadata["roc_auc"],  4) if _metadata else None,
        } if _metadata else {},
    }


@app.post("/predict")
def predict_attrition(req: AttritionPredictionRequest):
    """Predict employee attrition status and retention probability."""
    if _model is None or _metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run with --train first.")

    try:
        feature_names = _metadata["feature_names"]
        scaler        = _metadata["scaler"]
        le_map        = _metadata["label_encoders"]
        cat_cols      = _metadata["cat_cols"]

        # Build raw feature dict matching training columns
        raw = {
            "age":               req.age,
            "length_of_service": req.length_of_service,
            "gender_short":      req.gender_short,
            "city_name":         req.city_name,
            "department_name":   req.department_name,
            "job_title":         req.job_title,
            "store_name":        req.store_name,
            "BUSINESS_UNIT":     req.business_unit,
        }

        # Encode categorical features using saved LabelEncoders
        for col in cat_cols:
            if col in raw:
                le  = le_map[col]
                val = str(raw[col])
                if val in le.classes_:
                    raw[col] = int(le.transform([val])[0])
                else:
                    # Unseen label → use middle class index
                    raw[col] = int(len(le.classes_) // 2)

        # Build feature row in exact training order
        row      = [raw.get(f, 0) for f in feature_names]
        X_input  = np.array(row, dtype=float).reshape(1, -1)
        X_scaled = scaler.transform(X_input)

        pred_label = int(_model.predict(X_scaled)[0])
        pred_proba = _model.predict_proba(X_scaled)[0]

        retention_prob  = round(float(pred_proba[0]) * 100, 2)  # probability of ACTIVE
        attrition_prob  = round(float(pred_proba[1]) * 100, 2)  # probability of TERMINATED
        predicted_status = "TERMINATED" if pred_label == 1 else "ACTIVE"

        # Tree-level confidence interval on retention probability
        tree_probs   = np.array([t.predict_proba(X_scaled)[0][0] for t in _model.estimators_])
        ci_lower     = round(float(max(0, tree_probs.mean() - tree_probs.std())) * 100, 2)
        ci_upper     = round(float(min(1, tree_probs.mean() + tree_probs.std())) * 100, 2)

        return {
            "predicted_status":       predicted_status,
            "retention_probability":  retention_prob,
            "attrition_probability":  attrition_prob,
            "confidence_interval":    {"lower": ci_lower, "upper": ci_upper},
            "risk_level":             "High" if attrition_prob > 60 else ("Medium" if attrition_prob > 30 else "Low"),
            "model_accuracy":         round(_metadata["accuracy"], 4),
            "model_roc_auc":          round(_metadata["roc_auc"], 4),
            "employee_details": {
                "age":               req.age,
                "department":        req.department_name,
                "job_title":         req.job_title,
                "length_of_service": req.length_of_service,
                "business_unit":     req.business_unit,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/results/metrics")
def get_metrics():
    """Return model evaluation report and available plot paths."""
    metrics_file = os.path.join(RESULTS_DIR, "accuracy_results.txt")
    if not os.path.exists(metrics_file):
        raise HTTPException(status_code=404, detail="Metrics not found. Train the model first.")
    with open(metrics_file, "r", encoding="utf-8") as fh:
        content = fh.read()
    return {
        "text_report": content,
        "plots": {
            "attrition_distribution":  "/results/image/attrition_distribution.png",
            "correlation_heatmap":     "/results/image/correlation_heatmap.png",
            "roc_curve":               "/results/image/roc_curve.png",
            "confusion_matrix":        "/results/image/confusion_matrix.png",
            "feature_importance":      "/results/image/feature_importance.png",
            "precision_recall_curve":  "/results/image/precision_recall_curve.png",
            "retention_vs_attrition":  "/results/image/retention_vs_attrition.png",
            "model_metrics":           "/results/image/model_metrics.png",
        },
    }


@app.get("/results/image/{image_name}")
def serve_plot(image_name: str):
    """Serve a generated employee attrition visualization."""
    path = os.path.join(RESULTS_DIR, image_name)
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail=f"Image '{image_name}' not found.")


def _background_retrain():
    print("[RETRAIN] Starting background employee attrition model retraining...")
    try:
        train_attrition_model()
        load_attrition_assets()
        print("[RETRAIN] Employee attrition model retrained successfully.")
    except Exception as e:
        print(f"[RETRAIN] Error during retraining: {e}")


@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    """Retrain the employee attrition model in the background."""
    background_tasks.add_task(_background_retrain)
    return {"status": "accepted", "message": "Employee attrition model retraining initiated in the background."}


# ──────────────────────────────────────────────────────────────────────────────
# 3. Main Execution Block
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Employee Attrition Prediction – unified ML pipeline and FastAPI backend"
    )
    parser.add_argument("--train", action="store_true", help="Train the model and regenerate all results")
    parser.add_argument("--port",  type=int, default=8000, help="FastAPI server port (default: 8000)")
    args = parser.parse_args()

    if args.train or not os.path.exists(MODEL_FILE) or not os.path.exists(METADATA_FILE):
        try:
            train_attrition_model()
            load_attrition_assets()
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            sys.exit(1)

    if "--train" not in sys.argv or args.port:
        import uvicorn
        print(f"[INFO] Starting Employee Attrition Prediction API server on port {args.port}...")
        print(f"[INFO] Open your browser at: http://127.0.0.1:{args.port}")
        uvicorn.run(
            "employee_attrition_prediction:app",
            host="127.0.0.1",
            port=args.port,
            reload=False,
        )
