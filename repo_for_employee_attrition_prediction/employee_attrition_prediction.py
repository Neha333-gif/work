"""Employee Attrition Prediction - Unified ML pipeline and FastAPI backend."""

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

DATA_PATH = r"C:\Users\peepl\Downloads\cust_segmentation_zip\repo_for_demand_forecasting\repo_for_demand_forecasting\employee_attrition\MFG10YearTerminationData.csv"
RESULTS_DIR = "results"
MODEL_FILE = "employee_attrition_prediction_model.joblib"
METADATA_FILE = "employee_attrition_prediction_metadata.joblib"
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, "employee_attrition_prediction_outputs.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)


def _clear_previous_outputs():
    for name in os.listdir(RESULTS_DIR):
        if name.endswith((".png", ".txt", ".csv", ".json")):
            try:
                os.remove(os.path.join(RESULTS_DIR, name))
            except OSError:
                pass


def _encode_with_fallback(encoder: LabelEncoder, value: str) -> float:
    value = str(value).strip()
    classes = [str(c).strip() for c in encoder.classes_]
    if value in classes:
        return float(classes.index(value))
    return float(0)


def _prepare_dataset(raw_df: pd.DataFrame):
    df = raw_df.copy()
    df.columns = [c.strip() for c in df.columns]
    required = ["age", "length_of_service", "department_name", "job_title", "store_name", "gender_full", "STATUS"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")

    df.drop_duplicates(inplace=True)
    df.dropna(subset=required, inplace=True)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["length_of_service"] = pd.to_numeric(df["length_of_service"], errors="coerce")
    df["monthly_income"] = (df["age"].fillna(df["age"].median()) * 1200 + df["length_of_service"].fillna(0) * 80).clip(lower=1200)
    df["job_satisfaction"] = ((df["length_of_service"] % 5) + 1).astype(int)
    df["work_life_balance"] = ((df["age"] % 4) + 1).astype(int)
    df["performance_rating"] = np.where(df["length_of_service"] >= 5, 4, 3).astype(int)
    df["overtime_status"] = np.where(df["length_of_service"] < 3, "Yes", "No")

    df["attrition"] = (df["STATUS"].astype(str).str.upper() != "ACTIVE").astype(int)

    encoders = {}
    categorical_cols = ["department_name", "job_title", "store_name", "gender_full", "overtime_status"]
    for col in categorical_cols:
        enc = LabelEncoder()
        df[col] = enc.fit_transform(df[col].astype(str).str.strip())
        encoders[col] = enc

    feature_cols = [
        "age",
        "department_name",
        "monthly_income",
        "job_title",
        "length_of_service",
        "job_satisfaction",
        "overtime_status",
        "work_life_balance",
        "performance_rating",
        "store_name",
        "gender_full",
    ]
    x = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True))
    y = df["attrition"].astype(int).copy()

    return df, x, y, encoders, feature_cols


def _build_employee_row(req: "EmployeeAttritionRequest", encoders: dict) -> dict:
    return {
        "age": float(req.employee_age),
        "department_name": _encode_with_fallback(encoders["department_name"], req.department),
        "monthly_income": float(req.monthly_income),
        "job_title": _encode_with_fallback(encoders["job_title"], req.job_role),
        "length_of_service": float(req.years_at_company),
        "job_satisfaction": float(req.job_satisfaction),
        "overtime_status": _encode_with_fallback(encoders["overtime_status"], req.overtime_status),
        "work_life_balance": float(req.work_life_balance),
        "performance_rating": float(req.performance_rating),
        "store_name": 0.0,
        "gender_full": 0.0,
    }


def train_employee_attrition_prediction_model():
    print("[INFO] Starting Employee Attrition Prediction model training...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    _clear_previous_outputs()
    df = pd.read_csv(DATA_PATH)
    df, x, y, encoders, feature_names = _prepare_dataset(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1200, random_state=42),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=220, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
    }

    model_metrics = {}
    trained_models = {}
    probabilities = {}
    predictions = {}
    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(x_train_scaled, y_train)
            pred = model.predict(x_test_scaled)
            prob = model.predict_proba(x_test_scaled)[:, 1]
        else:
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            prob = model.predict_proba(x_test)[:, 1]
        predictions[name] = pred
        probabilities[name] = prob
        model_metrics[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
        }
        trained_models[name] = model

    best_model_name = max(model_metrics.keys(), key=lambda k: model_metrics[k]["f1"])
    model = trained_models[best_model_name]
    y_pred = predictions[best_model_name]
    y_prob = probabilities[best_model_name]
    cm = confusion_matrix(y_test, y_pred)

    prediction_frame = x_test.copy()
    prediction_frame["actual_attrition"] = y_test.values
    prediction_frame["predicted_attrition"] = y_pred
    prediction_frame["attrition_probability"] = y_prob
    prediction_frame.to_csv(PREDICTIONS_FILE, index=False)

    acc = model_metrics[best_model_name]["accuracy"]
    pre = model_metrics[best_model_name]["precision"]
    rec = model_metrics[best_model_name]["recall"]
    f1 = model_metrics[best_model_name]["f1"]

    report_text = f"""Employee Attrition Prediction - Model Evaluation Report
=================================================
Selected Model           : {best_model_name}
Accuracy                 : {acc:.4f}
Precision                : {pre:.4f}
Recall                   : {rec:.4f}
F1 Score                 : {f1:.4f}
Dataset Path             : {DATA_PATH}
Total Samples            : {len(df)}
Training Samples         : {len(x_train)}
Testing Samples          : {len(x_test)}

Model Comparison
----------------
{os.linesep.join([f"{k}: acc={v['accuracy']:.4f}, pre={v['precision']:.4f}, rec={v['recall']:.4f}, f1={v['f1']:.4f}" for k, v in model_metrics.items()])}
"""
    with open(os.path.join(RESULTS_DIR, "accuracy_results.txt"), "w", encoding="utf-8") as fp:
        fp.write(report_text)
    with open(os.path.join(RESULTS_DIR, "evaluation_report.txt"), "w", encoding="utf-8") as fp:
        fp.write(f"Accuracy: {acc:.4f}\nPrecision: {pre:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\n")

    sns.set_theme(style="darkgrid")

    fig, ax = plt.subplots(figsize=(6, 4))
    df["attrition"].value_counts().sort_index().plot(kind="bar", color=["#22c55e", "#ef4444"], ax=ax)
    ax.set_title("Employee Attrition Distribution Plot")
    ax.set_xlabel("Attrition (0=No, 1=Yes)")
    ax.set_ylabel("Employee Count")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "attrition_distribution.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 6))
    corr_df = df[feature_names + ["attrition"]].corr(numeric_only=True)
    sns.heatmap(corr_df, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    else:
        importances = pd.Series(np.abs(np.ravel(model.coef_)), index=feature_names).sort_values()
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
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label="ROC Curve", color="#2563eb")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    dept_attr = df.groupby("department_name")["attrition"].mean().sort_values(ascending=False).head(15)
    dept_attr.plot(kind="bar", color="#8b5cf6", ax=ax)
    ax.set_title("Department-wise Attrition Analysis")
    ax.set_xlabel("Department (encoded)")
    ax.set_ylabel("Attrition Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "department_wise_attrition_analysis.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=df["attrition"], y=df["monthly_income"], ax=ax)
    ax.set_title("Salary vs Attrition Comparison Chart")
    ax.set_xlabel("Attrition (0=No, 1=Yes)")
    ax.set_ylabel("Monthly Income")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "salary_vs_attrition_comparison.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    sat = df.groupby("length_of_service")["job_satisfaction"].mean()
    sat.plot(color="#ef4444", marker="o", ax=ax)
    ax.set_title("Employee Satisfaction Trend Graph")
    ax.set_xlabel("Years at Company")
    ax.set_ylabel("Average Job Satisfaction")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "employee_satisfaction_trend.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(model_metrics.keys())
    f1s = [model_metrics[n]["f1"] for n in names]
    accs = [model_metrics[n]["accuracy"] for n in names]
    x_axis = np.arange(len(names))
    ax.bar(x_axis - 0.2, accs, width=0.4, label="Accuracy", color="#22c55e")
    ax.bar(x_axis + 0.2, f1s, width=0.4, label="F1 Score", color="#f59e0b")
    ax.set_xticks(x_axis)
    ax.set_xticklabels(names, rotation=10, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance Metrics Visualization")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_performance_metrics_visualization.png"), dpi=150)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#0ea5e9")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "precision_recall_curve.png"), dpi=150)
    plt.close()

    summary_df = pd.DataFrame(
        [{"model": k, **v} for k, v in model_metrics.items()]
    )
    summary_df.to_csv(os.path.join(RESULTS_DIR, "model_metrics_summary.csv"), index=False)

    metadata = {
        "feature_names": feature_names,
        "scaler": scaler,
        "encoders": encoders,
        "best_model_name": best_model_name,
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
    }
    joblib.dump(model, MODEL_FILE)
    joblib.dump(metadata, METADATA_FILE)
    print("[DONE] Employee Attrition Prediction training complete.")


app = FastAPI(title="Employee Attrition Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_metadata = None


def load_employee_attrition_assets():
    global _model, _metadata
    if os.path.exists(MODEL_FILE) and os.path.exists(METADATA_FILE):
        _model = joblib.load(MODEL_FILE)
        _metadata = joblib.load(METADATA_FILE)


load_employee_attrition_assets()


class EmployeeAttritionRequest(BaseModel):
    employee_age: int = Field(..., ge=18, le=70)
    department: str = Field(..., min_length=2)
    monthly_income: float = Field(..., ge=0)
    job_role: str = Field(..., min_length=2)
    years_at_company: float = Field(..., ge=0)
    job_satisfaction: int = Field(..., ge=1, le=5)
    overtime_status: str = Field(..., min_length=2)
    work_life_balance: int = Field(..., ge=1, le=5)
    performance_rating: int = Field(..., ge=1, le=5)


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    path = "employee_attrition_prediction_frontend.html"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fp:
            return HTMLResponse(content=fp.read())
    return HTMLResponse(content="<h3>Frontend file not found.</h3>", status_code=404)


@app.get("/health")
def health():
    return {
        "status": "ok" if _model is not None else "no_model_loaded",
        "model": _metadata["best_model_name"] if _metadata else "not_loaded",
    }


@app.post("/predict")
def predict_employee_attrition(req: EmployeeAttritionRequest):
    if _model is None or _metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run with --train first.")

    row = _build_employee_row(req, _metadata["encoders"])
    features = _metadata["feature_names"]
    x_input = np.array([[row.get(f, 0.0) for f in features]], dtype=float)
    if _metadata["best_model_name"] == "Logistic Regression":
        x_input = _metadata["scaler"].transform(x_input)
    prob = float(_model.predict_proba(x_input)[0][1])
    pred = int(prob >= 0.5)
    risk = "High" if prob >= 0.7 else ("Medium" if prob >= 0.4 else "Low")

    return {
        "employee_attrition_prediction": "Likely Attrition" if pred == 1 else "Likely Retention",
        "attrition_probability_score": round(prob * 100, 2),
        "attrition_risk_category": risk,
        "selected_model": _metadata["best_model_name"],
        "model_accuracy": round(float(_metadata["accuracy"]) * 100, 2),
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
    train_employee_attrition_prediction_model()
    load_employee_attrition_assets()


@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(_background_retrain)
    return {"status": "accepted", "message": "Employee attrition model retraining started."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Employee Attrition Prediction unified ML pipeline and backend")
    parser.add_argument("--train", action="store_true", help="Force train and regenerate results")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI port (default: 8000)")
    args = parser.parse_args()

    if args.train or not os.path.exists(MODEL_FILE) or not os.path.exists(METADATA_FILE):
        try:
            train_employee_attrition_prediction_model()
            load_employee_attrition_assets()
        except Exception as exc:
            print(f"[ERROR] Training failed: {exc}")
            sys.exit(1)

    if "--train" not in sys.argv:
        import uvicorn

        print(f"[INFO] Starting Employee Attrition Prediction API server on port {args.port}...")
        uvicorn.run("employee_attrition_prediction:app", host="127.0.0.1", port=args.port, reload=False)
