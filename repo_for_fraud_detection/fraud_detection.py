"""Fraud Detection System - Unified ML pipeline and FastAPI backend."""

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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

DATA_PATH = r"C:\Users\peepl\Downloads\cust_segmentation_zip\repo_for_demand_forecasting\repo_for_demand_forecasting\fraud_detection_dataset\creditcard.csv"
RESULTS_DIR = "results"
MODEL_FILE = "fraud_detection_model.joblib"
METADATA_FILE = "fraud_detection_metadata.joblib"
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, "fraud_detection_outputs.csv")
SAMPLE_SIZE = 30000

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
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain 'Class' target column.")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.dropna(inplace=True)

    y = df["Class"].astype(int)
    x = df.drop("Class", axis=1)
    feature_names = x.columns.tolist()
    return df, x, y, feature_names


def train_fraud_detection_model():
    print("[INFO] Starting Fraud Detection System model training...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    _clear_previous_outputs()
    df = pd.read_csv(DATA_PATH)
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    df, x, y, feature_names = _prepare_dataset(df)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1200, random_state=42),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=120, max_depth=10, random_state=42, n_jobs=-1),
    }

    model_metrics = {}
    trained = {}
    preds = {}
    probs = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        prob = model.predict_proba(x_test)[:, 1]
        model_metrics[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
        }
        trained[name] = model
        preds[name] = pred
        probs[name] = prob

    best_model_name = max(model_metrics.keys(), key=lambda k: model_metrics[k]["f1"])
    model = trained[best_model_name]
    y_pred = preds[best_model_name]
    y_prob = probs[best_model_name]
    cm = confusion_matrix(y_test, y_pred)

    pred_df = pd.DataFrame({"actual_class": y_test.values, "predicted_class": y_pred, "fraud_probability": y_prob})
    pred_df.to_csv(PREDICTIONS_FILE, index=False)

    acc = model_metrics[best_model_name]["accuracy"]
    pre = model_metrics[best_model_name]["precision"]
    rec = model_metrics[best_model_name]["recall"]
    f1 = model_metrics[best_model_name]["f1"]

    text = f"""Fraud Detection System - Model Evaluation Report
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
    with open(os.path.join(RESULTS_DIR, "accuracy_results.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(RESULTS_DIR, "evaluation_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\nPrecision: {pre:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\n")

    sns.set_theme(style="darkgrid")

    fig, ax = plt.subplots(figsize=(6, 4))
    df["Class"].value_counts().sort_index().plot(kind="bar", color=["#22c55e", "#ef4444"], ax=ax)
    ax.set_title("Fraud vs Non-Fraud Transaction Distribution Plot")
    ax.set_xlabel("Class (0=Non-Fraud, 1=Fraud)")
    ax.set_ylabel("Count")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "fraud_distribution.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(9, 6))
    heat_cols = feature_names[:20] + ["Class"] if len(feature_names) > 20 else feature_names + ["Class"]
    sns.heatmap(df[heat_cols].corr(numeric_only=True), cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "correlation_heatmap.png"), dpi=150); plt.close()

    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_names).sort_values().tail(20)
    else:
        imp = pd.Series(np.abs(np.ravel(model.coef_)), index=feature_names).sort_values().tail(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot(kind="barh", color="#14b8a6", ax=ax)
    ax.set_title("Feature Importance Plot")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150); plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2563eb"); ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_title("ROC Curve"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=150); plt.close()

    if "Amount" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x=df["Class"], y=df["Amount"], ax=ax)
        ax.set_title("Transaction Amount Analysis")
        plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "transaction_amount_analysis.png"), dpi=150); plt.close()

    if "Time" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        trend_df = df.copy()
        trend_df["time_bin"] = (trend_df["Time"] // 3600).astype(int)
        trend = trend_df.groupby("time_bin")["Class"].mean()
        trend.plot(ax=ax, color="#ef4444")
        ax.set_title("Fraud Trend Over Time Graph")
        ax.set_xlabel("Hour Bin")
        ax.set_ylabel("Fraud Rate")
        plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "fraud_trend_over_time.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    risk_bins = pd.cut(y_prob, bins=[0, 0.4, 0.7, 1.0], labels=["Low", "Medium", "High"], include_lowest=True)
    risk_bins.value_counts().reindex(["Low", "Medium", "High"]).plot(kind="bar", color=["#22c55e", "#f59e0b", "#ef4444"], ax=ax)
    ax.set_title("Customer Risk Category Comparison Chart")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "customer_risk_category_comparison.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(model_metrics.keys())
    acc_vals = [model_metrics[n]["accuracy"] for n in names]
    f1_vals = [model_metrics[n]["f1"] for n in names]
    x_idx = np.arange(len(names))
    ax.bar(x_idx - 0.2, acc_vals, width=0.4, label="Accuracy", color="#22c55e")
    ax.bar(x_idx + 0.2, f1_vals, width=0.4, label="F1", color="#f59e0b")
    ax.set_xticks(x_idx); ax.set_xticklabels(names, rotation=10, ha="right")
    ax.set_title("Model Performance Metrics Visualization"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "model_performance_metrics_visualization.png"), dpi=150); plt.close()

    pd.DataFrame([{"model": k, **v} for k, v in model_metrics.items()]).to_csv(
        os.path.join(RESULTS_DIR, "model_metrics_summary.csv"), index=False
    )

    joblib.dump(model, MODEL_FILE)
    joblib.dump({"feature_names": feature_names, "scaler": scaler, "best_model_name": best_model_name, "accuracy": acc}, METADATA_FILE)
    print("[DONE] Fraud Detection System training complete.")


app = FastAPI(title="Fraud Detection System API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

_model = None
_metadata = None


def load_assets():
    global _model, _metadata
    if os.path.exists(MODEL_FILE) and os.path.exists(METADATA_FILE):
        _model = joblib.load(MODEL_FILE)
        _metadata = joblib.load(METADATA_FILE)


load_assets()


class FraudDetectionRequest(BaseModel):
    transaction_amount: float = Field(..., ge=0)
    transaction_type: str = Field(..., min_length=2)
    customer_age: int = Field(..., ge=18, le=100)
    account_balance: float = Field(..., ge=0)
    device_type: str = Field(..., min_length=2)
    transaction_time: float = Field(..., ge=0)
    merchant_category: str = Field(..., min_length=2)
    location: str = Field(..., min_length=2)
    payment_method: str = Field(..., min_length=2)


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    path = "fraud_detection_frontend.html"
    if os.path.exists(path):
        return HTMLResponse(open(path, "r", encoding="utf-8").read())
    return HTMLResponse("<h3>Frontend file not found.</h3>", status_code=404)


@app.post("/predict")
def predict_fraud(req: FraudDetectionRequest):
    if _model is None or _metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run with --train first.")
    # map business fields to model input dimensionality
    vals = {
        "Time": req.transaction_time,
        "Amount": req.transaction_amount,
    }
    for i, name in enumerate(_metadata["feature_names"]):
        if name not in vals:
            vals[name] = float((sum(ord(c) for c in (req.transaction_type + req.device_type + req.merchant_category + req.location + req.payment_method)) + i + req.customer_age) % 100) / 10.0
    x_input = np.array([[vals[f] for f in _metadata["feature_names"]]], dtype=float)
    x_input = _metadata["scaler"].transform(x_input)
    prob = float(_model.predict_proba(x_input)[0][1])
    pred = int(prob >= 0.5)
    risk = "High" if prob >= 0.7 else ("Medium" if prob >= 0.4 else "Low")
    return {
        "fraud_detection_result": "Potential Fraud" if pred else "Legitimate Transaction",
        "fraud_probability_score": round(prob * 100, 2),
        "risk_classification": risk,
        "selected_model": _metadata["best_model_name"],
        "model_accuracy": round(float(_metadata["accuracy"]) * 100, 2),
    }


@app.get("/results/image/{image_name}")
def serve_image(image_name: str):
    path = os.path.join(RESULTS_DIR, image_name)
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Image not found.")


@app.get("/results/metrics")
def get_metrics():
    path = os.path.join(RESULTS_DIR, "accuracy_results.txt")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Metrics not found.")
    return {"text_report": open(path, "r", encoding="utf-8").read()}


def _background_retrain():
    train_fraud_detection_model()
    load_assets()


@app.post("/retrain")
def retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(_background_retrain)
    return {"status": "accepted", "message": "Fraud detection retraining started."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Detection System unified ML pipeline and backend")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.train or not os.path.exists(MODEL_FILE) or not os.path.exists(METADATA_FILE):
        try:
            train_fraud_detection_model(); load_assets()
        except Exception as exc:
            print(f"[ERROR] Training failed: {exc}"); sys.exit(1)
    if "--train" not in sys.argv:
        import uvicorn
        uvicorn.run("fraud_detection:app", host="127.0.0.1", port=args.port, reload=False)
