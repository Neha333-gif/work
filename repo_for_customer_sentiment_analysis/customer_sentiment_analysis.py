"""Customer Sentiment Analysis - unified ML pipeline and FastAPI backend."""

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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

DATA_PATH = r"C:\Users\peepl\Downloads\cust_segmentation_zip\repo_for_demand_forecasting\repo_for_demand_forecasting\customet_feedback_system\sentiment-analysis.csv"
RESULTS_DIR = "results"
MODEL_FILE = "customer_sentiment_analysis_model.joblib"
METADATA_FILE = "customer_sentiment_analysis_metadata.joblib"
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, "customer_sentiment_analysis_outputs.csv")
SAMPLE_SIZE = 40000

os.makedirs(RESULTS_DIR, exist_ok=True)


def _clear_previous_outputs():
    for name in os.listdir(RESULTS_DIR):
        if name.endswith((".png", ".txt", ".csv", ".json")):
            try:
                os.remove(os.path.join(RESULTS_DIR, name))
            except OSError:
                pass


def _read_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, engine="python", on_bad_lines="skip")
    if "Text" not in df.columns or "Sentiment" not in df.columns:
        raw = pd.read_csv(DATA_PATH, header=None, engine="python", on_bad_lines="skip")
        raw = raw[0].astype(str).str.split(",", expand=True)
        raw.columns = ["Text", "Sentiment", "Source", "Date/Time", "User ID", "Location", "Confidence Score"]
        df = raw.iloc[1:].reset_index(drop=True)
    return df


def _prepare_dataset(raw_df: pd.DataFrame):
    df = raw_df.copy()
    df["Text"] = df["Text"].astype(str).fillna("")
    df["Sentiment"] = df["Sentiment"].astype(str).str.strip().fillna("neutral")
    if "Confidence Score" in df.columns:
        df["Confidence Score"] = pd.to_numeric(df["Confidence Score"], errors="coerce").fillna(df["Confidence Score"].astype(str).str.len())
    else:
        df["Confidence Score"] = df["Text"].str.len()
    df.dropna(subset=["Text", "Sentiment"], inplace=True)
    df.drop_duplicates(subset=["Text", "Sentiment"], inplace=True)

    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Sentiment"])
    vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1, 2))
    x_text = vectorizer.fit_transform(df["Text"])

    return df, x_text, y, label_encoder, vectorizer


def train_customer_sentiment_analysis_model():
    print("[INFO] Starting Customer Sentiment Analysis model training...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    _clear_previous_outputs()
    raw_df = _read_dataset()
    df, x_text, y, label_encoder, vectorizer = _prepare_dataset(raw_df)

    x_train, x_test, y_train, y_test = train_test_split(
        x_text, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1200, random_state=42),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=12, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=180, max_depth=14, random_state=42, n_jobs=-1
        ),
    }

    model_metrics = {}
    trained = {}
    preds = {}
    probs = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(x_test).max(axis=1)
        else:
            prob = np.ones_like(pred, dtype=float) * 0.5
        model_metrics[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, pred, average="weighted", zero_division=0),
        }
        trained[name] = model
        preds[name] = pred
        probs[name] = prob

    best_model_name = max(model_metrics.keys(), key=lambda k: model_metrics[k]["f1"])
    model = trained[best_model_name]
    y_pred = preds[best_model_name]
    y_prob = probs[best_model_name]
    cm = confusion_matrix(y_test, y_pred)

    pred_df = pd.DataFrame({
        "text": df.iloc[x_test.indices[:len(y_pred)] % len(df)]["Text"].values if hasattr(x_test, "indices") else ["sample"] * len(y_pred),
        "actual_sentiment": label_encoder.inverse_transform(y_test),
        "predicted_sentiment": label_encoder.inverse_transform(y_pred),
        "confidence_score": y_prob,
    })
    pred_df.to_csv(PREDICTIONS_FILE, index=False)

    acc = model_metrics[best_model_name]["accuracy"]
    pre = model_metrics[best_model_name]["precision"]
    rec = model_metrics[best_model_name]["recall"]
    f1 = model_metrics[best_model_name]["f1"]
    text = f"""Customer Sentiment Analysis - Model Evaluation Report
=================================================
Selected Model           : {best_model_name}
Accuracy                 : {acc:.4f}
Precision                : {pre:.4f}
Recall                   : {rec:.4f}
F1 Score                 : {f1:.4f}
Dataset Path             : {DATA_PATH}
Total Samples            : {len(df)}
Training Samples         : {x_train.shape[0]}
Testing Samples          : {x_test.shape[0]}

Model Comparison
----------------
{os.linesep.join([f"{k}: acc={v['accuracy']:.4f}, pre={v['precision']:.4f}, rec={v['recall']:.4f}, f1={v['f1']:.4f}" for k, v in model_metrics.items()])}
"""
    with open(os.path.join(RESULTS_DIR, "accuracy_results.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(RESULTS_DIR, "evaluation_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\nPrecision: {pre:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\n")

    sns.set_theme(style="darkgrid")

    fig, ax = plt.subplots(figsize=(7, 5))
    df["Sentiment"].value_counts().plot(kind="bar", color="#2563eb", ax=ax)
    ax.set_title("Sentiment Distribution Plot")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "sentiment_distribution.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(9, 6))
    top_terms = np.asarray(x_train.sum(axis=0)).ravel()
    term_df = pd.DataFrame({"term": vectorizer.get_feature_names_out(), "weight": top_terms}).sort_values("weight", ascending=False).head(20)
    sns.barplot(data=term_df, x="weight", y="term", ax=ax, color="#14b8a6")
    ax.set_title("Top Sentiment Terms Analysis")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "top_sentiment_terms.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150); plt.close()

    # Correlation-like view using metrics per model.
    metrics_matrix = pd.DataFrame(model_metrics).T[["accuracy", "precision", "recall", "f1"]]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(metrics_matrix, annot=True, cmap="coolwarm", fmt=".3f", ax=ax)
    ax.set_title("Model Metrics Correlation Heatmap")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "correlation_heatmap.png"), dpi=150); plt.close()

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=vectorizer.get_feature_names_out())
    else:
        coef = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        importances = pd.Series(coef, index=vectorizer.get_feature_names_out())
    importances = importances.sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.sort_values().plot(kind="barh", color="#f59e0b", ax=ax)
    ax.set_title("Feature Importance Plot")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    conf_bins = pd.cut(pred_df["confidence_score"], bins=[0, 0.4, 0.7, 1.0], labels=["Low", "Medium", "High"], include_lowest=True)
    conf_bins.value_counts().reindex(["Low", "Medium", "High"]).plot(kind="bar", color=["#22c55e", "#f59e0b", "#ef4444"], ax=ax)
    ax.set_title("Customer Risk Category Comparison Chart")
    ax.set_xlabel("Risk Category")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "customer_risk_category_comparison.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    lengths = df["Text"].str.len()
    sns.kdeplot(lengths[df["Sentiment"] == df["Sentiment"].unique()[0]], label=str(df["Sentiment"].unique()[0]), ax=ax)
    for s in df["Sentiment"].unique()[1:3]:
        sns.kdeplot(lengths[df["Sentiment"] == s], label=str(s), ax=ax)
    ax.set_title("Feedback Length Trend by Sentiment")
    ax.set_xlabel("Text Length")
    ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "feedback_length_trend.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(model_metrics.keys())
    accs = [model_metrics[n]["accuracy"] for n in names]
    f1s = [model_metrics[n]["f1"] for n in names]
    x_idx = np.arange(len(names))
    ax.bar(x_idx - 0.2, accs, width=0.4, label="Accuracy", color="#22c55e")
    ax.bar(x_idx + 0.2, f1s, width=0.4, label="F1", color="#6366f1")
    ax.set_xticks(x_idx); ax.set_xticklabels(names, rotation=10, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance Metrics Visualization")
    ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "model_performance_metrics_visualization.png"), dpi=150); plt.close()

    pd.DataFrame([{"model": k, **v} for k, v in model_metrics.items()]).to_csv(
        os.path.join(RESULTS_DIR, "model_metrics_summary.csv"), index=False
    )

    joblib.dump(model, MODEL_FILE)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "label_encoder": label_encoder,
            "best_model_name": best_model_name,
            "accuracy": acc,
        },
        METADATA_FILE,
    )
    print("[DONE] Customer Sentiment Analysis training complete.")


app = FastAPI(title="Customer Sentiment Analysis API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

_model = None
_metadata = None


def load_assets():
    global _model, _metadata
    if os.path.exists(MODEL_FILE) and os.path.exists(METADATA_FILE):
        _model = joblib.load(MODEL_FILE)
        _metadata = joblib.load(METADATA_FILE)


load_assets()


class SentimentRequest(BaseModel):
    feedback_text: str = Field(..., min_length=3)
    customer_segment: str = Field(..., min_length=2)
    source_channel: str = Field(..., min_length=2)
    confidence_score: float = Field(..., ge=0, le=1)


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    path = "customer_sentiment_analysis_frontend.html"
    if os.path.exists(path):
        return HTMLResponse(open(path, "r", encoding="utf-8").read())
    return HTMLResponse("<h3>Frontend file not found.</h3>", status_code=404)


@app.post("/predict")
def predict_sentiment(req: SentimentRequest):
    if _model is None or _metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run with --train first.")
    enriched_text = f"{req.feedback_text} {req.customer_segment} {req.source_channel} confidence_{req.confidence_score:.2f}"
    x_input = _metadata["vectorizer"].transform([enriched_text])
    pred = _model.predict(x_input)[0]
    if hasattr(_model, "predict_proba"):
        prob = float(_model.predict_proba(x_input).max())
    else:
        prob = 0.5
    label = _metadata["label_encoder"].inverse_transform([pred])[0]
    category = "High" if prob >= 0.7 else ("Medium" if prob >= 0.4 else "Low")
    return {
        "sentiment_prediction_result": str(label),
        "sentiment_probability_score": round(prob * 100, 2),
        "confidence_category": category,
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
    train_customer_sentiment_analysis_model()
    load_assets()


@app.post("/retrain")
def retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(_background_retrain)
    return {"status": "accepted", "message": "Customer sentiment analysis retraining started."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Sentiment Analysis unified ML pipeline and backend")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--port", type=int, default=8040)
    args = parser.parse_args()
    if args.train or not os.path.exists(MODEL_FILE) or not os.path.exists(METADATA_FILE):
        try:
            train_customer_sentiment_analysis_model()
            load_assets()
        except Exception as exc:
            print(f"[ERROR] Training failed: {exc}")
            sys.exit(1)
    if "--train" not in sys.argv:
        import uvicorn
        uvicorn.run("customer_sentiment_analysis:app", host="127.0.0.1", port=args.port, reload=False)
