# -*- coding: utf-8 -*-
"""
Product Recommendation System - Unified ML Pipeline and FastAPI Backend
Trains lightweight classification models on customer product interaction data,
generates visualization artifacts, saves model assets, and
serves real-time product recommendations via FastAPI.
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

DATA_PATH = r"C:\Users\peepl\Downloads\cust_segmentation_zip\repo_for_demand_forecasting\repo_for_demand_forecasting\product_recommendation_engine\ratings_Beauty.csv"
RESULTS_DIR = "results"
MODEL_FILE = "product_recommendation_system_model.joblib"
METADATA_FILE = "product_recommendation_system_metadata.joblib"
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, "product_recommendation_outputs.csv")
SAMPLE_SIZE = 30000

os.makedirs(RESULTS_DIR, exist_ok=True)


def _clear_previous_outputs():
    for name in os.listdir(RESULTS_DIR):
        if name.endswith((".png", ".txt", ".csv", ".json")):
            try:
                os.remove(os.path.join(RESULTS_DIR, name))
            except OSError:
                pass


def _encode_with_fallback(encoder: LabelEncoder, value: str):
    value = str(value).strip()
    classes = [str(c).strip() for c in encoder.classes_]
    if value in classes:
        return float(classes.index(value))
    return float(0)


def _prepare_dataset(raw_df: pd.DataFrame):
    df = raw_df.copy()
    df.columns = [c.strip() for c in df.columns]

    expected = ["UserId", "ProductId", "Rating"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s", errors="coerce")
        df["Review_Month"] = df["Timestamp"].dt.month.fillna(1).astype(int)
    else:
        df["Review_Month"] = 1

    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df.dropna(subset=["Rating"], inplace=True)
    df["Liked"] = (df["Rating"] >= 4).astype(int)

    user_stats = df.groupby("UserId").agg(
        User_Avg_Rating=("Rating", "mean"),
        User_Review_Count=("Rating", "count"),
        User_Liked_Rate=("Liked", "mean"),
    )
    product_stats = df.groupby("ProductId").agg(
        Product_Avg_Rating=("Rating", "mean"),
        Product_Popularity=("Rating", "count"),
    )

    df = df.merge(user_stats, on="UserId", how="left")
    df = df.merge(product_stats, on="ProductId", how="left")

    df["Product_Category"] = pd.qcut(
        df["Product_Popularity"].rank(method="first"),
        q=4,
        labels=["Beauty Essentials", "Popular Beauty", "Trending Beauty", "Top Rated Beauty"],
    )
    df["Preferred_Brand"] = df["ProductId"].astype(str).str[:3]
    df["Budget_Range"] = pd.cut(
        df["Product_Avg_Rating"],
        bins=[0, 2.5, 3.5, 4.5, 5.1],
        labels=["Low", "Medium", "High", "Premium"],
    )
    df["Shopping_Frequency"] = pd.cut(
        df["User_Review_Count"],
        bins=[0, 2, 5, 20, 10000],
        labels=["Rare", "Occasional", "Regular", "Frequent"],
    )
    df["Product_Interest_Type"] = np.where(df["Liked"] == 1, "High Interest", "Low Interest")

    encoders = {}
    for col in ["UserId", "ProductId", "Product_Category", "Preferred_Brand", "Budget_Range", "Shopping_Frequency"]:
        enc = LabelEncoder()
        df[col] = enc.fit_transform(df[col].astype(str).str.strip())
        encoders[col] = enc

    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    feature_cols = [
        "UserId",
        "ProductId",
        "User_Avg_Rating",
        "User_Review_Count",
        "User_Liked_Rate",
        "Product_Avg_Rating",
        "Product_Popularity",
        "Review_Month",
        "Product_Category",
        "Preferred_Brand",
        "Budget_Range",
        "Shopping_Frequency",
    ]
    X = df[feature_cols].copy()
    y = df["Liked"].astype(int).copy()

    top_products = (
        df.groupby("ProductId")["Liked"]
        .count()
        .sort_values(ascending=False)
        .head(30)
        .index.astype(str)
        .tolist()
    )

    return df, X, y, encoders, feature_cols, top_products


def _build_customer_row(req, encoders):
    rating = float(req.product_ratings)
    return {
        "UserId": _encode_with_fallback(encoders["UserId"], req.customer_id),
        "ProductId": _encode_with_fallback(encoders["ProductId"], req.customer_id[:10] if len(req.customer_id) >= 10 else req.customer_id),
        "User_Avg_Rating": rating,
        "User_Review_Count": float(req.previous_purchases),
        "User_Liked_Rate": min(1.0, rating / 5.0),
        "Product_Avg_Rating": rating,
        "Product_Popularity": float(req.previous_purchases) * 2,
        "Review_Month": 6.0,
        "Product_Category": _encode_with_fallback(encoders["Product_Category"], req.product_category),
        "Preferred_Brand": _encode_with_fallback(encoders["Preferred_Brand"], req.preferred_brand[:3]),
        "Budget_Range": _encode_with_fallback(encoders["Budget_Range"], req.budget_range),
        "Shopping_Frequency": _encode_with_fallback(encoders["Shopping_Frequency"], req.shopping_frequency),
    }


def train_product_recommendation_model():
    print("[INFO] Starting Product Recommendation System model training...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    _clear_previous_outputs()

    df = pd.read_csv(DATA_PATH, nrows=200000)
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    df, X, y, encoders, feature_names, top_products = _prepare_dataset(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=180, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1
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
    report = classification_report(y_test, y_pred, target_names=["Not Liked", "Liked"], zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    _, X_test_raw, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    prediction_frame = X_test_raw.copy()
    prediction_frame["actual_liked"] = y_test.values
    prediction_frame["predicted_liked"] = y_pred
    if y_prob is not None:
        prediction_frame["recommendation_confidence"] = y_prob
    prediction_frame.to_csv(PREDICTIONS_FILE, index=False)

    metrics_text = f"""Product Recommendation System - Model Evaluation Report
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
    df.groupby("ProductId")["Rating"].count().sort_values(ascending=False).head(20).plot(
        kind="bar", color="#2563eb", ax=ax
    )
    ax.set_title("Product Popularity Distribution")
    ax.set_xlabel("Product ID (Top 20)")
    ax.set_ylabel("Review Count")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "product_popularity.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        df[feature_names + ["Liked"]].corr(numeric_only=True),
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
    else:
        importances = pd.Series(np.ones(len(feature_names)), index=feature_names).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind="barh", color="#14b8a6", ax=ax)
    ax.set_title("Feature Importance Plot")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    sample_df = df.sample(min(500, len(df)), random_state=42)
    ax.scatter(sample_df["User_Review_Count"], sample_df["Product_Popularity"], c=sample_df["Liked"], cmap="viridis", alpha=0.5)
    ax.set_title("User-Product Interaction Graph")
    ax.set_xlabel("User Review Count")
    ax.set_ylabel("Product Popularity")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "user_product_interaction.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(model_metrics.keys())
    f1_vals = [model_metrics[n]["f1"] for n in names]
    ax.bar(names, f1_vals, color=["#6366f1", "#f59e0b", "#22c55e"])
    ax.set_title("Recommendation Accuracy Comparison")
    ax.set_ylabel("F1 Score")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "recommendation_accuracy_comparison.png"), dpi=150)
    plt.close()

    cat_labels = encoders["Product_Category"].classes_
    cat_map = {i: str(l) for i, l in enumerate(cat_labels)}
    fig, ax = plt.subplots(figsize=(8, 5))
    df.groupby(df["Product_Category"].map(cat_map))["Liked"].mean().plot(kind="bar", color="#8b5cf6", ax=ax)
    ax.set_title("Category-wise Recommendation Analysis")
    ax.set_xlabel("Product Category")
    ax.set_ylabel("Like Rate")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "category_wise_recommendation.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    df.groupby("Review_Month")["User_Liked_Rate"].mean().plot(marker="o", color="#ef4444", ax=ax)
    ax.set_title("Customer Preference Trends")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Like Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "customer_preference_trends.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["Rating"], bins=20, kde=True, color="#22c55e", ax=ax)
    ax.set_title("Product Rating Distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "product_rating_distribution.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
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
        "top_products": top_products,
        "category_labels": [str(c) for c in encoders["Product_Category"].classes_],
    }
    joblib.dump(model, MODEL_FILE)
    joblib.dump(metadata, METADATA_FILE)
    print("[DONE] Product Recommendation System training complete.")


app = FastAPI(title="Product Recommendation System API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_metadata = None


def load_product_recommendation_assets():
    global _model, _metadata
    if os.path.exists(MODEL_FILE) and os.path.exists(METADATA_FILE):
        _model = joblib.load(MODEL_FILE)
        _metadata = joblib.load(METADATA_FILE)


load_product_recommendation_assets()


class ProductRecommendationRequest(BaseModel):
    customer_id: str = Field(..., min_length=2)
    product_category: str = Field(..., min_length=2)
    previous_purchases: int = Field(..., ge=0)
    product_ratings: float = Field(..., ge=1, le=5)
    preferred_brand: str = Field(..., min_length=1)
    budget_range: str = Field(..., min_length=2)
    shopping_frequency: str = Field(..., min_length=2)
    product_interest_type: str = Field(..., min_length=2)


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    path = "product_recommendation_system_frontend.html"
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
def predict_product_recommendation(req: ProductRecommendationRequest):
    if _model is None or _metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run with --train first.")

    base_row = _build_customer_row(req, _metadata["encoders"])
    feature_names = _metadata["feature_names"]
    top_products = _metadata.get("top_products", [])[:5]

    scores = []
    for product_id in top_products:
        row = base_row.copy()
        row["ProductId"] = _encode_with_fallback(_metadata["encoders"]["ProductId"], product_id)
        X_input = np.array([[row.get(f, 0.0) for f in feature_names]], dtype=float)
        X_scaled = _metadata["scaler"].transform(X_input)
        if hasattr(_model, "predict_proba"):
            prob = float(_model.predict_proba(X_scaled)[0][1])
        else:
            prob = float(_model.predict(X_scaled)[0])
        scores.append((product_id, prob))

    scores.sort(key=lambda x: x[1], reverse=True)
    recommended = [p for p, _ in scores[:3]]
    confidence = round(scores[0][1] * 100, 2) if scores else 0.0

    categories = _metadata.get("category_labels", [])
    suggested = [req.product_category]
    if categories:
        idx = int(_encode_with_fallback(_metadata["encoders"]["Product_Category"], req.product_category))
        if idx < len(categories):
            suggested.append(str(categories[idx]))

    return {
        "recommended_products": recommended,
        "recommendation_confidence_score": confidence,
        "suggested_product_categories": list(dict.fromkeys(suggested))[:3],
        "selected_model": _metadata["best_model_name"],
        "model_accuracy": round(float(_metadata["accuracy"]) * 100, 2),
        "product_interest_type": req.product_interest_type,
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
            "product_popularity": "/results/image/product_popularity.png",
            "correlation_heatmap": "/results/image/correlation_heatmap.png",
            "feature_importance": "/results/image/feature_importance.png",
            "user_product_interaction": "/results/image/user_product_interaction.png",
            "recommendation_accuracy_comparison": "/results/image/recommendation_accuracy_comparison.png",
            "category_wise_recommendation": "/results/image/category_wise_recommendation.png",
            "customer_preference_trends": "/results/image/customer_preference_trends.png",
            "product_rating_distribution": "/results/image/product_rating_distribution.png",
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
    train_product_recommendation_model()
    load_product_recommendation_assets()


@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(_background_retrain)
    return {"status": "accepted", "message": "Product recommendation model retraining started."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Product Recommendation System unified ML pipeline and backend")
    parser.add_argument("--train", action="store_true", help="Force train the model and regenerate results")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI port (default: 8000)")
    args = parser.parse_args()

    if args.train or not os.path.exists(MODEL_FILE) or not os.path.exists(METADATA_FILE):
        try:
            train_product_recommendation_model()
            load_product_recommendation_assets()
        except Exception as exc:
            print(f"[ERROR] Training failed: {exc}")
            sys.exit(1)

    if "--train" not in sys.argv:
        import uvicorn
        print(f"[INFO] Starting Product Recommendation System API server on port {args.port}...")
        uvicorn.run("product_recommendation_system:app", host="127.0.0.1", port=args.port, reload=False)
