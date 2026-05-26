import joblib
import pandas as pd
import numpy as np
import os
import io
import subprocess
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app for Email Spam Prediction
app = FastAPI(title="AI Email Spam Prediction API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model artifacts
model = None
vectorizer = None
label_encoder = None
metadata = {}

def load_models():
    """Load all email spam prediction model artifacts from disk."""
    global model, vectorizer, label_encoder, metadata
    try:
        model_path     = "email_spam_model.joblib"
        vec_path       = "email_spam_vectorizer.joblib"
        le_path        = "email_spam_label_encoder.joblib"
        meta_path      = "email_spam_metadata.joblib"

        if os.path.exists(model_path) and os.path.exists(vec_path) and os.path.exists(le_path):
            model         = joblib.load(model_path)
            vectorizer    = joblib.load(vec_path)
            label_encoder = joblib.load(le_path)
            if os.path.exists(meta_path):
                metadata  = joblib.load(meta_path)
            print("[INFO] Email Spam Prediction models loaded successfully.")
        else:
            print("[WARNING] Model assets not found. Run email_spam_prediction.py first to train the model.")
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")

# Load on startup
load_models()


class EmailRequest(BaseModel):
    email_text: str = Field(..., description="Raw email body text to classify as spam or ham")


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serve the Email Spam Prediction frontend UI."""
    html_path = "email_spam_prediction_frontend.html"
    if not os.path.exists(html_path):
        return HTMLResponse(content="<h1>Frontend not found. Please place email_spam_prediction_frontend.html in this directory.</h1>")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
def health():
    """Health check endpoint to verify model status."""
    status = "ok" if model is not None else "no_model_loaded"
    model_type = metadata.get("model_type", "Unknown") if metadata else "Not loaded"
    return {
        "status": status,
        "model": f"Email Spam Classifier ({model_type})",
        "assets_exist": {
            "model":         os.path.exists("email_spam_model.joblib"),
            "vectorizer":    os.path.exists("email_spam_vectorizer.joblib"),
            "label_encoder": os.path.exists("email_spam_label_encoder.joblib"),
        }
    }


@app.post("/predict")
def predict_spam(req: EmailRequest):
    """
    Classify a single email body as spam or ham.
    Returns prediction label, confidence score, top indicator words found.
    """
    if model is None or vectorizer is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please run the training script first.")

    text = req.email_text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Email text cannot be empty.")

    try:
        # Vectorize input
        vec_input = vectorizer.transform([text])

        # Predict
        pred_encoded  = int(model.predict(vec_input)[0])
        probs         = model.predict_proba(vec_input)[0]
        prediction    = str(label_encoder.inverse_transform([pred_encoded])[0])
        spam_prob     = float(probs[1])   # probability of class=spam
        ham_prob      = float(probs[0])

        # Explainability: which top indicator words appear in the email?
        top_indicators = metadata.get("top_spam_indicators", [])
        email_lower    = text.lower()
        matched_words  = [
            item for item in top_indicators[:50]
            if item["word"].lower() in email_lower
        ][:10]

        return {
            "prediction":      prediction,           # "spam" or "ham"
            "spam_probability": spam_prob,
            "ham_probability":  ham_prob,
            "confidence":       max(spam_prob, ham_prob),
            "is_spam":          prediction == "spam",
            "matched_spam_indicators": matched_words,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/upload")
async def upload_csv_and_predict(file: UploadFile = File(...)):
    """
    Upload a CSV containing email text rows and predict spam/ham for each.
    Accepts columns named: 'text', 'email', 'v2', or 'message'.
    Returns annotated CSV with Prediction and Confidence columns.
    """
    if model is None or vectorizer is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        contents = await file.read()
        df_input = pd.read_csv(io.BytesIO(contents))

        # Auto-detect text column
        text_col = None
        for candidate in ['text', 'email', 'v2', 'message', 'body']:
            if candidate in df_input.columns:
                text_col = candidate
                break

        if text_col is None:
            # Use first string column
            for col in df_input.columns:
                if df_input[col].dtype == object:
                    text_col = col
                    break

        if text_col is None:
            raise HTTPException(status_code=400, detail="No text column found in CSV. Use 'text', 'email', 'v2', or 'message'.")

        texts = df_input[text_col].fillna("").astype(str).tolist()
        vecs  = vectorizer.transform(texts)

        preds     = label_encoder.inverse_transform(model.predict(vecs))
        probs     = model.predict_proba(vecs)
        spam_prob = probs[:, 1]

        df_input["Prediction"]      = preds
        df_input["Spam_Probability"] = np.round(spam_prob, 4)
        df_input["Confidence"]      = np.round(np.max(probs, axis=1), 4)

        stream = io.StringIO()
        df_input.to_csv(stream, index=False)
        response = StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = "attachment; filename=spam_prediction_results.csv"
        return response

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")


@app.get("/results/metrics")
def get_metrics():
    """Return the saved accuracy report and available plot filenames."""
    metrics_file = "results/accuracy_results.txt"
    if not os.path.exists(metrics_file):
        raise HTTPException(status_code=404, detail="Metrics report not found. Run the training script first.")

    with open(metrics_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Also embed stored metadata metrics for structured access
    structured = {}
    if metadata:
        structured = {
            "accuracy":  metadata.get("accuracy"),
            "precision": metadata.get("precision"),
            "recall":    metadata.get("recall"),
            "f1_score":  metadata.get("f1_score"),
            "model_type": metadata.get("model_type"),
        }

    return {
        "text_report": content,
        "structured":  structured,
        "plots": {
            "confusion_matrix":    "/results/image/confusion_matrix.png",
            "class_distribution":  "/results/image/class_distribution.png",
            "roc_curve":           "/results/image/roc_curve.png",
            "feature_importance":  "/results/image/feature_importance.png",
            "correlation_heatmap": "/results/image/correlation_heatmap.png",
        }
    }


@app.get("/results/image/{image_name}")
def get_results_image(image_name: str):
    """Serve a generated plot PNG from the results/ directory."""
    image_path = os.path.join("results", image_name)
    if os.path.exists(image_path):
        return FileResponse(image_path)
    raise HTTPException(status_code=404, detail=f"Image '{image_name}' not found in results/.")


def background_retrain():
    """Background task: retrain the email spam model and reload assets."""
    print("[RETRAIN] Starting background email spam model training...")
    # Script is one directory up inside email program.py folder
    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "email program.py", "email_spam_prediction.py"
    )
    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))
        )
        if result.returncode == 0:
            print("[RETRAIN] Training completed. Reloading models...")
            load_models()
        else:
            print(f"[RETRAIN] Training failed:\n{result.stderr}")
    except Exception as e:
        print(f"[RETRAIN] Error: {e}")


@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    """Trigger email spam model retraining in the background."""
    background_tasks.add_task(background_retrain)
    return {"status": "accepted", "message": "Email spam model retraining initiated in the background."}
