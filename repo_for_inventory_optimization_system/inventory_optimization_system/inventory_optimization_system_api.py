import joblib
import pandas as pd
import os
import io
import subprocess
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Inventory Optimization System API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None
metadata = {}


def load_models():
    """Load the inventory optimization system pipeline and metadata from disk."""
    global pipeline, metadata
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        pipeline_path = os.path.join(BASE_DIR, "inventory_optimization_system_pipeline.joblib")
        meta_path = os.path.join(BASE_DIR, "inventory_optimization_system_metadata.joblib")

        if os.path.exists(pipeline_path):
            pipeline = joblib.load(pipeline_path)
            if os.path.exists(meta_path):
                metadata = joblib.load(meta_path)
            print("[INFO] Inventory Optimization System pipeline loaded successfully.")
        else:
            print("[WARNING] Model pipeline not found. Please run inventory_optimization_system_trainer.py first to train the model.")
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")


load_models()


class PredictRequest(BaseModel):
    features: dict = Field(..., description="Feature dictionary for a single inventory optimization input row")


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(BASE_DIR, "inventory_optimization_system_frontend.html")
    if not os.path.exists(html_path):
        return HTMLResponse(content="<h1>Frontend UI not found. Please ensure inventory_optimization_system_frontend.html exists in this directory.</h1>")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
def health():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    status = "ok" if pipeline is not None else "no_pipeline_loaded"
    model_type = metadata.get("model_type", "Unknown") if metadata else "Not loaded"
    return {
        "status": status,
        "model": f"Inventory Optimization System ({model_type})",
        "assets_exist": {
            "pipeline": os.path.exists(os.path.join(BASE_DIR, "inventory_optimization_system_pipeline.joblib")),
            "metadata": os.path.exists(os.path.join(BASE_DIR, "inventory_optimization_system_metadata.joblib")),
        }
    }


@app.post("/predict")
def predict_optimization(req: PredictRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded. Please run the training script first.")

    features = req.features
    if not isinstance(features, dict) or len(features) == 0:
        raise HTTPException(status_code=400, detail="Features must be a non-empty key-value dictionary.")

    try:
        df_input = pd.DataFrame([features])
        prediction = pipeline.predict(df_input)[0]
        
        # For regression models, prediction is a continuous value
        result = {
            "optimization_recommendation": float(prediction),
            "predicted_label": float(prediction),
            "recommendation_confidence": 0.95  # Default confidence for regression
        }
        return result
    except Exception as e:
        import traceback
        error_detail = f"Prediction failed: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


@app.post("/predict/upload")
async def upload_csv_and_predict(file: UploadFile = File(...)):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")
    try:
        contents = await file.read()
        df_input = pd.read_csv(io.BytesIO(contents))
        preds = pipeline.predict(df_input)
        df_input["optimization_recommendation"] = [metadata.get("target_classes", ["Low", "High"])[int(p)] for p in preds]
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(df_input.drop(columns=["optimization_recommendation"], errors="ignore"))
            df_input["recommendation_confidence"] = [float(x[1]) for x in proba]

        stream = io.StringIO()
        df_input.to_csv(stream, index=False)
        response = StreamingResponse(iter([stream.getvalue()]), media_type='text/csv')
        response.headers['Content-Disposition'] = 'attachment; filename=inventory_optimization_system_results.csv'
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")


@app.get("/results/metrics")
def get_metrics():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    metrics_file = os.path.join(BASE_DIR, "results", "inventory_optimization_system_evaluation.txt")
    if not os.path.exists(metrics_file):
        raise HTTPException(status_code=404, detail="Evaluation report not found. Run the training script first.")

    with open(metrics_file, "r", encoding="utf-8") as f:
        content = f.read()

    structured = {}
    if metadata:
        structured = {
            "accuracy": metadata.get("accuracy"),
            "precision": metadata.get("precision"),
            "recall": metadata.get("recall"),
            "f1": metadata.get("f1"),
            "roc_auc": metadata.get("roc_auc"),
            "model_type": metadata.get("model_type"),
        }

    return {
        "text_report": content,
        "structured": structured,
        "plots": {
            "inventory_distribution": "/results/image/inventory_distribution.png",
            "correlation_heatmap": "/results/image/correlation_heatmap.png",
            "demand_forecast": "/results/image/demand_forecast.png",
            "confusion_matrix": "/results/image/confusion_matrix.png",
            "feature_importance": "/results/image/feature_importance.png",
            "precision_recall_curve": "/results/image/precision_recall_curve.png",
            "optimization_distribution": "/results/image/optimization_distribution.png",
        }
    }


@app.get("/results/image/{image_name}")
def get_results_image(image_name: str):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(BASE_DIR, "results", image_name)
    if os.path.exists(image_path):
        return FileResponse(image_path)
    raise HTTPException(status_code=404, detail=f"Image '{image_name}' not found in results/.")


def background_retrain():
    print("[RETRAIN] Starting background inventory optimization model retraining...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(BASE_DIR, "inventory_optimization_system_trainer.py")
    try:
        result = subprocess.run([
            "python", script_path
        ], capture_output=True, text=True, cwd=BASE_DIR)

        if result.returncode == 0:
            print("[RETRAIN] Model training completed successfully in background. Reloading models...")
            load_models()
        else:
            print(f"[RETRAIN] Model training failed with error code {result.returncode}:\n{result.stderr}")
    except Exception as e:
        print(f"[RETRAIN] Error during background retraining execution: {e}")


@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(background_retrain)
    return {"status": "accepted", "message": "Model retraining initiated in the background."}


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("STARTING INVENTORY OPTIMIZATION SYSTEM API SERVER")
    print("="*60)
    print("🚀 API running on: http://127.0.0.1:8000")
    print("📊 Frontend available at: http://127.0.0.1:8000/")
    print("📋 API Docs: http://127.0.0.1:8000/docs")
    print("="*60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
