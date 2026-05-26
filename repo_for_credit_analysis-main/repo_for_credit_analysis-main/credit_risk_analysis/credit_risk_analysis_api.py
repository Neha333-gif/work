# -*- coding: utf-8 -*-
"""credit_risk_analysis_api.py

FastAPI backend for credit risk analysis model serving and predictions.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn


# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Analysis API",
    description="API for credit risk classification predictions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(BASE_DIR, 'credit_risk_analysis_pipeline.joblib')
METADATA_PATH = os.path.join(BASE_DIR, 'credit_risk_analysis_metadata.joblib')

pipeline = None
metadata = None

if os.path.exists(PIPELINE_PATH) and os.path.exists(METADATA_PATH):
    pipeline = joblib.load(PIPELINE_PATH)
    metadata = joblib.load(METADATA_PATH)
else:
    print("[WARNING] Model artifacts not found. Train the model first using credit_risk_analysis_trainer.py")


class CreditRiskRequest(BaseModel):
    """Input model for credit risk prediction."""
    features: Dict[str, float]


class CreditRiskResponse(BaseModel):
    """Output model for credit risk prediction."""
    prediction: int
    probability: float
    risk_level: str
    message: str


@app.get("/")
def read_root():
    """Root endpoint."""
    return {
        "service": "Credit Risk Analysis API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    if pipeline is None or metadata is None:
        return {"status": "warning", "message": "Model not loaded"}
    return {"status": "healthy", "model": metadata.get('model_type', 'unknown')}


@app.get("/metadata")
def get_metadata():
    """Get model metadata and metrics."""
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": metadata.get('model_type'),
        "metrics": {
            "accuracy": metadata.get('accuracy'),
            "precision": metadata.get('precision'),
            "recall": metadata.get('recall'),
            "f1_score": metadata.get('f1'),
            "roc_auc": metadata.get('roc_auc')
        },
        "all_models_metrics": metadata.get('all_metrics'),
        "feature_names": metadata.get('feature_names')
    }


@app.post("/predict", response_model=CreditRiskResponse)
def predict_credit_risk(request: CreditRiskRequest):
    """Predict credit risk for given features."""
    if pipeline is None or metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input data
        feature_names = metadata.get('feature_names', [])
        input_data = pd.DataFrame([request.features])
        
        # Make prediction
        prediction = pipeline.predict(input_data)[0]
        probabilities = pipeline.predict_proba(input_data)[0]
        probability = float(probabilities[int(prediction)])
        
        # Determine risk level
        if prediction == 1:
            risk_level = "HIGH RISK" if probability > 0.7 else "MEDIUM RISK"
            message = f"This applicant is likely to default with probability {probability:.2%}"
        else:
            risk_level = "LOW RISK"
            message = f"This applicant is unlikely to default with probability {probability:.2%}"
        
        return CreditRiskResponse(
            prediction=int(prediction),
            probability=probability,
            risk_level=risk_level,
            message=message
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/batch_predict")
def batch_predict_credit_risk(requests: List[CreditRiskRequest]):
    """Batch predict credit risk for multiple samples."""
    if pipeline is None or metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        feature_names = metadata.get('feature_names', [])
        input_data = pd.DataFrame([r.features for r in requests])
        
        predictions = pipeline.predict(input_data)
        probabilities = pipeline.predict_proba(input_data)
        
        results = []
        for i, pred in enumerate(predictions):
            prob = float(probabilities[i][int(pred)])
            if pred == 1:
                risk_level = "HIGH RISK" if prob > 0.7 else "MEDIUM RISK"
            else:
                risk_level = "LOW RISK"
            
            results.append({
                "prediction": int(pred),
                "probability": prob,
                "risk_level": risk_level
            })
        
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")


@app.get("/results/{file_name}")
def get_result_file(file_name: str):
    """Download result files (visualizations, evaluation reports, predictions)."""
    results_dir = os.path.join(BASE_DIR, 'results')
    file_path = os.path.join(results_dir, file_name)
    
    if not os.path.exists(file_path) or not file_path.startswith(results_dir):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)


@app.get("/results/list")
def list_results():
    """List all available result files."""
    results_dir = os.path.join(BASE_DIR, 'results')
    if not os.path.exists(results_dir):
        return {"files": []}
    
    files = os.listdir(results_dir)
    return {"files": files}


if __name__ == "__main__":
    print("Starting Credit Risk Analysis API...")
    print(f"Model status: {'Loaded' if pipeline is not None else 'Not loaded'}")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
