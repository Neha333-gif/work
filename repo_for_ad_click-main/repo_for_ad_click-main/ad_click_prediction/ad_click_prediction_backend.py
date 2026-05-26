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

# Initialize FastAPI app for Ad Click Prediction
app = FastAPI(title="AI Ad Click Prediction API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and preprocessors
model = None
preprocessors = None

def load_assets():
    model_name = 'ad_click_model.joblib'
    preprocessors_name = 'ad_click_preprocessors.joblib'
    
    paths_to_check = [
        '.',
        '..',
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    ]
    
    loaded_model = None
    loaded_preprocessors = None
    
    for path in paths_to_check:
        model_path = os.path.join(path, model_name)
        preprocessors_path = os.path.join(path, preprocessors_name)
        if os.path.exists(model_path) and os.path.exists(preprocessors_path):
            try:
                loaded_model = joblib.load(model_path)
                loaded_preprocessors = joblib.load(preprocessors_path)
                print(f"[INFO] Loaded model and preprocessors from {os.path.abspath(path)}")
                return loaded_model, loaded_preprocessors
            except Exception as e:
                print(f"[ERROR] Failed to load from {path}: {e}")
                
    print("[WARNING] Model or preprocessors not found. Please run ad_click_predictions.py first.")
    return None, None

# Initial load
model, preprocessors = load_assets()

# Ad click information database for engagement insights
ENGAGEMENT_INFO = {
    0: {
        "description": "No clicks predicted. The banner failed to engage the user under these conditions.",
        "insights": [
            "Optimize banner design or copy to increase visual appeal",
            "Consider changing target audience demographics or interests",
            "Adjust bidding strategies for better ad placement",
            "Check if the event date corresponds to a low-traffic day"
        ]
    },
    1: {
        "description": "Low engagement predicted. The user is likely to click once.",
        "insights": [
            "Monitor CTR and impressions for performance stability",
            "A/B test similar creative designs to scale performance",
            "Check if the frequency capping needs adjustment",
            "Explore contextual targeting options"
        ]
    },
    2: {
        "description": "Moderate engagement predicted. The user is expected to click twice.",
        "insights": [
            "Good engagement. Allocate more budget to this banner/audience segment",
            "Analyze conversion rates post-click to ensure ROI",
            "Review landing page alignment with the banner ad content",
            "Perform creative refreshes to avoid banner fatigue"
        ]
    },
    3: {
        "description": "High engagement predicted. Multiple click-throughs expected.",
        "insights": [
            "High performance ad interaction! Run lookalike campaigns on this segment",
            "Lock in premium ad placements for this banner ID",
            "Ensure inventory availability remains high for peak date ranges",
            "Document design patterns of this banner as a benchmark"
        ]
    },
    4: {
        "description": "Exceptional engagement predicted. Outstanding interaction with this banner!",
        "insights": [
            "Maximum ad engagement achieved! Prioritize budget allocation here",
            "Cross-promote this banner on other relevant ad channels",
            "Verify that click metrics are not fraudulent (e.g. accidental clicks)",
            "Promote this banner layout as a primary template for upcoming campaigns"
        ]
    }
}

class AdClickRequest(BaseModel):
    event_date: str = Field(..., description="Interaction date (e.g., YYYY-MM-DD)", example="2026-01-01")
    banner_id: str = Field(..., description="Banner ID identifier", example="b_0082")
    impressions: int = Field(..., description="Number of impressions shown", example=3)
    ctr: float = Field(..., description="Click-through rate (CTR)", example=0.0)

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serve the main ad click prediction frontend UI"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_path = os.path.join(base_dir, "ad_click_prediction_frontend.html")
    if os.path.exists(frontend_path):
        with open(frontend_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        # Fallback error response
        return HTMLResponse(content="<h1>Ad Click Prediction Frontend not found</h1><p>Please ensure ad_click_prediction_frontend.html is in the folder.</p>", status_code=404)

@app.get("/health")
def health():
    """Health check endpoint to verify model status"""
    status = "ok" if model is not None else "no_model_loaded"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check parent and local paths
    model_exists = (
        os.path.exists(os.path.join(base_dir, 'ad_click_model.joblib')) or 
        os.path.exists(os.path.join(base_dir, '..', 'ad_click_model.joblib'))
    )
    preprocessors_exist = (
        os.path.exists(os.path.join(base_dir, 'ad_click_preprocessors.joblib')) or 
        os.path.exists(os.path.join(base_dir, '..', 'ad_click_preprocessors.joblib'))
    )
    
    return {
        "status": status,
        "model": "Random Forest Ad Click Classifier",
        "assets_exist": {
            "model": model_exists,
            "preprocessors": preprocessors_exist
        }
    }

@app.get("/features")
def get_features():
    """Get unique banner list and date range details"""
    if preprocessors is None:
        raise HTTPException(status_code=503, detail="Model assets not loaded. Train the model first.")
        
    return {
        "unique_banners": preprocessors.get("unique_banners", []),
        "date_range": preprocessors.get("date_range", {}),
    }

@app.post("/predict")
def predict_ad_click(req: AdClickRequest):
    """Predict ad click number class and probability based on inputs"""
    global model, preprocessors
    if model is None or preprocessors is None:
        # Reload assets just in case they were generated since server startup
        model, preprocessors = load_assets()
        if model is None or preprocessors is None:
            raise HTTPException(status_code=503, detail="Model is not loaded. Please train a model first.")

    try:
        label_encoders = preprocessors["label_encoders"]
        scalers = preprocessors["scalers"]
        clicks_encoder = preprocessors["clicks_encoder"]
        
        # 1. Encode event_date
        event_date = req.event_date
        if "event_date" in label_encoders:
            try:
                encoded_date = label_encoders["event_date"].transform([event_date])[0]
            except Exception:
                encoded_date = 0  # Fallback
        else:
            encoded_date = 0
            
        # 2. Encode banner_id
        banner_id = req.banner_id
        if "banner_id" in label_encoders:
            try:
                encoded_banner = label_encoders["banner_id"].transform([banner_id])[0]
            except Exception:
                encoded_banner = 0  # Fallback
        else:
            encoded_banner = 0
            
        # 3. Scale impressions
        impressions = float(req.impressions)
        if "impressions" in scalers:
            scaled_impressions = scalers["impressions"].transform([[impressions]])[0][0]
        else:
            scaled_impressions = impressions
            
        # 4. Scale ctr
        ctr = float(req.ctr)
        if "ctr" in scalers:
            scaled_ctr = scalers["ctr"].transform([[ctr]])[0][0]
        else:
            scaled_ctr = ctr
            
        # Match training column order: event_date, banner_id, impressions, ctr
        features_df = pd.DataFrame(
            [[encoded_date, encoded_banner, scaled_impressions, scaled_ctr]],
            columns=['event_date', 'banner_id', 'impressions', 'ctr']
        )
        
        # Predict class and probabilities
        pred_class_id = int(model.predict(features_df)[0])
        probs = model.predict_proba(features_df)[0]
        
        # Resolve class clicks label (original clicks count)
        predicted_clicks = int(clicks_encoder.inverse_transform([pred_class_id])[0])
        confidence = float(probs[pred_class_id])
        
        # Fetch insights database details
        info = ENGAGEMENT_INFO.get(predicted_clicks, {
            "description": f"Predicted clicks: {predicted_clicks}",
            "insights": ["Monitor CTR and impressions for performance stability"]
        })
        
        # Format top probabilities for visualization
        class_probs = []
        for idx, prob in enumerate(probs):
            clicks_val = int(clicks_encoder.inverse_transform([idx])[0])
            class_probs.append({"clicks": clicks_val, "probability": float(prob)})
            
        class_probs = sorted(class_probs, key=lambda x: x['probability'], reverse=True)
        
        return {
            "prediction": predicted_clicks,
            "confidence": confidence,
            "description": info["description"],
            "insights": info["insights"],
            "top_probabilities": class_probs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/upload")
async def upload_csv_and_predict(file: UploadFile = File(...)):
    """Upload a CSV containing ad interaction rows and predict the clicks for each row"""
    global model, preprocessors
    if model is None or preprocessors is None:
        model, preprocessors = load_assets()
        if model is None or preprocessors is None:
            raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        contents = await file.read()
        df_input = pd.read_csv(io.BytesIO(contents))
        
        # Check required columns
        required_cols = ['event_date', 'banner_id', 'impressions', 'ctr']
        for col in required_cols:
            if col not in df_input.columns:
                raise HTTPException(status_code=400, detail=f"CSV must contain '{col}' column. Provided: {list(df_input.columns)}")
        
        predictions = []
        confidences = []
        
        label_encoders = preprocessors["label_encoders"]
        scalers = preprocessors["scalers"]
        clicks_encoder = preprocessors["clicks_encoder"]
        
        for idx, row in df_input.iterrows():
            event_date = str(row['event_date']).strip()
            banner_id = str(row['banner_id']).strip()
            try:
                impressions = float(row['impressions'])
            except ValueError:
                impressions = 0.0
            try:
                ctr = float(row['ctr'])
            except ValueError:
                ctr = 0.0
                
            # Process event_date
            try:
                encoded_date = label_encoders["event_date"].transform([event_date])[0]
            except Exception:
                encoded_date = 0
                
            # Process banner_id
            try:
                encoded_banner = label_encoders["banner_id"].transform([banner_id])[0]
            except Exception:
                encoded_banner = 0
                
            # Process impressions
            try:
                scaled_impressions = scalers["impressions"].transform([[impressions]])[0][0]
            except Exception:
                scaled_impressions = impressions
                
            # Process ctr
            try:
                scaled_ctr = scalers["ctr"].transform([[ctr]])[0][0]
            except Exception:
                scaled_ctr = ctr
                
            features_df = pd.DataFrame(
                [[encoded_date, encoded_banner, scaled_impressions, scaled_ctr]],
                columns=['event_date', 'banner_id', 'impressions', 'ctr']
            )
            
            pred_id = int(model.predict(features_df)[0])
            probs = model.predict_proba(features_df)[0]
            
            predicted_clicks = int(clicks_encoder.inverse_transform([pred_id])[0])
            conf = float(probs[pred_id])
            
            predictions.append(predicted_clicks)
            confidences.append(conf)
            
        df_input['Predicted Clicks'] = predictions
        df_input['Confidence Score'] = confidences
        
        # Return as downloadable CSV
        stream = io.StringIO()
        df_input.to_csv(stream, index=False)
        response = StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = "attachment; filename=predicted_ad_clicks_results.csv"
        return response
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@app.get("/results/metrics")
def get_metrics():
    """Get metrics and listings of generated plots"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    metrics_file = os.path.join(results_dir, "accuracy_results.txt")
    
    if not os.path.exists(metrics_file):
        raise HTTPException(status_code=404, detail="Metrics report not found. Run model training first.")
        
    with open(metrics_file, "r", encoding="utf-8") as f:
        content = f.read()
        
    return {
        "text_report": content,
        "plots": {
            "confusion_matrix": "/results/image/confusion_matrix.png",
            "feature_importance": "/results/image/feature_importance.png",
            "correlation_heatmap": "/results/image/correlation_heatmap.png",
            "class_distribution": "/results/image/class_distribution.png",
            "roc_curve": "/results/image/roc_curve.png"
        }
    }

@app.get("/results/image/{image_name}")
def get_results_image(image_name: str):
    """Serve generated plots from the results directory"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(base_dir, "results", image_name)
    if os.path.exists(image_path):
        return FileResponse(image_path)
    else:
        raise HTTPException(status_code=404, detail="Image not found")

def background_retrain():
    """Asynchronous background retraining task"""
    print("[RETRAIN] Starting background model training...")
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(base_dir, "..", "ad_click_predictions.py")
        if not os.path.exists(script_path):
            script_path = os.path.join(base_dir, "ad_click_predictions.py")
            
        print(f"[RETRAIN] Executing script: {script_path}")
        result = subprocess.run(["python", script_path], capture_output=True, text=True)
        if result.returncode == 0:
            print("[RETRAIN] Model trained successfully.")
            global model, preprocessors
            model, preprocessors = load_assets()
        else:
            print(f"[RETRAIN] Training script failed: {result.stderr}")
    except Exception as e:
        print(f"[RETRAIN] Error during background model training: {e}")

@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    """Retrain model in the background"""
    background_tasks.add_task(background_retrain)
    return {"status": "accepted", "message": "Model training initiated in the background."}
