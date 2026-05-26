"""
Customer Churn Prediction Backend API

FastAPI server for predicting customer churn probability.
Serves predictions and analytics for customer retention insights.
"""

import joblib
import pandas as pd
import numpy as np
import os
import io
import subprocess
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="AI-powered customer churn prediction system",
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

# Global variables
model = None
scaler = None
feature_names = None

def load_assets():
    """Load trained model and preprocessors"""
    model_name = 'customer_churn_model.joblib'
    scaler_name = 'customer_churn_preprocessors.joblib'
    features_name = 'customer_churn_features.joblib'
    
    paths_to_check = [
        '.',
        '..',
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    ]
    
    for path in paths_to_check:
        model_path = os.path.join(path, model_name)
        scaler_path = os.path.join(path, scaler_name)
        features_path = os.path.join(path, features_name)
        
        if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path):
            try:
                loaded_model = joblib.load(model_path)
                loaded_scaler = joblib.load(scaler_path)
                loaded_features = joblib.load(features_path)
                print(f"[INFO] Loaded model and preprocessors from {os.path.abspath(path)}")
                return loaded_model, loaded_scaler, loaded_features
            except Exception as e:
                print(f"[ERROR] Failed to load from {path}: {e}")
                
    print("[WARNING] Model or preprocessors not found. Please run customer_churn.py first.")
    return None, None, None

# Initial load
model, scaler, feature_names = load_assets()

# Churn insights database
CHURN_INSIGHTS = {
    "high_churn_risk": {
        "title": "⚠️ HIGH CHURN RISK DETECTED",
        "description": "This customer has a high probability of churning. Immediate retention action is recommended.",
        "actions": [
            "Contact customer proactively with personalized retention offers",
            "Offer loyalty rewards or service upgrades",
            "Review service quality and satisfaction",
            "Provide dedicated support and attention",
            "Consider special pricing or contract incentives"
        ]
    },
    "medium_churn_risk": {
        "title": "⚡ MEDIUM CHURN RISK",
        "description": "This customer shows moderate churn indicators. Enhanced engagement recommended.",
        "actions": [
            "Monitor customer engagement and satisfaction metrics",
            "Send targeted communication with special offers",
            "Encourage account usage and feature adoption",
            "Schedule regular check-ins",
            "Offer complementary services or upgrades"
        ]
    },
    "low_churn_risk": {
        "title": "✅ LOW CHURN RISK",
        "description": "This customer appears satisfied and committed. Maintain current service level.",
        "actions": [
            "Continue providing excellent service",
            "Keep customer engaged with relevant updates",
            "Request feedback and testimonials",
            "Consider for upsell opportunities",
            "Treat as advocate for referral programs"
        ]
    }
}

class ChurnPredictionRequest(BaseModel):
    """Request model for customer churn prediction"""
    gender: str = Field(..., description="Customer gender (Male/Female)", example="Male")
    senior_citizen: int = Field(..., description="Senior citizen status (0/1)", example=0)
    tenure: int = Field(..., description="Months of customer tenure", example=24)
    monthly_charges: float = Field(..., description="Monthly charges in dollars", example=65.5)
    total_charges: float = Field(..., description="Total charges paid in dollars", example=1570.0)
    contract: str = Field(..., description="Contract type (Month-to-month/One year/Two year)", example="Month-to-month")
    internet_service: str = Field(..., description="Internet service type (DSL/Fiber optic/No)", example="Fiber optic")
    payment_method: str = Field(..., description="Payment method", example="Electronic check")

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serve the customer churn prediction frontend"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_path = os.path.join(base_dir, "customer_churn_prediction_frontend.html")
    if os.path.exists(frontend_path):
        with open(frontend_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(
            content="<h1>Customer Churn Prediction Frontend Not Found</h1><p>Please ensure customer_churn_prediction_frontend.html is in the folder.</p>",
            status_code=404
        )

@app.get("/health")
def health():
    """Health check endpoint"""
    status = "ok" if model is not None else "no_model_loaded"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_exists = (
        os.path.exists(os.path.join(base_dir, 'customer_churn_model.joblib')) or 
        os.path.exists(os.path.join(base_dir, '..', 'customer_churn_model.joblib'))
    )
    scaler_exists = (
        os.path.exists(os.path.join(base_dir, 'customer_churn_preprocessors.joblib')) or 
        os.path.exists(os.path.join(base_dir, '..', 'customer_churn_preprocessors.joblib'))
    )
    
    return {
        "status": status,
        "model": "Random Forest Customer Churn Classifier",
        "description": "Predicts customer churn probability",
        "assets": {
            "model_loaded": model_exists,
            "preprocessor_loaded": scaler_exists
        }
    }

@app.post("/predict")
def predict_churn(req: ChurnPredictionRequest):
    """Predict customer churn probability"""
    global model, scaler, feature_names
    
    if model is None or scaler is None or feature_names is None:
        model, scaler, feature_names = load_assets()
        if model is None or scaler is None:
            raise HTTPException(
                status_code=503,
                detail="Model is not loaded. Please train the model first by running customer_churn.py"
            )

    try:
        # Create feature vector with all 30 features
        # Initialize all features to 0
        feature_vector = {feature: 0 for feature in feature_names}
        
        # Set numeric features
        feature_vector['SeniorCitizen'] = req.senior_citizen
        feature_vector['tenure'] = req.tenure
        feature_vector['MonthlyCharges'] = req.monthly_charges
        feature_vector['TotalCharges'] = req.total_charges
        
        # Gender
        if req.gender.lower() == 'male':
            feature_vector['gender_Male'] = 1
        
        # Contract type
        if req.contract.lower() == 'one year':
            feature_vector['Contract_One year'] = 1
        elif req.contract.lower() == 'two year':
            feature_vector['Contract_Two year'] = 1
        
        # Internet Service
        if req.internet_service.lower() == 'fiber optic':
            feature_vector['InternetService_Fiber optic'] = 1
        elif req.internet_service.lower() == 'no':
            feature_vector['InternetService_No'] = 1
        
        # Payment Method
        if req.payment_method.lower() == 'electronic check':
            feature_vector['PaymentMethod_Electronic check'] = 1
        elif req.payment_method.lower() == 'mailed check':
            feature_vector['PaymentMethod_Mailed check'] = 1
        elif req.payment_method.lower() == 'credit card (automatic)':
            feature_vector['PaymentMethod_Credit card (automatic)'] = 1
        
        # Create input dataframe in correct column order
        input_df = pd.DataFrame([feature_vector])[feature_names]
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        input_df_scaled = pd.DataFrame(input_scaled, columns=feature_names)
        
        # Make prediction
        churn_prediction = model.predict(input_df_scaled)[0]
        churn_probability = model.predict_proba(input_df_scaled)[0]
        
        # Calculate probabilities
        churn_prob = float(churn_probability[1]) if len(churn_probability) > 1 else 0.0
        retention_prob = 1.0 - churn_prob
        
        # Determine risk level
        if churn_prob >= 0.7:
            risk_level = "high_churn_risk"
        elif churn_prob >= 0.4:
            risk_level = "medium_churn_risk"
        else:
            risk_level = "low_churn_risk"
        
        insights = CHURN_INSIGHTS.get(risk_level, CHURN_INSIGHTS["low_churn_risk"])
        
        return {
            "customer_id": "NEW_PREDICTION",
            "churn_prediction": "Likely to Churn" if churn_prediction == 'Yes' else "Likely to Stay",
            "churn_probability": round(churn_prob * 100, 2),
            "retention_probability": round(retention_prob * 100, 2),
            "risk_level": risk_level,
            "risk_title": insights["title"],
            "risk_description": insights["description"],
            "recommended_actions": insights["actions"],
            "model_confidence": round(max(churn_probability) * 100, 2)
        }
    except Exception as e:
        import traceback
        print(f"Error in prediction: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def batch_predict(file: UploadFile = File(...)):
    """Batch predict churn for multiple customers from CSV"""
    global model, scaler, feature_names
    
    if model is None or scaler is None:
        model, scaler, feature_names = load_assets()
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded.")
    
    try:
        contents = await file.read()
        df_input = pd.read_csv(io.BytesIO(contents))
        
        # Map columns
        required_cols = [
            'gender', 'senior_citizen', 'tenure', 'monthly_charges',
            'total_charges', 'contract', 'internet_service', 'payment_method'
        ]
        
        # Check for case-insensitive matches
        df_cols_lower = {col.lower(): col for col in df_input.columns}
        for required in required_cols:
            if required.lower() not in df_cols_lower:
                raise HTTPException(
                    status_code=400,
                    detail=f"CSV must contain '{required}' column"
                )
        
        predictions = []
        churn_probs = []
        retention_probs = []
        risk_levels = []
        
        for idx, row in df_input.iterrows():
            try:
                gender = str(row[df_cols_lower['gender']]).strip()
                senior = int(row[df_cols_lower['senior_citizen']])
                tenure = int(row[df_cols_lower['tenure']])
                monthly = float(row[df_cols_lower['monthly_charges']])
                total = float(row[df_cols_lower['total_charges']])
                contract = str(row[df_cols_lower['contract']]).strip()
                internet = str(row[df_cols_lower['internet_service']]).strip()
                payment = str(row[df_cols_lower['payment_method']]).strip()
                
                # Encode
                gender_encoded = 1 if gender.lower() == 'male' else 0
                contract_mapping = {'month-to-month': 0, 'one year': 1, 'two year': 2}
                contract_encoded = contract_mapping.get(contract.lower(), 0)
                internet_mapping = {'dsl': 0, 'fiber optic': 1, 'no': 2}
                internet_encoded = internet_mapping.get(internet.lower(), 2)
                payment_mapping = {
                    'electronic check': 0, 'mailed check': 1,
                    'bank transfer (automatic)': 2, 'credit card (automatic)': 3
                }
                payment_encoded = payment_mapping.get(payment.lower(), 0)
                
                # Create feature dataframe
                input_data = pd.DataFrame({
                    'Gender_Male': [gender_encoded],
                    'SeniorCitizen': [senior],
                    'Tenure': [tenure],
                    'MonthlyCharges': [monthly],
                    'TotalCharges': [total],
                    'Contract_One year': [1 if contract_encoded == 1 else 0],
                    'Contract_Two year': [1 if contract_encoded == 2 else 0],
                    'InternetService_Fiber optic': [1 if internet_encoded == 1 else 0],
                    'InternetService_No': [1 if internet_encoded == 2 else 0],
                    'PaymentMethod_Credit card (automatic)': [1 if payment_encoded == 3 else 0],
                    'PaymentMethod_Electronic check': [1 if payment_encoded == 0 else 0],
                    'PaymentMethod_Mailed check': [1 if payment_encoded == 1 else 0]
                })
                
                input_scaled = scaler.transform(input_data)
                input_df = pd.DataFrame(input_scaled, columns=input_data.columns)
                
                pred = model.predict(input_df)[0]
                probs = model.predict_proba(input_df)[0]
                churn_prob = float(probs[1]) if len(probs) > 1 else 0.0
                retention_prob = 1.0 - churn_prob
                
                if churn_prob >= 0.7:
                    risk = "High Risk"
                elif churn_prob >= 0.4:
                    risk = "Medium Risk"
                else:
                    risk = "Low Risk"
                
                predictions.append("Churn" if pred == 'Yes' else "Retain")
                churn_probs.append(round(churn_prob * 100, 2))
                retention_probs.append(round(retention_prob * 100, 2))
                risk_levels.append(risk)
            except Exception as e:
                predictions.append("Error")
                churn_probs.append(0.0)
                retention_probs.append(0.0)
                risk_levels.append("Error")
        
        df_input['Churn_Prediction'] = predictions
        df_input['Churn_Probability_%'] = churn_probs
        df_input['Retention_Probability_%'] = retention_probs
        df_input['Risk_Level'] = risk_levels
        
        # Return as CSV
        stream = io.StringIO()
        df_input.to_csv(stream, index=False)
        response = StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = "attachment; filename=customer_churn_predictions.csv"
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@app.get("/results/metrics")
def get_metrics():
    """Get model performance metrics and results"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    metrics_file = os.path.join(results_dir, "accuracy_results.txt")
    
    if not os.path.exists(metrics_file):
        raise HTTPException(
            status_code=404,
            detail="Metrics report not found. Run customer_churn.py first."
        )
    
    with open(metrics_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    return {
        "text_report": content,
        "visualizations": {
            "confusion_matrix": "/results/image/confusion_matrix.png",
            "feature_importance": "/results/image/feature_importance.png",
            "correlation_heatmap": "/results/image/correlation_heatmap.png",
            "class_distribution": "/results/image/class_distribution.png",
            "roc_curve": "/results/image/roc_curve.png",
            "precision_recall": "/results/image/precision_recall_curve.png"
        }
    }

@app.get("/results/image/{image_name}")
def get_results_image(image_name: str):
    """Serve visualization images from results folder"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(base_dir, "results", image_name)
    if os.path.exists(image_path):
        return FileResponse(image_path)
    else:
        raise HTTPException(status_code=404, detail="Image not found")

@app.get("/info")
def get_info():
    """Get API information"""
    return {
        "title": "Customer Churn Prediction API",
        "version": "1.0.0",
        "description": "Predicts customer churn probability using machine learning",
        "endpoints": {
            "/": "Serve frontend UI",
            "/health": "Health check",
            "/predict": "Predict single customer churn",
            "/predict/batch": "Batch predict from CSV",
            "/results/metrics": "Get model metrics",
            "/info": "This information"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
