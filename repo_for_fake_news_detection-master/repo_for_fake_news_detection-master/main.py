import joblib
import pandas as pd
import numpy as np
import re
import os
import io
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from train_model import train_and_save_model

# Initialize FastAPI app for Fake News Detection
app = FastAPI(title="Fake News Detection API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
vectorizer = None
model = None
word_coefs = None
metadata = None

def load_models():
    global vectorizer, model, word_coefs, metadata
    try:
        if os.path.exists('tfidf_vectorizer.joblib'):
            vectorizer = joblib.load('tfidf_vectorizer.joblib')
            model = joblib.load('logistic_regression_model.joblib')
            word_coefs = joblib.load('word_coefficients.joblib')
            metadata = joblib.load('model_metadata.joblib')
            print("[INFO] Model assets loaded successfully.")
        else:
            print("[WARNING] Model assets not found. Server is running but requests might fail until a model is trained.")
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")

# Initial load
load_models()

class NewsRequest(BaseModel):
    title: str = Field(..., description="Title of the news article")
    text: str = Field(..., description="Full text content of the news article")

class NewsBatchRequest(BaseModel):
    articles: list[NewsRequest]

@app.get("/health")
def health():
    """Health check endpoint"""
    status = "ok" if model is not None else "no_model_loaded"
    return {
        "status": status, 
        "model": "Fake News Detection TF-IDF Classifier",
        "original_colab_model_exists": os.path.exists('fake news detection.pkl')
    }

@app.get("/info")
def get_info():
    """Get model training metrics and vocabulary metadata"""
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not loaded. Train a model first.")
    return metadata

def clean_word(word):
    """Clean word for lookup in TF-IDF coefficients"""
    return re.sub(r'[^\w]', '', word.lower())

def extract_explanation(title: str, text: str):
    """Extract suspicious (fake-associated) and trustworthy (true-associated) words"""
    if word_coefs is None:
        return {"suspicious_words": [], "trustworthy_words": []}
        
    full_content = f"{title} {text}"
    # Tokenize text into words (including alphanumeric and punctuation)
    words = re.findall(r'\b\w+\b', full_content)
    
    suspicious_list = []
    trustworthy_list = []
    
    seen_words = set()
    
    for word in words:
        cleaned = clean_word(word)
        if not cleaned or cleaned in seen_words:
            continue
            
        # Check if the word is in our vocabulary
        if cleaned in word_coefs:
            coef = word_coefs[cleaned]
            seen_words.add(cleaned)
            
            # Coefficients < 0 imply Fake Association (suspicious)
            # Coefficients > 0 imply True Association (trustworthy)
            # Filter minor/noisy coefficients (absolute value > 0.15)
            if coef < -0.15:
                suspicious_list.append({"word": cleaned, "score": float(coef)})
            elif coef > 0.15:
                trustworthy_list.append({"word": cleaned, "score": float(coef)})
                
    # Sort suspicious words by most negative score
    suspicious_list = sorted(suspicious_list, key=lambda x: x['score'])[:15]
    # Sort trustworthy words by most positive score
    trustworthy_list = sorted(trustworthy_list, key=lambda x: x['score'], reverse=True)[:15]
    
    return {
        "suspicious_words": suspicious_list,
        "trustworthy_words": trustworthy_list
    }

@app.post("/predict")
def predict_news(req: NewsRequest):
    """Predict whether a single article is True or Fake"""
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please train a model first.")
        
    try:
        combined_text = f"{req.title} {req.text}"
        
        # Preprocess and vectorize
        X_vec = vectorizer.transform([combined_text])
        
        # Predict probability
        # Class 0: Fake, Class 1: True
        probs = model.predict_proba(X_vec)[0]
        prob_fake = float(probs[0])
        prob_true = float(probs[1])
        
        prediction_val = int(model.predict(X_vec)[0])
        prediction_label = "True" if prediction_val == 1 else "Fake"
        confidence = prob_true if prediction_val == 1 else prob_fake
        
        # Extract word-level explanations
        explanation = extract_explanation(req.title, req.text)
        
        return {
            "prediction": prediction_label,
            "prediction_value": prediction_val,
            "confidence": confidence,
            "probability_true": prob_true,
            "probability_fake": prob_fake,
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
def predict_batch(req: NewsBatchRequest):
    """Predict truthfulness for multiple articles"""
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
        
    try:
        results = []
        for i, article in enumerate(req.articles):
            combined_text = f"{article.title} {article.text}"
            X_vec = vectorizer.transform([combined_text])
            
            probs = model.predict_proba(X_vec)[0]
            prob_fake = float(probs[0])
            prob_true = float(probs[1])
            
            prediction_val = int(model.predict(X_vec)[0])
            prediction_label = "True" if prediction_val == 1 else "Fake"
            confidence = prob_true if prediction_val == 1 else prob_fake
            
            results.append({
                "index": i,
                "title": article.title,
                "prediction": prediction_label,
                "confidence": confidence,
                "probability_true": prob_true
            })
            
        return {
            "count": len(req.articles),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/predict/upload")
async def upload_csv_and_predict(file: UploadFile = File(...)):
    """Upload a CSV file containing 'title' and 'text' columns, predict, and return the annotated CSV"""
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
        
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Check required columns
        if 'title' not in df.columns or 'text' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'title' and 'text' columns.")
            
        df['title'] = df['title'].fillna('')
        df['text'] = df['text'].fillna('')
        
        combined_texts = df['title'] + " " + df['text']
        X_vec = vectorizer.transform(combined_texts)
        
        probs = model.predict_proba(X_vec)
        probs_true = probs[:, 1]
        probs_fake = probs[:, 0]
        
        preds = model.predict(X_vec)
        df['prediction'] = ["True" if p == 1 else "Fake" for p in preds]
        df['confidence_score'] = [float(probs_true[i]) if preds[i] == 1 else float(probs_fake[i]) for i in range(len(preds))]
        df['probability_true'] = [float(p) for p in probs_true]
        
        # Save output in memory and return as download
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        response = StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = "attachment; filename=predicted_news_results.csv"
        return response
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

def background_retrain():
    """Background task to trigger model retraining"""
    print("[RETRAIN] Background retraining starting...")
    try:
        train_and_save_model()
        load_models()
        print("[RETRAIN] Background retraining complete and models loaded.")
    except Exception as e:
        print(f"[RETRAIN] Error during background retraining: {e}")

@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    """Trigger asynchronous model retraining"""
    background_tasks.add_task(background_retrain)
    return {"status": "accepted", "message": "Model retraining initiated in the background."}
