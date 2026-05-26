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
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app for Disease Prediction
app = FastAPI(title="AI Disease Prediction API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
model = None
disease_encoder = None
symptom_encoder_data = None
symptom_to_int = {}
unique_symptoms = []
model_type = None

def load_models():
    global model, disease_encoder, symptom_encoder_data, symptom_to_int, unique_symptoms, model_type
    try:
        if os.path.exists('disease_label_encoder.joblib') and os.path.exists('symptom_encoder.joblib'):
            disease_encoder = joblib.load('disease_label_encoder.joblib')
            symptom_encoder_data = joblib.load('symptom_encoder.joblib')
            symptom_to_int = symptom_encoder_data["symptom_to_int"]
            unique_symptoms = symptom_encoder_data["unique_symptoms"]
        else:
            print("[WARNING] Encoder or symptom metadata missing. Train the model first.")
            return

        if os.path.exists('logistic_regression_model.joblib'):
            try:
                candidate = joblib.load('logistic_regression_model.joblib')
                if getattr(candidate, 'n_features_in_', None) == 17:
                    model = candidate
                    model_type = "Logistic Regression (lightweight)"
                    print("[INFO] Loaded lightweight logistic regression disease prediction model.")
                else:
                    print(f"[WARNING] Skipping logistic regression model because it expects {getattr(candidate, 'n_features_in_', '?')} features, not 17.")
            except Exception as e:
                print(f"[WARNING] Failed to load logistic regression model: {e}. Falling back to XGBoost if available.")

        if model is None and os.path.exists('xgboost_disease_model.joblib'):
            try:
                model = joblib.load('xgboost_disease_model.joblib')
                model_type = "XGBoost"
                print("[INFO] Loaded XGBoost disease prediction model.")
            except Exception as e:
                print(f"[ERROR] Failed to load XGBoost model: {e}")

        if model is None:
            print("[WARNING] No model could be loaded. Please train the disease prediction model.")
        else:
            print("[INFO] Disease Prediction models loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")

# Initial load
load_models()

# Disease information database for premium details
DISEASE_INFO = {
    "Fungal infection": {
        "description": "A skin infection caused by a fungus, leading to irritation, redness, itching, and scaling.",
        "precautions": ["Keep skin clean and dry", "Avoid sharing personal items", "Wear loose cotton clothes", "Use antifungal creams as prescribed"]
    },
    "Allergy": {
        "description": "An immune response to foreign substances (allergens) such as pollen, dust mites, or certain foods.",
        "precautions": ["Avoid known allergy triggers", "Keep indoor air clean with HEPA filters", "Take antihistamines if recommended", "Wash bedding in hot water weekly"]
    },
    "GERD": {
        "description": "Gastroesophageal Reflux Disease (GERD) is a chronic digestive disease where stomach acid flows back into the food pipe.",
        "precautions": ["Avoid trigger foods (spicy, fatty, caffeine)", "Eat smaller meals frequently", "Do not lie down right after eating", "Elevate the head of your bed"]
    },
    "Chronic cholestasis": {
        "description": "A condition where the flow of bile from the liver is reduced or blocked, leading to itching and yellow skin.",
        "precautions": ["Follow a low-fat diet", "Avoid alcohol completely", "Stay hydrated", "Consult a hepatologist immediately"]
    },
    "Drug Reaction": {
        "description": "An adverse reaction by the body to a medication, manifesting in skin rashes, itching, or fever.",
        "precautions": ["Stop the suspected drug immediately", "Consult your physician for alternatives", "Seek emergency help if breathing is difficult", "Always carry an allergy card"]
    },
    "Peptic ulcer diseae": {
        "description": "Sores that develop on the lining of the stomach, lower esophagus, or small intestine.",
        "precautions": ["Avoid NSAID pain relievers", "Limit alcohol and smoking", "Eat smaller, non-spicy meals", "Reduce emotional stress"]
    },
    "AIDS": {
        "description": "Acquired Immunodeficiency Syndrome (AIDS) is a chronic, potentially life-threatening condition caused by HIV.",
        "precautions": ["Practice safe sexual relations", "Never share needles/syringes", "Take prescribed antiretroviral therapy (ART)", "Regular health checkups and immune screening"]
    },
    "Diabetes": {
        "description": "A group of diseases that result in too much sugar in the blood (high blood glucose).",
        "precautions": ["Monitor blood sugar levels regularly", "Adopt a low-carb, high-fiber diet", "Exercise at least 30 minutes daily", "Take insulin or oral medications strictly as prescribed"]
    },
    "Gastroenteritis": {
        "description": "An intestinal infection marked by diarrhea, cramps, nausea, vomiting, and fever (often called stomach flu).",
        "precautions": ["Drink plenty of fluids/ORS to prevent dehydration", "Eat bland foods like bananas and rice", "Wash hands thoroughly with soap", "Avoid dairy and caffeine"]
    },
    "Bronchial Asthma": {
        "description": "A condition in which a person's airways become inflamed, narrow, and swell, making it difficult to breathe.",
        "precautions": ["Keep your inhaler with you at all times", "Identify and avoid asthma triggers (dust, cold air)", "Get annual flu vaccinations", "Follow an asthma action plan"]
    },
    "Hypertension": {
        "description": "A condition in which the force of the blood against the artery walls is too high (high blood pressure).",
        "precautions": ["Reduce salt intake in diet", "Exercise regularly", "Manage stress levels", "Avoid tobacco and limit alcohol"]
    },
    "Migraine": {
        "description": "A neurological condition that causes intense, debilitating headaches, often accompanied by nausea and light sensitivity.",
        "precautions": ["Avoid trigger foods like aged cheese and caffeine", "Maintain a regular sleep schedule", "Stay in a dark, quiet room during an attack", "Apply cold compresses to your forehead"]
    },
    "Cervical spondylosis": {
        "description": "Age-related wear and tear affecting the spinal disks in your neck.",
        "precautions": ["Avoid sitting in one posture for too long", "Do gentle neck exercises daily", "Use a supportive cervical pillow", "Apply heat/cold therapy for pain relief"]
    },
    "Paralysis (brain hemorrhage)": {
        "description": "Loss of muscle function in parts of your body caused by bleeding in the brain.",
        "precautions": ["Control high blood pressure immediately", "Go for physical/occupational therapy", "Avoid sudden physical strain", "Adhere strictly to anti-platelet therapy"]
    },
    "Jaundice": {
        "description": "A yellowing of the skin and eyes caused by high levels of bilirubin in the blood, indicating liver issues.",
        "precautions": ["Drink boiled, warm water", "Eat a light, easily digestible diet", "Avoid fatty and junk foods completely", "Rest extensively and consult a doctor"]
    },
    "Malaria": {
        "description": "A disease caused by a plasmodium parasite, transmitted by the bite of infected mosquitoes.",
        "precautions": ["Use mosquito nets and repellents", "Wear long-sleeved clothes", "Clear stagnant water near your home", "Take antimalarial medications as prescribed"]
    },
    "Chicken pox": {
        "description": "A highly contagious viral infection causing an itchy, blister-like rash on the skin.",
        "precautions": ["Isolate the patient to prevent spread", "Avoid scratching the blisters (causes scars)", "Take oatmeal baths to relieve itching", "Use calamine lotion on rashes"]
    },
    "Dengue": {
        "description": "A mosquito-borne viral disease causing high fever, severe headache, joint pain, and skin rashes.",
        "precautions": ["Stay hydrated (drink water, juices, coconut water)", "Use mosquito repellents", "Take paracetamol for pain (avoid aspirin/ibuprofen)", "Monitor platelet counts regularly"]
    },
    "Typhoid": {
        "description": "A bacterial infection spread through contaminated food and water, causing high fever and gastrointestinal issues.",
        "precautions": ["Drink only boiled or bottled water", "Eat thoroughly cooked, hot food", "Maintain strict personal hygiene", "Complete the full course of antibiotics"]
    },
    "hepatitis A": {
        "description": "An infectious liver disease caused by the Hepatitis A virus, usually spread through contaminated food/water.",
        "precautions": ["Avoid street food and raw foods", "Wash hands before eating", "Drink clean, purified water", "Get vaccinated against Hepatitis A"]
    },
    "Hepatitis B": {
        "description": "A serious liver infection caused by the Hepatitis B virus, spread through body fluids.",
        "precautions": ["Avoid sharing razors, toothbrushes, or needles", "Practice safe intercourse", "Get Hepatitis B vaccination", "Regular liver monitoring"]
    },
    "Hepatitis C": {
        "description": "An infection caused by the Hepatitis C virus, attacking the liver and causing inflammation.",
        "precautions": ["Do not share personal grooming items", "Ensure sterile equipment for tattoos/piercings", "Avoid alcohol", "Take antiviral therapy as prescribed"]
    },
    "Hepatitis D": {
        "description": "A liver disease that only occurs in people who are already infected with Hepatitis B.",
        "precautions": ["Follow Hepatitis B precautions strictly", "Get Hepatitis B vaccination (prevents Hep D)", "Avoid sharing needles or personal items", "Consult a hepatologist"]
    },
    "Hepatitis E": {
        "description": "A liver disease caused by Hepatitis E virus, commonly transmitted through fecal-contaminated drinking water.",
        "precautions": ["Drink clean, filtered, or boiled water", "Ensure proper sanitation and hygiene", "Avoid raw shellfish and uncooked meats", "Rest and stay hydrated"]
    },
    "Alcoholic hepatitis": {
        "description": "Inflammation of the liver caused by heavy, long-term alcohol consumption.",
        "precautions": ["Stop drinking alcohol completely and permanently", "Eat a nutrient-rich, balanced diet", "Take prescribed vitamin supplements", "Regularly screen liver function"]
    },
    "Tuberculosis": {
        "description": "A serious infectious bacterial disease that mainly affects the lungs, spread through cough droplets.",
        "precautions": ["Wear a mask in public places", "Isolate in a well-ventilated room", "Take the full 6-9 month DOTS course of medicines", "Eat high-protein foods to boost immunity"]
    },
    "Common Cold": {
        "description": "A common viral infection of the nose and throat, causing sneezing, sore throat, runny nose, and cough.",
        "precautions": ["Rest well and drink warm fluids", "Wash hands frequently", "Gargle with warm salt water", "Avoid exposure to cold winds"]
    },
    "Pneumonia": {
        "description": "An infection that inflames the air sacs in one or both lungs, which may fill with fluid or pus.",
        "precautions": ["Keep warm and rest", "Take prescribed antibiotics or antivirals fully", "Use a steam inhaler to clear lungs", "Get pneumonia/influenza vaccines"]
    },
    "Dimorphic hemmorhoids(piles)": {
        "description": "Swollen veins in your anus and lower rectum, similar to varicose veins, causing pain and bleeding.",
        "precautions": ["Eat a high-fiber diet (fruits, vegetables, whole grains)", "Drink plenty of water", "Avoid straining during bowel movements", "Take warm sitz baths daily"]
    },
    "Heart attack": {
        "description": "A medical emergency where flow of blood to the heart muscle is severely blocked, causing tissue damage.",
        "precautions": ["Call emergency medical services immediately", "Chew an aspirin if recommended by dispatch", "Sit down and stay calm", "Adopt a heart-healthy diet post-recovery"]
    },
    "Varicose veins": {
        "description": "Gnarled, enlarged veins, most commonly appearing in the legs due to weak valve function.",
        "precautions": ["Avoid standing or sitting for long periods", "Elevate your legs when resting", "Wear compression stockings", "Exercise regularly to improve circulation"]
    },
    "Hypothyroidism": {
        "description": "A condition in which the thyroid gland doesn't produce enough thyroid hormone, slowing metabolism.",
        "precautions": ["Take thyroid hormone replacement pills in the morning", "Ensure adequate iodine intake", "Monitor thyroid hormone levels (TSH) regularly", "Stay active to counter fatigue"]
    },
    "Hyperthyroidism": {
        "description": "The overproduction of thyroid hormones by the thyroid gland, accelerating your body's metabolism.",
        "precautions": ["Take anti-thyroid medications strictly", "Include calcium and vitamin D in your diet", "Limit caffeine and stimulants", "Manage stress and anxiety"]
    },
    "Hypoglycemia": {
        "description": "An abnormally low blood sugar level, commonly occurring in people with diabetes who miss meals or take excess insulin.",
        "precautions": ["Always carry fast-acting sugar (candy, juice)", "Eat meals at consistent times", "Check blood glucose levels frequently", "Educate family on glucagon injection"]
    },
    "Osteoarthristis": {
        "description": "The most common form of arthritis, caused by the gradual breakdown of protective joint cartilage.",
        "precautions": ["Engage in low-impact exercise (swimming, cycling)", "Maintain a healthy body weight", "Apply hot/cold packs to painful joints", "Use supportive footwear"]
    },
    "Arthritis": {
        "description": "Inflammation of one or more joints, causing pain, stiffness, and reduced range of motion.",
        "precautions": ["Perform gentle range-of-motion stretching", "Eat an anti-inflammatory diet (omega-3s)", "Avoid lifting heavy loads", "Take prescribed joint supplements"]
    },
    "(vertigo) Paroymsal  Positional Vertigo": {
        "description": "A condition causing sudden sensation of spinning or dizziness, triggered by specific head movements.",
        "precautions": ["Change positions slowly (e.g. sit up slowly)", "Avoid bending down suddenly", "Perform Epley maneuvers if trained", "Sit down immediately during a dizzy spell"]
    },
    "Acne": {
        "description": "A skin condition that occurs when hair follicles become plugged with oil and dead skin cells.",
        "precautions": ["Wash your face twice daily with a gentle cleanser", "Avoid touching or picking pimples", "Use non-comedogenic (oil-free) cosmetics", "Drink plenty of water and limit sugary foods"]
    },
    "Urinary tract infection": {
        "description": "An infection in any part of the urinary system, most commonly the bladder or kidneys.",
        "precautions": ["Drink plenty of water to flush bacteria", "Maintain clean personal hygiene", "Do not hold urine for long periods", "Take the full antibiotic prescription"]
    },
    "Psoriasis": {
        "description": "A skin disease that causes itchy, scaly red patches, most commonly on the knees, elbows, and scalp.",
        "precautions": ["Keep skin well-moisturized", "Avoid triggers like stress and smoking", "Get moderate, safe sunlight exposure", "Use medicated soaps or coal tar products"]
    },
    "Impetigo": {
        "description": "A highly contagious bacterial skin infection, causing sores and crusts, usually on the face.",
        "precautions": ["Keep sores clean and loosely covered", "Wash the infected person's linen separately", "Wash hands frequently with antibacterial soap", "Apply antibiotic ointment as directed"]
    }
}

class SymptomRequest(BaseModel):
    symptoms: list[str] = Field(..., description="List of patient symptoms")
    age: Optional[int] = Field(None, description="Patient age (optional)")
    gender: Optional[str] = Field(None, description="Patient gender (optional)")

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serve the main disease prediction frontend UI"""
    with open("disease_prediction_frontend.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/disease_prediction_frontend.html", response_class=HTMLResponse)
def serve_disease_frontend():
    """Serve the disease prediction frontend UI at a stable file path."""
    with open("disease_prediction_frontend.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/health")
def health():
    """Health check endpoint to verify model status"""
    status = "ok" if model is not None else "no_model_loaded"
    return {
        "status": status,
        "model": model_type or "No model loaded",
        "assets_exist": {
            "xgboost_model": os.path.exists('xgboost_disease_model.joblib'),
            "logistic_model": os.path.exists('logistic_regression_model.joblib'),
            "disease_encoder": os.path.exists('disease_label_encoder.joblib'),
            "symptom_encoder": os.path.exists('symptom_encoder.joblib')
        }
    }

@app.get("/symptoms")
def get_symptoms():
    """Get list of clean symptoms alphabetical order"""
    if not unique_symptoms:
        raise HTTPException(status_code=503, detail="Model vocabulary not loaded. Train a model first.")
    return {"symptoms": unique_symptoms}

@app.get("/diseases")
def get_diseases():
    """Get all diagnosis entries from the disease knowledge base."""
    diseases = []
    for name, info in DISEASE_INFO.items():
        diseases.append({
            "name": name,
            "description": info.get("description", ""),
            "precautions": info.get("precautions", [])
        })
    return {"diseases": diseases}

@app.post("/predict")
def predict_disease(req: SymptomRequest):
    """Predict disease based on symptom list"""
    if model is None or disease_encoder is None or not symptom_to_int:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please train a model first.")
    
    if not req.symptoms:
        raise HTTPException(status_code=400, detail="Symptom list cannot be empty.")
    if len(req.symptoms) > 5:
        raise HTTPException(status_code=400, detail="Please provide no more than 5 symptoms.")

    try:
        # Use up to 5 symptoms and pad to the model's expected length of 17 inputs
        padded_symptoms = req.symptoms[:5]
        while len(padded_symptoms) < 17:
            padded_symptoms.append('None')
            
        # Map string symptoms to integer features
        encoded_features = []
        for sym in padded_symptoms:
            val = symptom_to_int.get(sym, 0) # Fallback to 0 if unknown/None
            encoded_features.append(val)
            
        # Create input array
        features_arr = np.array([encoded_features])
        
        # Predict class and probabilities
        pred_class_id = int(model.predict(features_arr)[0])
        probs = model.predict_proba(features_arr)[0]
        
        # Resolve labels
        disease_name = str(disease_encoder.inverse_transform([pred_class_id])[0])
        confidence = float(probs[pred_class_id])
        
        # Get details from info database
        info = DISEASE_INFO.get(disease_name.strip(), {
            "description": "No clinical description available in local database. Please consult a doctor.",
            "precautions": ["Consult a medical practitioner", "Rest and monitor symptoms"]
        })
        
        # Format probabilities for all classes to show details
        class_probs = []
        for idx, prob in enumerate(probs):
            cls_name = str(disease_encoder.inverse_transform([idx])[0])
            if prob > 0.01: # Only include classes with > 1% confidence to reduce payload
                class_probs.append({"disease": cls_name, "probability": float(prob)})
                
        class_probs = sorted(class_probs, key=lambda x: x['probability'], reverse=True)[:5]
        
        return {
            "prediction": disease_name,
            "confidence": confidence,
            "description": info["description"],
            "precautions": info["precautions"],
            "top_probabilities": class_probs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/upload")
async def upload_csv_and_predict(file: UploadFile = File(...)):
    """Upload a CSV containing symptom rows and predict the disease for each row"""
    if model is None or disease_encoder is None or not symptom_to_int:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        contents = await file.read()
        df_input = pd.read_csv(io.BytesIO(contents))
        
        # Determine symptom columns
        # Accept either Symptom_1..17 or dynamically inspect columns
        cols = [c for c in df_input.columns if 'symptom' in c.lower()]
        if not cols:
            raise HTTPException(status_code=400, detail="CSV must contain at least one column with 'Symptom' in the header.")
        
        predictions = []
        confidences = []
        
        for idx, row in df_input.iterrows():
            row_symptoms = []
            for col in cols:
                val = str(row[col]).strip()
                if val and val.lower() not in ['nan', 'none', '']:
                    row_symptoms.append(val)
            
            # Use up to 5 symptoms per row and pad to the expected feature length
            padded = row_symptoms[:5]
            while len(padded) < 17:
                padded.append('None')
                
            encoded = [symptom_to_int.get(s, 0) for s in padded]
            
            pred_id = int(model.predict([encoded])[0])
            probs = model.predict_proba([encoded])[0]
            
            name = str(disease_encoder.inverse_transform([pred_id])[0])
            conf = float(probs[pred_id])
            
            predictions.append(name)
            confidences.append(conf)
            
        df_input['Predicted Disease'] = predictions
        df_input['Confidence Score'] = confidences
        
        # Return as downloadable CSV
        stream = io.StringIO()
        df_input.to_csv(stream, index=False)
        response = StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = "attachment; filename=predicted_disease_results.csv"
        return response
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@app.get("/results/metrics")
def get_metrics():
    """Get metrics and listings of generated plots"""
    metrics_file = "results/accuracy_results.txt"
    if not os.path.exists(metrics_file):
        raise HTTPException(status_code=404, detail="Metrics report not found. Run model training first.")
        
    with open(metrics_file, "r") as f:
        content = f.read()
        
    return {
        "text_report": content,
        "plots": {
            "confusion_matrix": "/results/image/confusion_matrix.png",
            "disease_distribution": "/results/image/disease_distribution.png",
            "symptom_frequency": "/results/image/symptom_frequency.png"
        }
    }

@app.get("/results/image/{image_name}")
def get_results_image(image_name: str):
    """Serve generated plots from the results directory"""
    image_path = os.path.join("results", image_name)
    if os.path.exists(image_path):
        return FileResponse(image_path)
    else:
        raise HTTPException(status_code=404, detail="Image not found")

def background_retrain():
    """Asynchronous background retraining task"""
    print("[RETRAIN] Starting background model training...")
    try:
        # Run disease_prediction.py as a separate process
        result = subprocess.run(["python", "disease_prediction.py"], capture_output=True, text=True)
        if result.returncode == 0:
            print("[RETRAIN] Model trained successfully.")
            load_models()
        else:
            print(f"[RETRAIN] Training script failed: {result.stderr}")
    except Exception as e:
        print(f"[RETRAIN] Error during background model training: {e}")

@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    """Retrain model in the background"""
    background_tasks.add_task(background_retrain)
    return {"status": "accepted", "message": "Model training initiated in the background."}
