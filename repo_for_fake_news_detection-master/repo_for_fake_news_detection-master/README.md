# 🛡️ TRUTH SHIELD | AI Fake News Detector

A complete, production-ready Natural Language Processing (NLP) machine learning pipeline for detecting Fake News articles. The system features a **FastAPI backend REST API** with a high-accuracy TF-IDF model, **explainable AI (XAI)** to spotlight suspicious and verified words, and a premium **glassmorphic dark-mode web user interface**.

---

## 📦 Project Structure

```
fake_news_detection/
├── fake news detection dataset/
│   ├── fake.csv                              # Fake news dataset (~23.4k rows)
│   └── true.csv                              # True news dataset (~21.4k rows)
├── train_model.py                             # High-performance NLP TF-IDF trainer
├── fake_news_detection.py                     # User's reference XGBoost training script (patched)
├── main.py                                    # FastAPI REST API serving the NLP model
├── fake_news_detection_frontend.html          # Frosted glassmorphic Web UI
├── index.html                                 # Direct copy of frontend
├── requirements.txt                           # Python dependencies
├── README.md                                  # Complete reference guide (this file)
├── INDEX.md                                   # Project Overview & Quick Start
├── SETUP_GUIDE.md                             # Step-by-step setup walkthrough
└── PROJECT_COMPARISON.md                      # Comparison: retail clustering vs news classification
```

---

## 🚀 Quick Start (5 Steps)

### 1️⃣ Install Dependencies
Ensure you have the required packages installed:
```bash
pip install -r requirements.txt
```

### 2️⃣ Train the NLP Model
Run the high-performance training script:
```bash
python train_model.py
```
This trains a TF-IDF + Logistic Regression classification pipeline on combined news titles and texts. It achieves **~98.6% validation accuracy** and generates:
- `tfidf_vectorizer.joblib`: TF-IDF vectorizer (5000 features).
- `logistic_regression_model.joblib`: Linear classifier for fast and explainable predictions.
- `word_coefficients.joblib`: Mapping of words to fakeness weights for explanation.
- `model_metadata.joblib`: Performance metrics and top vocab indicators.
- `sample_news_predictions.csv`: Sample predictions for previews.

*(Optional)* Run the original patched script:
```bash
python fake_news_detection.py
```
This trains an XGBoost model on label-encoded features and creates `fake news detection.pkl`.

### 3️⃣ Launch the FastAPI Server
Start the backend API using uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
The server will start at: **http://localhost:8000**
- Interactive Swagger UI: `http://localhost:8000/docs`

### 4️⃣ Launch the Web Interface
Simply open `index.html` or `fake_news_detection_frontend.html` directly in your browser, or serve it via python:
```bash
python -m http.server 8080
```
Then visit: **http://localhost:8080/index.html** or **http://localhost:8080/fake_news_detection_frontend.html**

---

## 🎨 Web Interface Features

### 🔍 Tab 1: Single Article Scanner
- **Direct Input**: Enter news title and full text.
- **Authenticity Dial**: Beautiful circular SVG progress gauge displaying predicted authenticity percentage, transitioning from glowing neon red (Fake) to glowing neon emerald (True).
- **Explainable AI (XAI) Spotlight**: The scanned text is re-rendered with key words highlighted. Hovering over highlighted terms displays tooltips with their specific impact score (negative = suspicious/fake, positive = verified/true).
- **Keyword Analytics**: Side-by-side bar chart lists showing top suspect and verified terms parsed from the text.

### 📁 Tab 2: Batch CSV Scanner
- **Drag & Drop Uploader**: Drag and drop large CSV datasets of articles to process them in bulk.
- **CSV Template**: Built-in button to download a standard CSV template containing `title` and `text` columns.
- **Interactive Results**: Review rows on-screen with color-coded badges and confidence scores.
- **Bulk Export**: Export processed results containing prediction and confidence scores in one click.

### 📊 Tab 3: Model Insights & Training Hub
- **Diagnostics Dashboard**: View global metrics (Accuracy, Precision, F1 Score) and a dynamically rendered confusion matrix.
- **Vocabulary Signals**: Side-by-side bar chart of the top 10 overall fake indicators and true indicators in the model's vocabulary.
- **Live Retraining Console**: Trigger background model retraining with an on-screen console showing live poll messages.

---

## 🔌 API Endpoints Reference

### 🟢 Server Status
`GET /health`
- Checks model load status and outputs backend health.

### 📊 Global Info & Metrics
`GET /info`
- Returns vocabulary statistics, validation metrics, and top indicator words.

### 🔍 Single Article Prediction
`POST /predict`
- **Request Body:**
  ```json
  {
    "title": "Shocking discovery in the city",
    "text": "Breaking news reports indicate an alien ship landed in downtown."
  }
  ```
- **Response:** Predictions, confidence probabilities, and lists of highlighted words.

### 📁 Batch JSON Predictions
`POST /predict/batch`
- Returns a list of predictions for multiple articles.

### 📥 Bulk CSV Upload
`POST /predict/upload`
- Accepts a `.csv` file upload, appends prediction columns, and returns a downloadable CSV stream.

### 🔄 Asynchronous Retraining
`POST /retrain`
- Initiates training of the TF-IDF model as an async task, updating the server assets on the fly.
