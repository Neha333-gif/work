# 🚀 SETUP GUIDE | Truth Shield Fake News Detector

This step-by-step guide walks you through setting up, training, and running the Truth Shield Fake News Detector.

---

## 🛠️ Step-by-Step Walkthrough

### Step 1: Install Dependencies
Open your command terminal (Command Prompt, PowerShell, or Git Bash) inside the project directory and run:
```bash
pip install -r requirements.txt
```
This installs standard analytical frameworks (Pandas, Numpy, Scikit-learn), visual utilities (Matplotlib, Seaborn, Jinja2), the ML serving stack (FastAPI, Uvicorn, Python-multipart), and advanced classifiers (XGBoost, Imbalanced-learn).

---

### Step 2: Prepare Your Datasets
Ensure your files are located under the correct subfolder:
- `fake news detection dataset/fake.csv`
- `fake news detection dataset/true.csv`

The datasets should contain `title` and `text` columns as their primary text features.

---

### Step 3: Train the Models
Train both models in your project directory:

1. **High-Performance Explainable NLP Model**:
   ```bash
   python train_model.py
   ```
   *Expected Console Output:*
   ```
   ============================================================
   STARTING FAKE NEWS DETECTION NLP MODEL TRAINING
   ============================================================
   Loading Fake News dataset...
   Loading True News dataset...
   Fake news count: 23481, True news count: 21417
   Preprocessing text data...
   Train set size: 35918, Test set size: 8980
   Vectorizing text using TF-IDF (unigrams & bigrams, max 5000 features)...
   Training Logistic Regression Classifier...
   Evaluating model performance...
   Accuracy:  0.9859
   Precision: 0.9840
   Recall:    0.9863
   F1 Score:  0.9852
   [OK] Assets saved successfully.
   ```
   This generates the vectorizer and models needed by the API server.

2. **Original patched Colab XGBoost Model** (Optional):
   ```bash
   python fake_news_detection.py
   ```
   This will train the XGBoost classifier on label-encoded variables (utilizing SMOTE to resolve class imbalance) and save it as `fake news detection.pkl`.

---

### Step 4: Start the FastAPI Backend
Start the server using Uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
*Expected Console Output:*
```
[INFO] Model assets loaded successfully.
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Verify your API is online by visiting: **http://localhost:8000/health**

---

### Step 5: Open the Web User Interface
Double-click `index.html` or `fake_news_detection_frontend.html` in your file explorer to open the UI directly in your browser.

If you are using a local web server (recommended to avoid CORS issues on file uploads):
```bash
python -m http.server 8080
```
Then open: **http://localhost:8080/index.html** or **http://localhost:8080/fake_news_detection_frontend.html**

---

## 🐛 Troubleshooting

### Issue: "Failed to load model assets"
* **Symptom:** Server starts but endpoints throw a 503 error, or console logs warn about missing joblib files.
* **Solution:** Make sure you ran `python train_model.py` inside the project root folder. Verify that `tfidf_vectorizer.joblib`, `logistic_regression_model.joblib`, and `word_coefficients.joblib` have appeared in your project directory.

### Issue: "CORS blocking batch file upload"
* **Symptom:** The single scan works, but drag-and-drop CSV upload throws a console network error.
* **Solution:** Running the HTML directly as a local file (`file:///...`) sometimes restricts browser fetch operations on form data. Serve your HTML from a local HTTP server:
  ```bash
  python -m http.server 8080
  ```

### Issue: "CSV upload fails with 400 Bad Request"
* **Symptom:** File starts uploading but fails, returning "CSV must contain title and text columns."
* **Solution:** Open your CSV in Excel or Notepad and verify that the column headers are exactly named `title` and `text` (case-sensitive, lowercase). If your columns are named differently, click "Download CSV Template" in the batch tab for reference.
