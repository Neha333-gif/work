# 🎯 AI Ad Click Prediction System

A complete, production-grade, end-to-end Machine Learning pipeline and interactive serving dashboard for predicting user ad click engagement. The project implements a lightweight **Random Forest Classifier** trained on banner interactions, served via a **FastAPI backend REST API**, and visualized in a premium **glassmorphic dark-mode web user interface**.

---

## 🗺️ End-to-End ML Flow Mapping (30 Steps Covered)

This project has been structured to cover the key phases of a production machine learning lifecycle:

### Phase 1: Foundation & Data
1. **Business Understanding**: Predict user click engagement classes (0 to 4 clicks) based on campaign factors (banner, date, impressions, CTR) to optimize marketing spend, banner designs, and target bidding.
2. **Requirement Gathering**: Define specifications for dynamic web interaction, low-latency scoring, asynchronous retraining capability, and lightweight compute footprints (RF instead of XGBoost).
3. **Data Collection**: Gather historical banner performance logs containing dates, impressions, CTR, banner IDs, and click outputs.
4. **Data Ingestion**: Standardized CSV loading with defensive path handling implemented in [ad_click_predictions.py](file:///c:/Users/peepl/Desktop/New%20folder%20%282%29/repo_for_ad_click/ad_click_predictions.py).
5. **Data Storage**: Local analytical repository containing raw files (`banner_interactions.csv`).
6. **Data Cleaning**: Handle missing, corrupted, or null data entries automatically via `SimpleImputer` using the `most_frequent` imputation strategy.
7. **Data Preprocessing**: Type-safe feature pipeline that prevents data-coercion bugs, correctly standard-scaling continuous variables (`ctr`, `impressions`) and label-encoding discrete fields (`banner_id`, `event_date`).
8. **Exploratory Data Analysis (EDA)**: Automatic visual profiling of class imbalances, correlation metrics, and feature profiles.

### Phase 2: Feature Engineering & Modeling
9. **Feature Engineering**: Standard scale transformations and date-category mapping to convert text dates and IDs into dense mathematical coordinates.
10. **Feature Selection**: Drop high-cardinality metadata identifiers like `user_id` which cause overfitting, retaining only prediction-driving indicators.
11. **Train-Test Split**: Implement a deterministic 80/20 data partition to establish clear training controls and validation parameters.
12. **Model Selection**: Select a lightweight `RandomForestClassifier` to replace heavy, slow classifiers (XGBoost), ensuring fast inference times and instant background training.
13. **Model Training**: Fit decision forest trees on SMOTE-oversampled training sets to resolve severe click distribution skew.
14. **Hyperparameter Tuning**: Optimize model structure by constraining tree depth (`max_depth=8`) and quantity (`n_estimators=50`) to optimize memory footprint and model size (~1MB).
15. **Model Evaluation**: Generate precise metrics reports listing Accuracy, Precision, Recall, and F1-Scores per click category.
16. **Model Validation**: Out-of-sample testing validating generalization capabilities with a Confusion Matrix and a multiclass Receiver Operating Characteristic (ROC) curve.
17. **Model Interpretability**: Output a Feature Importance plot detailing which features (e.g. CTR vs Impressions) dictate the model's classifications.
18. **Model Saving**: Dump binary artifacts (`ad_click_model.joblib`, `ad_click_preprocessors.joblib`) containing fit state parameters.

### Phase 3: Serving & Operations
19. **API Development**: Develop a FastAPI REST API inside [ad_click_prediction_backend.py](file:///c:/Users/peepl/Desktop/New%20folder%20%282%29/repo_for_ad_click/ad_click_prediction/ad_click_prediction_backend.py) to manage predictions, file uploads, metrics, and system health.
20. **Model Deployment**: Expose the backend using Uvicorn web server gateways for low-latency JSON serving.
21. **Dockerization**: Containerize the app using a [Dockerfile](file:///c:/Users/peepl/Desktop/New%20folder%20%282%29/repo_for_ad_click/Dockerfile) for uniform deployment environments.
22. **CI/CD Pipeline Setup**: Configure a GitHub Actions workflow in [.github/workflows/ml_pipeline.yml](file:///c:/Users/peepl/Desktop/New%20folder%20%282%29/repo_for_ad_click/.github/workflows/ml_pipeline.yml) to validate coding style, verify dataset integrity, run test training runs, and test Docker image builds automatically on push.
23. **Cloud Deployment**: Supported by the standard container structure (deployable to Google Cloud Run, AWS ECS, or Azure Container Apps).
24. **Monitoring and Logging**: Integrated health checks (`GET /health`) and console request logs to monitor API traffic and load status.
25. **Model Retraining**: Trigger asynchronous training loops dynamically via the `POST /retrain` endpoint, allowing model updates on hot-standby.

### Phase 4: Business Integration
26. **Security and Governance**: Include Cross-Origin Resource Sharing (CORS) security setups and explicit Pydantic schema request validation.
27. **Dashboard and Reporting**: Build a premium glassmorphic user dashboard in [ad_click_prediction_frontend.html](file:///c:/Users/peepl/Desktop/New%20folder%20%282%29/repo_for_ad_click/ad_click_prediction/ad_click_prediction_frontend.html) displaying model diagnostics, single-prediction dials, batch CSV upload utilities, and training controls.
28. **Performance Optimization**: Use pandas DataFrames for fast vector mapping, multiclass binarization, and scikit-learn parallel execution (`n_jobs=-1`).
29. **Production Support**: Implement graceful fallbacks for missing assets, background task handles, and reload triggers.
30. **Final Business Insights**: Map predicted click values (0 to 4) directly to campaign suggestions (e.g. Budget reallocation, audience targeting adjustments, fraudulent traffic checks) via the backend insight engine.

---

## 📦 Project Structure

```
repo_for_ad_click/
├── .github/workflows/
│   └── ml_pipeline.yml                 # CI/CD validation workflow
├── ad_click_prediction/
│   ├── results/
│   │   ├── accuracy_results.txt        # Validation performance metrics report
│   │   ├── class_distribution.png      # Class counts visualization
│   │   ├── confusion_matrix.png        # Predictions vs Actual matrix
│   │   ├── correlation_heatmap.png     # Feature correlation coefficients
│   │   ├── feature_importance.png      # Relative feature predictive strength
│   │   └── roc_curve.png               # Multiclass ROC curves
│   ├── ad_click_prediction_backend.py  # FastAPI REST API serving endpoints
│   ├── ad_click_prediction_frontend.html # Dark-mode glassmorphic user interface
│   ├── requirements.txt                # Python dependencies
│   ├── INDEX.md                        # Package index & overview
│   ├── PROJECT_COMPARISON.md           # Breakdown: Legacy vs Ad Click pipeline
│   ├── README.md                       # Complete documentation (this file)
│   └── SETUP_GUIDE.md                  # Installation and startup guide
├── Dockerfile                          # Docker containerization config
├── ad_click_predictions.py             # Random Forest model training pipeline
├── banner_interactions.csv             # Raw ad click campaign dataset (~500k rows)
├── ad_click_model.joblib               # Trained Random Forest binary model
└── ad_click_preprocessors.joblib       # Fitted scalers, encoders, and date metadata
```

---

## 🚀 Quick Start (4 Steps)

### 1️⃣ Install Dependencies
Ensure you have the required packages installed:
```bash
pip install -r ad_click_prediction/requirements.txt
```

### 2️⃣ Train the Model & Generate Visualizations
Execute the pipeline script to train the model and output diagnostic plots:
```bash
python ad_click_predictions.py
```
This script splits the data, balances target classes using SMOTE, trains a Random Forest Classifier, and writes the accuracy report and all 5 plots into the `ad_click_prediction/results/` folder.

### 3️⃣ Launch the FastAPI Server
Start the backend server on port 8000:
```bash
uvicorn ad_click_prediction.ad_click_prediction_backend:app --reload --host 127.0.0.1 --port 8000
```
- Open Swagger UI Docs at: **http://127.0.0.1:8000/docs**
- Health Status Page: **http://127.0.0.1:8000/health**

### 4️⃣ Open the Dashboard Web Page
Double-click `ad_click_prediction/ad_click_prediction_frontend.html` in your file explorer, or serve it locally via python:
```bash
python -m http.server 8080
```
Then visit: **http://localhost:8080/ad_click_prediction/ad_click_prediction_frontend.html**

---

## 🔌 API Endpoints Reference

### `GET /health`
Verifies server health and checks whether `ad_click_model.joblib` and `ad_click_preprocessors.joblib` are loaded.

### `GET /features`
Returns dynamic features needed by the UI: lists of unique Banner IDs and the minimum/maximum campaign date ranges.

### `POST /predict`
Scans a single ad interaction payload and predicts the expected clicks class, confidence, and campaign business insights.
- **Request Format:**
  ```json
  {
    "event_date": "2026-01-01",
    "banner_id": "b_0082",
    "impressions": 3,
    "ctr": 0.05
  }
  ```

### `POST /predict/upload`
Accepts a `.csv` file upload containing columns `['event_date', 'banner_id', 'impressions', 'ctr']`, performs batch predictions, and returns a downloadable CSV with predicted click counts and confidence scores appended.

### `GET /results/metrics`
Reads the `accuracy_results.txt` text report and returns URL paths for the five generated diagnostic plots.

### `GET /results/image/{image_name}`
Serves a specific plot file (e.g. `confusion_matrix.png`) from the `results/` directory.

### `POST /retrain`
Asynchronously triggers model retraining on the server. Executes the training script in the background and hot-reloads the newly generated weights.
