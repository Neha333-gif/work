# 📦 Ad Click Prediction System - Package Overview

## 📋 Project Overview

This package contains a **complete, production-grade, and explainable Ad Click Prediction system**. It uses machine learning to predict user ad engagement classes (0 to 4 clicks) based on historical campaign banner interaction metrics.

**Framework:** FastAPI + Random Forest Classifier (scikit-learn) + SMOTE Oversampling  
**Architecture:** End-to-End ML Pipeline (Ingestion → Cleaning → SMOTE → RandomForest Training) → REST API Backend → Glassmorphic Dashboard UI  
**Deployment Ready:** Supports Dockerization & CI/CD workflow testing.

---

## 📂 All Files in This Package

### Core Application Files

| File | Size | Purpose | Status |
|------|------|---------|--------|
| [ad_click_predictions.py](file:///c:/Users/peepl/Desktop/New%20folder%20%282%29/repo_for_ad_click/ad_click_predictions.py) | ~11.5 KB | Model training pipeline with SMOTE & visual plot generators | ✓ Ready |
| [ad_click_prediction_backend.py](file:///c:/Users/peepl/Desktop/New%20folder%20%282%29/repo_for_ad_click/ad_click_prediction/ad_click_prediction_backend.py) | ~16.0 KB | FastAPI backend server with single & batch prediction + retraining endpoints | ✓ Ready |
| [ad_click_prediction_frontend.html](file:///c:/Users/peepl/Desktop/New%20folder%20%282%29/repo_for_ad_click/ad_click_prediction/ad_click_prediction_frontend.html) | ~45.9 KB | Glassmorphic dark-mode web user interface and metrics reporter | ✓ Ready |
| [Dockerfile](file:///c:/Users/peepl/Desktop/New%20folder%20%282%29/repo_for_ad_click/Dockerfile) | ~0.7 KB | Docker configuration for containerized deployment | ✓ Ready |
| [.github/workflows/ml_pipeline.yml](file:///c:/Users/peepl/Desktop/New%20folder%20%282%29/repo_for_ad_click/.github/workflows/ml_pipeline.yml) | ~1.5 KB | GitHub Actions CI/CD code verification pipeline | ✓ Ready |
| [requirements.txt](file:///c:/Users/peepl/Desktop/New%20folder%20%282%29/repo_for_ad_click/ad_click_prediction/requirements.txt) | ~0.2 KB | Python dependency list | ✓ Ready |

### Documentation Files

| File | Purpose |
|------|---------|
| [README.md](file:///c:/Users/peepl/Desktop/New%20folder%20%282%29/repo_for_ad_click/ad_click_prediction/README.md) | Central document with 30 End-to-End ML lifecycle steps mapping |
| [SETUP_GUIDE.md](file:///c:/Users/peepl/Desktop/New%20folder%20%282%29/repo_for_ad_click/ad_click_prediction/SETUP_GUIDE.md) | Step-by-step developer setup & troubleshooting guide |
| [PROJECT_COMPARISON.md](file:///c:/Users/peepl/Desktop/New%20folder%20%282%29/repo_for_ad_click/ad_click_prediction/PROJECT_COMPARISON.md) | Comparison of the refactored workflow changes |
| `INDEX.md` (This file) | Package index & quick file reference |

---

## 🚀 Quick Launch (3 Commands)

### 1️⃣ **Install dependencies**
```bash
pip install -r ad_click_prediction/requirements.txt
```

### 2️⃣ **Train the Model & Generate Plots**
```bash
python ad_click_predictions.py
```
This reads `banner_interactions.csv`, trains the Random Forest model, and writes evaluation reports & plots to `ad_click_prediction/results/`.

### 3️⃣ **Start FastAPI Backend Server**
```bash
uvicorn ad_click_prediction.ad_click_prediction_backend:app --reload
```

### 4️⃣ **Open Web UI Dashboard**
Open `ad_click_prediction/ad_click_prediction_frontend.html` in your web browser. You're ready to run predictions!

---

## 🎓 Key Machine Learning Concepts Explored

1. **Class Balancing (SMOTE)**: Resolves highly skewed target distributions by synthetically generating samples for minority classes (k_neighbors=4).
2. **Lightweight Modeling**: Random Forest classifier configured with optimized hyperparameter bounds to balance prediction speed and accuracy.
3. **Data Preprocessing Integrity**: Strict separation of categorical encoding and numerical feature standard scaling.
4. **Asynchronous Retraining serving**: Background worker capabilities inside FastAPI to update live model assets without restarting the server.
5. **Modern Dashboard Design**: Sleek dark-mode interface showcasing real-time metric visuals and campaign insights.
