# 📦 Truth Shield Fake News Detector - Package Overview

## 📋 Project Overview

This package contains a **complete, production-ready, and explainable Fake News Detection system**. It uses **Natural Language Processing (NLP)** and machine learning to distinguish authentic journalistic reports from fabricated articles. 

**Framework:** FastAPI + TF-IDF (scikit-learn) + XGBoost  
**Architecture:** Dual Model training (Patched Colab XGBoost + TF-IDF Explainable Linear Model) → REST API Backend → Interactive Web UI  
**Time to Launch:** ~5 minutes  

---

## 📂 All Files in This Package

### Core Application Files

| File | Size | Purpose | Status |
|------|------|---------|--------|
| `train_model.py` | 5.2 KB | High-performance TF-IDF NLP model trainer | ✓ Ready |
| `fake_news_detection.py` | 3.9 KB | User's reference XGBoost training notebook (patched & fully working) | ✓ Ready |
| `main.py` | 6.8 KB | FastAPI backend server with predictions & retraining | ✓ Ready |
| `fake_news_detection_frontend.html` | 53 KB | Premium glassmorphic web UI | ✓ Ready |
| `index.html` | 24 KB | Direct shortcut copy of web UI | ✓ Ready |
| `requirements.txt` | 180 B | Python dependencies | ✓ Ready |

### Documentation Files

| File | Size | Purpose |
|------|------|---------|
| `README.md` | 5 KB | Full developer documentation |
| `SETUP_GUIDE.md` | 6 KB | Step-by-step setup walkthrough |
| `PROJECT_COMPARISON.md` | 5 KB | Breakdown of clustering vs classification changes |
| `INDEX.md` | This file | Package index & quick reference |

---

## 🚀 Quick Launch (3 Commands)

### 1️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```

### 2️⃣ **Train the Model**
```bash
python train_model.py
```
This will read from the `fake news detection dataset/` folder and generate 4 joblib assets and a preview CSV.

### 3️⃣ **Start FastAPI server**
```bash
uvicorn main:app --reload
```

### 4️⃣ **Open Web UI**
Open `index.html` in your web browser. You're ready to scan news articles!

---

## 🎓 Learning Objectives

Through this project, you will explore:
1. **Natural Language Processing (NLP)** - Text representation using TF-IDF tokenization and bigram extraction.
2. **Explainable AI (XAI)** - Direct lookup of linear model coefficients to map word impact on predictions.
3. **Robust Backend API Serving** - Handling text payloads, CSV streaming file uploads, and background multitasking in FastAPI.
4. **Premium Frontend Engineering** - Building sleek, responsive user interfaces with CSS glassmorphism, glowing neons, animated SVG dials, and inline word tooltip injection.
