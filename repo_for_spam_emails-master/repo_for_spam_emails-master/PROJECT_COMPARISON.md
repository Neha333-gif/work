# 📊 PROJECT COMPARISON | Retail Clustering vs News NLP Classification

This document maps the architectural differences between your previous **Customer Segmentation** project and the newly developed **Truth Shield Fake News Detector** project.

---

## 🔍 Structural & Architectural Breakdown

| Feature | Legacy: Customer Segmentation | New: Fake News Detector |
|---------|------------------------------|-------------------------|
| **Core ML Paradigm** | Unsupervised Learning (Clustering) | Supervised Learning (Binary Classification) |
| **Primary Algorithm** | K-Means Clustering | TF-IDF + Logistic Regression (NLP Pipeline) |
| **Input Data Format** | Structured Demographics (Age, Income, Spending score, Married, etc.) | Unstructured News Articles (Title text, Body content string) |
| **Preprocessing Layer** | Imputation, Label Encoding, StandardScaler | Text tokenization, Stopword removal, Unigrams/Bigrams TF-IDF vectorization |
| **Model Output** | Cluster ID (0, 1, 2, 3) representing segments | Binary label: `0` (Fake) or `1` (True) with confidence probability |
| **Explainable AI (XAI)** | Cluster center distance and average KPI bounds | Word-level coefficient impact highlights mapped directly onto article text |
| **Batch Operations** | Classifies structured user rows in tabular format | Bulk scans lists of unstructured text articles from a CSV file, appending predictions |

---

## ⚙️ Preprocessing & NLP Upgrades

In the customer segmentation project, numerical values were scaled and categorical terms label-encoded to prepare coordinate indices for K-Means distance calculations.

For unstructured news text data, simple label encoding (as used in the original `fake_news_detection.py`) is suboptimal because the model memorizes exact string coordinates instead of analyzing vocabulary context. To address this, the **Truth Shield NLP pipeline** implements a **TF-IDF Vectorizer** (Term Frequency-Inverse Document Frequency) which:
1. Splits title and text blocks into structural unigrams and bigrams (e.g. "breaking news", "said spokesman").
2. Penalizes highly frequent words (using standard English stopword lists).
3. Normalizes word frequencies based on how often they appear across all documents.
4. Passes normalized vectors into a linear classifier, where every vocabulary term receives an impact coefficient representing its correlation with truth or fabrication.

---

## 📱 Backend & API Endpoint Refactoring

Your API endpoints have been redesigned for this NLP use-case:

```
FastAPI Server (Refactored)
├── GET  /health              → Health status check
├── GET  /info                → Returns validation scores & top 50 vocab indicator words
├── POST /predict             → Scans title + text, returning prediction & highlighted word indices
├── POST /predict/upload      → Streams CSV file, appends prediction columns, returns CSV download
└── POST /retrain             → Asynchronously retrains the TF-IDF classifier on local CSVs
```

---

## 🎨 User Interface Enhancements

The interface has been redesigned for an optimal user experience:
1. **Dynamic Authenticity Gauge**: The segment KPIs have been replaced by a glowing radial SVG speedometer reflecting the exact percentage probability of an article's truthfulness.
2. **Interactive Explainability Panel**: An explainability window overlays highlighted spans onto the user's article text. Hovering over suspicious terms reveals tooltips with fakeness metrics.
3. **Bulk Processing Pipeline**: Standard tabular grid upload is updated to support large news CSV files, including custom column parsing, progress indicator bars, and batch export files.
4. **Vocabulary Analytics Barcharts**: The static segment badge index is replaced by a live comparison of the top 20 strongest vocabulary signals used by the classifier.
