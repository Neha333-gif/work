# Customer Churn Prediction System

A production-ready AI-powered customer churn prediction system built with Random Forest classification. Predict customer churn risk and get personalized retention strategies.

## 📁 Project Structure

```
repo_for_customer_churn/
├── customer_churn.py                 # ML training pipeline
├── customer_churn_model.joblib       # Trained Random Forest model
├── customer_churn_preprocessors.joblib  # Feature scaler
├── customer_churn_features.joblib    # Feature names list
├── test_api.py                       # API testing script
├── Dockerfile                        # Docker configuration
├── README.md                         # This file
└── customer_churn_prediction/
    ├── customer_churn_prediction_backend.py    # FastAPI server
    ├── customer_churn_prediction_frontend.html # Web UI
    ├── requirements.txt              # Python dependencies
    ├── README.md                     # Detailed documentation
    └── results/
        ├── accuracy_results.txt      # Performance metrics
        ├── confusion_matrix.png
        ├── feature_importance.png
        ├── class_distribution.png
        ├── roc_curve.png
        ├── precision_recall_curve.png
        └── correlation_heatmap.png
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd customer_churn_prediction
pip install -r requirements.txt
```

### 2. Train the Model (if needed)
```bash
python ../customer_churn.py
```

### 3. Start the Backend Server
```bash
python -m uvicorn customer_churn_prediction_backend:app --host 0.0.0.0 --port 8000
```

### 4. Access the Frontend
Open your browser and navigate to: **http://localhost:8000**

## 📊 API Endpoints

### GET /health
Check API status and model health
```bash
curl http://localhost:8000/health
```

### POST /predict
Predict churn for a single customer
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "senior_citizen": 0,
    "tenure": 24,
    "monthly_charges": 65.5,
    "total_charges": 1570.0,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check"
  }'
```

### POST /predict/batch
Batch predict from CSV file
```bash
curl -X POST http://localhost:8000/predict/batch \
  -F "file=@customers.csv"
```

### GET /results/metrics
Get model performance metrics and visualization links
```bash
curl http://localhost:8000/results/metrics
```

## 📈 Model Performance

- **Accuracy**: 77.44%
- **Precision**: 56.14%
- **Recall**: 66.76%
- **F1-Score**: 60.99%
- **ROC-AUC**: 0.8321

## 🧠 Model Details

- **Algorithm**: Random Forest Classifier
- **Trees**: 300
- **Max Depth**: 10
- **Features**: 30 (preprocessed customer attributes)
- **Training Data**: 5,602 samples
- **Test Data**: 1,401 samples
- **Class Balancing**: SMOTE oversampling

## 🎯 Features

### Customer Input Fields
- Gender (Male/Female)
- Senior Citizen Status (Yes/No)
- Tenure (months of service)
- Monthly Charges (monthly bill)
- Total Charges (lifetime charges)
- Contract Type (Month-to-month / 1 year / 2 year)
- Internet Service (DSL / Fiber optic / None)
- Payment Method (Electronic check / Mailed check / Bank transfer / Credit card)

### Prediction Output
- **Churn Probability**: Likelihood customer will churn (0-100%)
- **Retention Probability**: Likelihood customer will stay (0-100%)
- **Risk Level**: Low / Medium / High churn risk
- **Risk Description**: Contextual insights
- **Recommended Actions**: 5 specific retention strategies
- **Model Confidence**: Overall prediction confidence

## 📝 Dataset

**Telco Customer Churn Dataset**
- Total Records: 7,043 customers
- Features: 21 customer attributes
- Target: Churn (Yes/No)
- Class Distribution: 73.5% No Churn, 26.5% Churn

## 🐳 Docker Support

Build and run with Docker:
```bash
docker build -t customer-churn-predictor .
docker run -p 8000:8000 customer-churn-predictor
```

## 📦 Requirements

- Python 3.8+
- FastAPI 0.104.1
- Uvicorn 0.24.0
- pandas 2.0.3
- scikit-learn 1.3.0
- joblib 1.3.2
- matplotlib 3.7.2
- seaborn 0.12.2
- imbalanced-learn 0.11.0

See `customer_churn_prediction/requirements.txt` for full list

## 🔍 Testing

Run API tests:
```bash
python test_api.py
```

## 📧 Support

For issues or questions, check the backend logs or review the model metrics in `customer_churn_prediction/results/accuracy_results.txt`

---

**Project Status**: ✅ Production Ready
**Last Updated**: May 26, 2026
