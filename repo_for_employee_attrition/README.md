# Employee Attrition Prediction

> **AI-powered workforce attrition prediction** using a Random Forest Classifier trained on the MFG 10-Year Termination Dataset. Predicts whether an employee will remain **Active** or be **Terminated**, with real-time inference served via a FastAPI backend and a premium dark-mode frontend UI.

---

## Project Structure

```
repo_for_employee_attrition_prediction/
│
├── employee_attrition_prediction.py         # FastAPI backend + ML training pipeline
├── employee_attrition_model.py              # Standalone ML training script
├── employee_attrition_prediction_frontend.html  # Premium dark-mode UI
│
├── employee_attrition_prediction_data/
│   └── MFG10YearTerminationData.csv         # Employee workforce dataset
│
├── results/
│   ├── accuracy_results.txt                 # Model evaluation report
│   ├── attrition_distribution.png           # Class distribution
│   ├── correlation_heatmap.png              # Feature correlation
│   ├── roc_curve.png                        # ROC curve
│   ├── confusion_matrix.png                 # Confusion matrix
│   ├── feature_importance.png               # Feature importance
│   ├── precision_recall_curve.png           # Precision-recall
│   ├── retention_vs_attrition.png           # Dept-level analysis
│   └── model_metrics.png                    # Performance summary
│
├── employee_attrition_prediction_model.joblib    # Saved model
├── employee_attrition_prediction_metadata.joblib # Saved encoders/scaler
├── requirements.txt
└── README.md
```

---

## Dataset

**MFG10YearTerminationData.csv** — A 10-year manufacturing workforce dataset with 49,653 records and 18 features including:

| Feature            | Description                        |
|--------------------|------------------------------------|
| `age`              | Employee age                       |
| `length_of_service`| Years at the company               |
| `department_name`  | Department                         |
| `job_title`        | Job role / title                   |
| `city_name`        | Office city                        |
| `store_name`       | Store identifier                   |
| `gender_short`     | Gender (M/F)                       |
| `BUSINESS_UNIT`    | Business unit (STORES/HEADOFFICE)  |
| `STATUS`           | Target: ACTIVE or TERMINATED       |

> **Note:** Features `terminationdate_key`, `termreason_desc`, and `termtype_desc` are dropped to prevent label leakage.

---

## ML Pipeline

- **Model:** `RandomForestClassifier` (n_estimators=150, max_depth=10, class_weight='balanced')
- **Balancing:** SMOTE oversampling on the minority (TERMINATED) class
- **Encoding:** LabelEncoder for all categorical features
- **Scaling:** StandardScaler
- **Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model & start the server
```bash
python employee_attrition_prediction.py --train
```

### 3. Open the frontend
Navigate to: **http://127.0.0.1:8000**

---

## API Endpoints

| Method | Endpoint               | Description                            |
|--------|------------------------|----------------------------------------|
| GET    | `/`                    | Serve the frontend UI                  |
| GET    | `/health`              | API health check + model status        |
| POST   | `/predict`             | Predict attrition for an employee      |
| GET    | `/results/metrics`     | Model evaluation report + plot paths   |
| GET    | `/results/image/{name}`| Serve a generated visualization        |
| POST   | `/retrain`             | Retrain model in background            |

### Example `/predict` payload:
```json
{
  "age": 35,
  "length_of_service": 5,
  "gender_short": "M",
  "city_name": "Vancouver",
  "department_name": "Sales",
  "job_title": "Sales Associate",
  "store_name": 1,
  "business_unit": "STORES"
}
```

### Example response:
```json
{
  "predicted_status": "ACTIVE",
  "retention_probability": 87.4,
  "attrition_probability": 12.6,
  "risk_level": "Low",
  "model_accuracy": 0.9123,
  "model_roc_auc": 0.9541
}
```

---

## Visualizations Generated

| Chart                    | Filename                        |
|--------------------------|---------------------------------|
| Class Distribution       | `attrition_distribution.png`   |
| Correlation Heatmap      | `correlation_heatmap.png`      |
| ROC Curve                | `roc_curve.png`                |
| Confusion Matrix         | `confusion_matrix.png`         |
| Feature Importance       | `feature_importance.png`       |
| Precision-Recall Curve   | `precision_recall_curve.png`   |
| Retention vs Attrition   | `retention_vs_attrition.png`   |
| Model Metrics Summary    | `model_metrics.png`            |

---

## License

MIT License — open source, free to use and modify.
