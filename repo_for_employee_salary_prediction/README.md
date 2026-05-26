# Employee Salary Prediction

> AI-powered employee salary prediction using a Random Forest Regression pipeline. Predict employee compensation based on experience, role, education, location, and performance.

---

## Project Structure

```
repo_for_employee_salary_prediction/
│
├── employee_salary_prediction.py
├── employee_salary_model.py
├── employee_salary_prediction_frontend.html
│
├── employee_salary_prediction_data/
│   └── employee_salary_prediction_dataset.csv
│
├── results/
│   ├── salary_evaluation_report.txt
│   ├── salary_distribution.png
│   ├── correlation_heatmap.png
│   ├── experience_vs_salary.png
│   ├── actual_vs_predicted.png
│   ├── residual_distribution.png
│   ├── department_salary_analysis.png
│   ├── feature_importance.png
│   └── model_metrics.png
│
├── employee_salary_prediction_model.joblib
├── employee_salary_prediction_metadata.joblib
├── requirements.txt
└── README.md
```

---

## Dataset

**employee_salary_prediction_dataset.csv** — Synthetic salary target data generated from the MFG 10-Year Termination dataset. The pipeline uses employee features to create a salary regression problem for demonstration and evaluation.

### Key fields used
- `age`
- `length_of_service`
- `department_name`
- `job_title`
- `city_name`
- `BUSINESS_UNIT`
- `education_level`
- `performance_rating`
- `certifications`
- `skills_level`

---

## ML Pipeline

- **Model:** `RandomForestRegressor`
- **Features:** numeric + categorical encoders
- **Scaling:** `StandardScaler`
- **Metrics:** MAE, MSE, RMSE, R²
- **Visualizations:** salary distribution, correlation heatmap, actual vs predicted, residuals, department salary analysis, feature importance

---

## Quickstart

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train and start the server:
```bash
python employee_salary_prediction.py --train
```

3. Open the frontend:
Navigate to **http://127.0.0.1:8000**

---

## API Endpoints

| Method | Endpoint               | Description                               |
|--------|------------------------|-------------------------------------------|
| GET    | `/`                    | Serve the salary prediction frontend      |
| GET    | `/health`              | Check model and server health             |
| POST   | `/predict`             | Predict employee salary                   |
| GET    | `/results/metrics`     | Retrieve evaluation report and plot URLs  |
| GET    | `/results/image/{name}`| Serve generated visualization images      |
| POST   | `/retrain`             | Retrain the salary model in background    |

---

## Example `/predict` payload

```json
{
  "age": 35,
  "length_of_service": 5,
  "gender_short": "M",
  "city_name": "Vancouver",
  "department_name": "Sales",
  "job_title": "Sales Associate",
  "store_name": 1,
  "business_unit": "STORES",
  "education_level": "Bachelor",
  "performance_rating": 4,
  "certifications": 2,
  "skills_level": "Intermediate"
}
```

---

## Outputs

- `results/salary_evaluation_report.txt`
- `results/salary_distribution.png`
- `results/correlation_heatmap.png`
- `results/experience_vs_salary.png`
- `results/actual_vs_predicted.png`
- `results/residual_distribution.png`
- `results/department_salary_analysis.png`
- `results/feature_importance.png`
- `results/model_metrics.png`

---

## Notes

This project preserves the original ML pipeline structure while updating the domain to employee salary forecasting and adding salary-specific visualizations and a frontend experience.
