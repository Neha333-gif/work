# Employee Attrition Prediction

Unified ML pipeline and FastAPI backend for employee attrition prediction using workforce data.

## Dataset

- `C:\Users\peepl\Downloads\cust_segmentation_zip\repo_for_demand_forecasting\repo_for_demand_forecasting\employee_attrition\MFG10YearTerminationData.csv`

## Quick Start

```bash
pip install -r requirements.txt
python employee_attrition_prediction.py --train
python employee_attrition_prediction.py --port 8000
```

Open `http://127.0.0.1:8000/`.

## Models (Lightweight Only)

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

## Outputs

All artifacts are saved in `results/`:

- Evaluation reports (`accuracy_results.txt`, `evaluation_report.txt`)
- Predictions CSV (`employee_attrition_prediction_outputs.csv`)
- Metrics summary CSV (`model_metrics_summary.csv`)
- Visualizations:
  - `attrition_distribution.png`
  - `correlation_heatmap.png`
  - `feature_importance.png`
  - `confusion_matrix.png`
  - `roc_curve.png`
  - `department_wise_attrition_analysis.png`
  - `salary_vs_attrition_comparison.png`
  - `employee_satisfaction_trend.png`
  - `model_performance_metrics_visualization.png`

## API Endpoints

- `GET /` - frontend UI
- `GET /health` - health check
- `POST /predict` - attrition prediction
- `GET /results/metrics` - evaluation report
- `GET /results/image/{image_name}` - generated image files
- `POST /retrain` - retrain model in background
