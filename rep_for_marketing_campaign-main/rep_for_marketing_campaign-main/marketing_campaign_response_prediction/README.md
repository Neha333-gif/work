# Marketing Campaign Response Prediction System

## 🚀 Predict Marketing Campaign Responses with Lightweight ML

This folder contains a complete marketing campaign response prediction solution built with scikit-learn and FastAPI.

### What is included

- `marketing_campaign_response_prediction_api.py` — FastAPI backend server
- `marketing_campaign_response_prediction_trainer.py` — training pipeline and visualization generation
- `marketing_campaign_response_prediction_frontend.html` — responsive web UI for single and batch prediction
- `requirements.txt` — backend dependencies
- `results/` — generated visualization charts and evaluation artifacts

### Dataset

The dataset lives in `../marketing_campaign_response_dataset/campaign_responses.csv` and includes fields such as:

- `age`
- `gender`
- `annual_income`
- `credit_score`
- `employed`
- `marital_status`
- `no_of_children`
- `responded`

### Run training

```bash
cd marketing_campaign_response_prediction
python marketing_campaign_response_prediction_trainer.py
```

### Start the backend

```bash
cd marketing_campaign_response_prediction
uvicorn marketing_campaign_response_prediction_api:app --reload
```

Then open:

```bash
http://127.0.0.1:8000/
```

### Available results

- `marketing_campaign_response_prediction_evaluation.txt`
- `sample_marketing_campaign_response_predictions.csv`
- `class_distribution.png`
- `correlation_heatmap.png`
- `roc_curve.png`
- `confusion_matrix.png`
- `feature_importance.png`
- `precision_recall_curve.png`
- `prediction_distribution.png`
- `model_comparison.png`

### Notes

- The project uses lightweight models only: Logistic Regression, Decision Tree, and Random Forest.
- The frontend is fully connected to the backend and supports single-record predictions and batch CSV uploads.
- All terminology is now aligned with marketing campaign response prediction.

