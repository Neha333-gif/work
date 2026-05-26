# Insurance Claim Prediction System

## 🚀 Predict Insurance Claim Approval with Lightweight ML

This folder contains a complete insurance claim prediction solution built with scikit-learn and FastAPI.

### What is included

- `insurance_claim_prediction_api.py` — FastAPI backend server
- `insurance_claim_prediction_trainer.py` — training pipeline and visualization generation
- `insurance_claim_prediction_frontend.html` — responsive web UI for single and batch prediction
- `insurance_claim_prediction.py` — insurance claim training script and example model generator
- `requirements.txt` — backend dependencies
- `results/` — generated visualization charts and evaluation artifacts

### Dataset

The dataset lives in `../insurance_claim_prediction_dataset/insurance_claims.csv` and includes fields such as:

- `age`
- `sex`
- `bmi`
- `children`
- `smoker`
- `region`
- `charges`
- `insuranceclaim`

### Run training

```bash
cd insurance_claim_prediction
python insurance_claim_prediction_trainer.py
```

### Start the backend

```bash
cd insurance_claim_prediction
uvicorn insurance_claim_prediction_api:app --reload
```

Then open:

```bash
http://127.0.0.1:8000/
```

### Available results

- `insurance_claim_prediction_evaluation.txt`
- `sample_insurance_claim_predictions.csv`
- `claim_class_distribution.png`
- `correlation_heatmap.png`
- `roc_curve.png`
- `confusion_matrix.png`
- `feature_importance.png`
- `precision_recall_curve.png`
- `claim_probability_distribution.png`
- `model_comparison.png`

### Notes

- The project uses lightweight models only: Logistic Regression, Decision Tree, and Random Forest.
- The frontend is fully connected to the backend and supports single-record predictions and batch CSV uploads.
- All terminology is now aligned with insurance claim prediction.

