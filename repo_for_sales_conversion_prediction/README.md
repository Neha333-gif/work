# Sales Conversion Prediction

Unified ML pipeline and FastAPI backend for predicting customer sales conversion from lead data.

## Dataset

`sales conversion/KAG_conversion_data.csv`

## Quick Start

```bash
pip install -r requirements.txt
python sales_conversion_prediction.py --train
python sales_conversion_prediction.py --port 8000
```

Open `http://127.0.0.1:8000/` for the frontend UI.

## Models

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

The best model (highest F1 score) is saved automatically.

## Outputs

All artifacts are written to `results/`:

- Evaluation reports (`accuracy_results.txt`, `evaluation_report.txt`)
- Predictions CSV (`sales_conversion_predictions.csv`)
- Visualizations (conversion distribution, correlation heatmap, ROC curve, etc.)
- Model files (`sales_conversion_prediction_model.joblib`, `sales_conversion_prediction_metadata.joblib`)

## API Endpoints

- `GET /` - Frontend UI
- `GET /health` - Health check
- `POST /predict` - Predict sales conversion
- `GET /results/metrics` - Evaluation report and plot paths
- `GET /results/image/{image_name}` - Serve generated plots
- `POST /retrain` - Retrain model in background
