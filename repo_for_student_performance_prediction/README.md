# Student Performance Prediction

Unified ML pipeline and FastAPI backend for predicting student final exam performance from academic data.

## Dataset

`student performance/student_performance_dataset.csv`

## Quick Start

```bash
pip install -r requirements.txt
python student_performance_prediction.py --train
python student_performance_prediction.py --port 8000
```

Open `http://127.0.0.1:8000/` for the frontend UI.

## Models

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

The best model (lowest RMSE) is saved automatically.

## Outputs

All artifacts are written to `results/`:

- Evaluation reports (`accuracy_results.txt`, `evaluation_report.txt`)
- Predictions CSV (`student_performance_predictions.csv`)
- Visualizations (score distribution, correlation heatmap, feature importance, etc.)
- Model files (`student_performance_prediction_model.joblib`, `student_performance_prediction_metadata.joblib`)

## API Endpoints

- `GET /` - Frontend UI
- `GET /health` - Health check
- `POST /predict` - Predict student performance
- `GET /results/metrics` - Evaluation report and plot paths
- `GET /results/image/{image_name}` - Serve generated plots
- `POST /retrain` - Retrain model in background
