# Taxi Trip Fare Prediction System

## 🚕 Predict Taxi Trip Fare with Lightweight ML Models

This folder contains a complete taxi fare prediction solution built with scikit-learn and FastAPI.

### What is included

- `taxi_trip_fare_prediction_api.py` — FastAPI backend server
- `taxi_trip_fare_prediction_trainer.py` — training pipeline and visualization generation
- `taxi_trip_fare_prediction_frontend.html` — responsive web UI for single and batch predictions
- `taxi_trip_fare_model.py` — standalone taxi trip fare regression demo script
- `requirements.txt` — backend dependencies
- `results/` — generated evaluation charts and artifacts

### Dataset

The dataset lives in `../taxi_trip_fare_prediction_dataset/` and includes fields such as:

- `trip_duration`
- `distance_traveled`
- `num_of_passengers`
- `fare`
- `surge_applied`
- `total_fare`

The model uses the trip features above to estimate the final taxi fare.

### Run training

```bash
cd taxi_trip_fare_prediction
python taxi_trip_fare_prediction_trainer.py
```

### Start the backend

```bash
cd taxi_trip_fare_prediction
uvicorn taxi_trip_fare_prediction_api:app --reload
```

Then open:

```bash
http://127.0.0.1:8000/
```

### Available results

- `taxi_trip_fare_prediction_evaluation.txt`
- `sample_taxi_trip_fare_predictions.csv`
- `fare_distribution.png`
- `correlation_heatmap.png`
- `actual_vs_predicted.png`
- `residuals_distribution.png`
- `distance_vs_fare.png`
- `feature_importance.png`
- `model_performance.png`
- `prediction_distribution.png`

### Notes

- The project uses lightweight regression models only: Linear Regression, Decision Tree, and Random Forest.
- The frontend supports single-record predictions and batch CSV upload.
- The entire codebase is aligned with taxi trip fare prediction terminology.
