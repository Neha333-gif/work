# Energy Consumption Prediction

> **AI-powered energy consumption forecasting** using a lightweight Random Forest Regression model trained on building energy usage, environmental, and operational data.

---

## Project Structure

```
repo_for_energy_consumption_prediction/
│
├── energy_consumption_prediction.py              # FastAPI backend + ML training pipeline
├── energy_consumption_model.py                   # Standalone training helper
├── energy_consumption_prediction_frontend.html   # Responsive frontend UI
│
├── energy_consumption_prediction_data/
│   └── Energy_consumption_dataset.csv            # Energy usage dataset
│
├── results/
│   ├── accuracy_results.txt                      # Model evaluation report
│   ├── predictions_report.csv                     # Actual vs predicted values
│   ├── actual_vs_predicted.png                    # Regression analysis
│   ├── correlation_heatmap.png                    # Feature correlation
│   ├── residual_distribution.png                  # Error distribution
│   ├── daily_energy_usage.png                     # Daily usage trend
│   ├── peak_consumption.png                       # Peak hour analysis
│   ├── feature_importance.png                     # Feature importance
│   ├── model_metrics.png                          # Performance summary
│   └── ...
│
├── energy_consumption_prediction_model.joblib    # Saved regression model
├── energy_consumption_prediction_metadata.joblib # Saved preprocessing metadata
├── requirements.txt
└── README.md
```

---

## Dataset

**Energy_consumption_dataset.csv** — Contains building energy consumption records with features such as:

- `Month`
- `Hour`
- `DayOfWeek`
- `Holiday`
- `Temperature`
- `Humidity`
- `SquareFootage`
- `Occupancy`
- `HVACUsage`
- `LightingUsage`
- `RenewableEnergy`
- `EnergyConsumption` (target)

---

## ML Pipeline

- **Model:** `RandomForestRegressor`
- **Preprocessing:** Label encoding, scaling, categorical handling
- **Evaluation:** MAE, MSE, RMSE, R²
- **Visualizations:** actual vs predicted, correlation heatmap, residuals, peak usage, feature importance
- **Backend:** FastAPI server with prediction, metrics, and visualization routes

---

## Quickstart

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the model and start the server:

```bash
python energy_consumption_prediction.py --train
```

3. Open the frontend in your browser:

```text
http://127.0.0.1:8000
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve the frontend UI |
| `GET` | `/health` | API health and model status |
| `POST` | `/predict` | Predict energy consumption |
| `GET` | `/results/metrics` | Return evaluation report and plots |
| `GET` | `/results/image/{name}` | Return generated plot image |
| `POST` | `/retrain` | Retrain the model in the background |

---

## Prediction Example

```json
{
  "Month": 7,
  "Hour": 14,
  "DayOfWeek": "Wednesday",
  "Holiday": "No",
  "Temperature": 27.3,
  "Humidity": 55.1,
  "SquareFootage": 1580.0,
  "Occupancy": 8,
  "HVACUsage": "On",
  "LightingUsage": "Off",
  "RenewableEnergy": 12.5
}
```

`/predict` returns a JSON response with predicted consumption, trend estimate, and model metrics.

---

## Notes

- The frontend, backend, and results are fully integrated.
- All project artifacts now reflect the energy consumption prediction domain.
