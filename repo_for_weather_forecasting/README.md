# METEOR IQ – AI-Powered Weather & Climate Forecasting Engine

METEOR IQ is a premium, state-of-the-art climate and weather forecasting system. It combines a robust Machine Learning pipeline with a high-performance FastAPI backend and a gorgeous, interactive dark-mode dashboard.

It trains **RandomForestRegressor** pipelines on weather and climate data, handles automatic preprocessing, scales features, balances suitability classes via SMOTE, and provides real-time weather forecasts and outdoor activity suitability analysis with confidence bounds. It also automatically generates **9 operational visualizations** for climate and operations managers.

---

## 🌟 Key Features

1. **Enterprise Machine Learning Pipeline**: 
   - Uses `RandomForestRegressor` with robust hyperparameters for accurate, non-linear climate and suitability modeling.
   - Robust scaling (`StandardScaler`) and categorical encoding (`LabelEncoder`).
   - Generates confidence intervals ($1\sigma$ standard deviation across the forest's estimators) for play suitability.
   - Smart class balancing using SMOTE.

2. **Interactive Forecast Dashboard**:
   - Premium dark-mode user interface using custom HSL colors, smooth glassmorphic gradients, and micro-animations.
   - Fully responsive design matching executive and operational workflows.
   - Includes full-screen lightbox visualization viewer.

3. **9 Operational Visualizations**:
   - **Temperature Trend Analysis**: Traces daily temperature trends.
   - **Humidity Distribution Profile**: Relative humidity counts and density.
   - **Weather & Climate Correlation Heatmap**: Highlights linear relationships among key climate indicators.
   - **Seasonal Rainfall/Precipitation**: Graph of average precipitation by season.
   - **Actual vs. Predicted**: Evaluates prediction accuracy of forecasted temperatures.
   - **Residuals Distribution**: Displays error profile of the temperature model.
   - **Feature Importance**: Ranks drivers of play suitability.
   - **Forecast Trend**: 30-day moving average tracing long-term temperature cycles.
   - **Model Metrics Summary**: Visually showcases $R^2$ scores comparison.

4. **High-Performance FastAPI Server**:
   - RESTful endpoints for real-time predictions, system health, metrics retrieval, and chart generation.
   - Back-channel background model retraining via a simple POST request.

---

## 📂 Directory Structure

```text
repo_for_weather_forecasting/
├── weather_forecasting_data/
│   ├── weather_forecast.csv             # Historical weather dataset (1000 rows)
│   └── weather_forecast_original.csv    # Original 14-row dataset backup
├── results/
│   ├── accuracy_results.txt             # Model evaluation report
│   ├── temperature_trend.png
│   ├── humidity_distribution.png
│   ├── correlation_heatmap.png
│   ├── rainfall_visualization.png
│   ├── actual_vs_predicted.png
│   ├── residuals_distribution.png
│   ├── feature_importance.png
│   ├── forecast_trend.png
│   └── model_metrics.png
├── weather_forecasting.py                # Unified ML pipeline & FastAPI backend
├── weather_forecasting_frontend.html     # Premium dark-mode dashboard UI
├── weather_forecasting_model.joblib      # Trained serialized models
├── weather_forecasting_metadata.joblib   # Serialized scaling, encoders, metrics
├── requirements.txt                     # Dependencies
└── README.md                            # Documentation (this file)
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
Ensure you have **Python 3.9+** installed on your system.

### 2. Install Dependencies
Navigate to this directory and install the required libraries:
```bash
pip install -r requirements.txt
```

---

## 📈 Running the Engine

### Step 1: Train the Model & Generate Visualizations
Before launching the server, train the model to build the predictive assets and visualizations:
```bash
python weather_forecasting.py --train
```
This runs the full training pipeline, outputs accuracy metrics, saves the model joblib files, and writes the 9 charts to the `results/` folder.

### Step 2: Start the Web Dashboard & Server
Launch the server to host the interactive dashboard and API:
```bash
python weather_forecasting.py
```
By default, the server runs on `http://127.0.0.1:8000`. Open this address in your web browser to access the gorgeous **METEOR IQ Dashboard**.

---

## 🔗 API Documentation

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Serves the premium dashboard user interface. |
| `/health` | `GET` | Returns system and model health status. |
| `/predict` | `POST` | Takes climate variables and predicts play suitability, forecasted temperature, and forecasted rainfall. |
| `/results/metrics` | `GET` | Returns the text evaluation report and paths to the generated plots. |
| `/results/image/{image_name}` | `GET` | Serves the specified plot image. |
| `/retrain` | `POST` | Initiates background retraining of the model without downtime. |

### Prediction Payload Example (`POST /predict`)
```json
{
  "temperature": 25.5,
  "humidity": 60.0,
  "wind_speed": 15.2,
  "atmospheric_pressure": 1012.3,
  "rainfall": 0.0,
  "region": "North",
  "date": "2026-05-26",
  "season": "Summer"
}
```

---

## 🛡️ License
Designed and developed for internal agricultural and operations intelligence. All rights reserved.
