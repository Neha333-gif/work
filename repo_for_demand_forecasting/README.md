# DEMAND IQ – AI-Powered Demand Forecasting Engine

DEMAND IQ is a premium, state-of-the-art enterprise demand forecasting system. It combines a robust Machine Learning pipeline with a high-performance FastAPI backend and a gorgeous, interactive dark-mode dashboard.

It trains a **RandomForestRegressor** pipeline on historical retail demand data, handles automatic preprocessing, scales features, and provides real-time demand predictions with confidence bounds. It also automatically generates **7 key operational visualizations** for inventory and supply-chain managers.

---

## 🌟 Key Features

1. **Enterprise Machine Learning Pipeline**: 
   - Uses `RandomForestRegressor` with robust hyperparameters for accurate, non-linear demand modeling.
   - Smart dataset sampling to ensure fast training times while keeping excellent data representation.
   - Robust scaling (`StandardScaler`) and categorical encoding (`LabelEncoder`).
   - Generates confidence intervals ($1\sigma$ standard deviation across the forest's estimators) for all predictions.

2. **Interactive Forecast Dashboard**:
   - Premium dark-mode user interface using custom HSL colors, smooth glassmorphic gradients, and micro-animations.
   - Fully responsive design matching executive and operational workflows.
   - Includes full-screen lightbox visualization viewer.

3. **7 Operational Visualizations**:
   - **Sales/Demand Distribution**: Insights into overall unit sales frequency.
   - **Feature Correlation Heatmap**: Highlights linear relationships among drivers of demand.
   - **Feature Importance**: Ranks the top drivers of demand (e.g. Price, Inventory, Promotion).
   - **Actual vs. Predicted**: Evaluates prediction accuracy against historical test data.
   - **Residuals Distribution**: Displays error profile to detect under/over-estimation bias.
   - **Model Metrics Summary**: Visually showcases $R^2$, MAE, and RMSE.
   - **Rolling Demand Trend**: 200-period rolling average tracing historical demand movements.

4. **High-Performance FastAPI Server**:
   - RESTful endpoints for real-time predictions, system health, metrics retrieval, and chart generation.
   - Back-channel background model retraining via a simple POST request.

---

## 📂 Directory Structure

```text
repo_for_demand_forecasting/
├── demand_forecasting_data/
│   └── demand_forecasting.csv           # Historical demand dataset
├── results/
│   ├── accuracy_results.txt             # Model evaluation report
│   ├── sales_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── actual_vs_predicted.png
│   ├── residuals_distribution.png
│   ├── model_metrics.png
│   └── demand_trend.png
├── demand_forecasting.py                # Unified ML pipeline & FastAPI backend
├── demand_forecasting_frontend.html     # Premium dark-mode dashboard UI
├── demand_forecasting_model.joblib      # Trained serialized model
├── demand_forecasting_metadata.joblib   # Serialized scaler, encoders, metrics
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
python demand_forecasting.py --train
```
This runs the full training pipeline, outputs accuracy metrics, saves the model joblib files, and writes the 7 charts to the `results/` folder.

### Step 2: Start the Web Dashboard & Server
Launch the server to host the interactive dashboard and API:
```bash
python demand_forecasting.py
```
By default, the server runs on `http://127.0.0.1:8000`. Open this address in your web browser to access the gorgeous **DEMAND IQ Dashboard**.

---

## 🔗 API Documentation

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Serves the premium dashboard user interface. |
| `/health` | `GET` | Returns system and model health status. |
| `/predict` | `POST` | Takes product/store metadata context and predicts units demand with upper/lower limits. |
| `/results/metrics` | `GET` | Returns the text evaluation report and paths to the generated plots. |
| `/results/image/{image_name}` | `GET` | Serves the specified plot image. |
| `/retrain` | `POST` | Initiates background retraining of the model without downtime. |

### Prediction Payload Example (`POST /predict`)
```json
{
  "store_id": "S001",
  "product_id": "P0001",
  "category": "Electronics",
  "region": "North",
  "inventory_level": 150.0,
  "price": 49.99,
  "discount": 10.0,
  "weather_condition": "Sunny",
  "promotion": 1,
  "competitor_pricing": 52.0,
  "seasonality": "Summer",
  "epidemic": 0
}
```

---

## 🛡️ License
Designed and developed for internal retail operations and supply chain intelligence. All rights reserved.
