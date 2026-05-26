# Inventory Optimization System

## 🏪 Optimize Inventory Levels with Machine Learning

This folder contains a complete inventory optimization solution built with scikit-learn and FastAPI. Predict optimal inventory levels, demand forecasts, and stock requirements using lightweight machine learning models.

### What is included

- `inventory_optimization_system_api.py` — FastAPI backend server for predictions
- `inventory_optimization_system_trainer.py` — training pipeline and visualization generation
- `inventory_optimization_system_frontend.html` — responsive web UI for inventory predictions
- `inventory_optimization_system_pipeline.joblib` — trained Random Forest model
- `inventory_optimization_system_metadata.joblib` — model metadata and feature information
- `requirements.txt` — backend dependencies
- `results/` — generated visualization charts and evaluation artifacts

### Dataset

The dataset comes from `../inventory_optimization_system_dataset/grocery_chain_data.csv` and includes fields such as:

- `store_name` — Store location/branch
- `aisle` — Product category/aisle
- `product_name` — Product identifier
- `quantity` — Units in transaction
- `unit_price` — Price per unit ($)
- `total_amount` — Transaction total ($)
- `discount_amount` — Discount applied ($)
- `loyalty_points` — Customer loyalty points
- `final_amount` — Final inventory value (target variable)

### Run training

```bash
cd inventory_optimization_system
python inventory_optimization_system_trainer.py
```

This will:
- Load and preprocess grocery chain data
- Train Linear Regression, Decision Tree, and Random Forest models
- Generate 7+ visualization charts
- Save model pipeline and metadata
- Export evaluation report and sample predictions

### Start the backend

```bash
cd inventory_optimization_system
python -m uvicorn inventory_optimization_system_api:app --host 127.0.0.1 --port 8000
```

Then open in browser:

```
http://127.0.0.1:8000/
```

### API Endpoints

- `GET /` — Serve frontend UI
- `GET /health` — Check API health and model status
- `POST /predict` — Single inventory prediction
- `GET /results/metrics` — View model evaluation metrics and charts
- `GET /results/image/{image_name}` — Access visualization images

### Available results

**Visualizations:**
- `inventory_distribution.png` — Distribution of inventory values
- `correlation_heatmap.png` — Feature correlation analysis
- `demand_forecast.png` — Actual vs predicted inventory values
- `residuals_plot.png` — Prediction error analysis
- `feature_importance.png` — Top contributing features
- `model_comparison.png` — Model performance comparison
- `optimization_distribution.png` — Predicted vs actual distribution

**Reports & Data:**
- `inventory_optimization_system_evaluation.txt` — Detailed model evaluation report
- `sample_inventory_optimization_predictions.csv` — Sample predictions with errors

### Model Performance

**Random Forest Regressor (Best Model)**
- Mean Absolute Error (MAE): $0.53
- Root Mean Squared Error (RMSE): $0.88
- R² Score: 0.9992
- Mean Absolute Percentage Error (MAPE): 0.04%

### Feature Engineering

**Categorical Features:**
- store_name (4 unique stores)
- aisle (6 product categories)
- product_name (various products)

**Numeric Features:**
- quantity (units purchased)
- unit_price ($/unit)
- total_amount (transaction total)
- discount_amount (discount %)
- loyalty_points (customer rewards)

### Frontend Capabilities

- ✅ Single inventory prediction with real-time results
- ✅ Interactive form with validation
- ✅ Status indicators (High/Medium/Low stock)
- ✅ Reorder recommendations
- ✅ Savings potential estimates
- ✅ Risk level assessment
- ✅ Responsive design for mobile/desktop

### Technical Stack

- **Backend:** FastAPI + Python 3.x
- **ML Models:** scikit-learn (Linear Regression, Decision Tree, Random Forest)
- **Data Processing:** pandas, numpy
- **Visualizations:** matplotlib, seaborn
- **Model Serialization:** joblib
- **Frontend:** HTML5 + vanilla JavaScript

### Notes

- The project uses only **lightweight ML models** (no deep learning)
- All models trained with 80/20 train-test split and stratification
- Feature scaling applied for numeric features
- One-hot encoding for categorical variables
- Model selection based on R² score and balanced metrics
- All terminology aligned with **inventory optimization domain**

### Future Improvements

- Real-time model retraining with new transaction data
- Seasonal pattern detection
- Demand forecasting with time-series models
- Integration with supply chain management systems
- Mobile app for on-field inventory checks
- Multi-warehouse optimization


