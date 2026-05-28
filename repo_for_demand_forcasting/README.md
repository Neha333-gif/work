# Demand Forecasting System

Unified ML pipeline and FastAPI backend for demand forecasting using sales and inventory demand data.

## Dataset Path

Primary path is kept same as requested in the Python file:

- `/content/demand_forecasting.csv`

If unavailable, the app falls back to:

- `../demand forcasting system/train_0irEZ2H.csv`

## Quick Start

```bash
pip install -r requirements.txt
python demand_forcasting.py --train
python demand_forcasting.py --port 8000
```

Open `http://127.0.0.1:8000/`.

## Models (Lightweight Only)

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

## Outputs

All artifacts are generated under `results/`:

- Evaluation reports (`accuracy_results.txt`, `evaluation_report.txt`)
- Forecast outputs (`demand_forecasting_outputs.csv`, `model_metrics_summary.csv`)
- Visualizations:
  - `demand_trend.png`
  - `correlation_heatmap.png`
  - `feature_importance.png`
  - `actual_vs_forecasted_demand.png`
  - `residual_error_distribution.png`
  - `seasonal_demand_analysis.png`
  - `product_category_demand_comparison.png`
  - `inventory_vs_demand_analysis.png`
  - `model_performance_metrics_visualization.png`

## API Endpoints

- `GET /` - frontend UI
- `GET /health` - health check
- `POST /forecast` - forecast demand
- `GET /results/metrics` - evaluation report
- `GET /results/image/{image_name}` - serve generated images
- `POST /retrain` - retrain in background
