# Retail Sales Forecasting

Unified ML pipeline and FastAPI backend for forecasting retail sales from transaction data.

## Dataset

`retail_sales_dataset.csv`

## Quick Start

```bash
pip install -r requirements.txt
python retail_sales_forecasting.py --train
python retail_sales_forecasting.py --port 8000
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
- Forecasts CSV (`retail_sales_forecasts.csv`)
- Visualizations (sales trend, correlation heatmap, feature importance, etc.)
- Model files (`retail_sales_forecasting_model.joblib`, `retail_sales_forecasting_metadata.joblib`)

## API Endpoints

- `GET /` - Frontend UI
- `GET /health` - Health check
- `POST /predict` - Forecast retail sales
- `GET /results/metrics` - Evaluation report and plot paths
- `GET /results/image/{image_name}` - Serve generated plots
- `POST /retrain` - Retrain model in background
