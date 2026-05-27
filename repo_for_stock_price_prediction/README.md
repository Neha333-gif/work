# Stock Price Prediction

Unified ML pipeline and FastAPI backend for predicting stock prices from market data.

## Dataset

`stock_data.csv` (Stock_1 through Stock_5 price columns)

## Quick Start

```bash
pip install -r requirements.txt
python stock_price_prediction.py --train
python stock_price_prediction.py --port 8000
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
- Predictions CSV (`stock_price_predictions.csv`)
- Visualizations (stock price trend, correlation heatmap, feature importance, etc.)
- Model files (`stock_price_prediction_model.joblib`, `stock_price_prediction_metadata.joblib`)

## API Endpoints

- `GET /` - Frontend UI
- `GET /health` - Health check
- `POST /predict` - Predict stock price
- `GET /results/metrics` - Evaluation report and plot paths
- `GET /results/image/{image_name}` - Serve generated plots
- `POST /retrain` - Retrain model in background
