# Home Price Prediction

This project provides a full home price prediction pipeline with FastAPI backend and modern web UI.

## Main Files

- `home_price_prediction.py`
- `home_price_prediction_frontend.html`
- `home_price_prediction_model.joblib`
- `home_price_prediction_metadata.joblib`
- `results/`

## Dataset Path

The backend is configured to use:

`C:\Users\peepl\Downloads\cust_segmentation_zip\repo_for_demand_forecasting\repo_for_demand_forecasting\House Price India.csv`

## Run

```bash
pip install -r requirements.txt
python home_price_prediction.py --train
python home_price_prediction.py
```

Then open:

`http://127.0.0.1:8000`
