# -*- coding: utf-8 -*-
"""
Telecom Revenue Prediction with FastAPI frontend and model training.

Run locally:
    python telecom_revenue_data.py --train
    python telecom_revenue_data.py

Open the UI at:
    http://127.0.0.1:8000
"""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

warnings.filterwarnings('ignore')

DATA_FILE = 'telecom_revenue_prediction_dataset.csv'
MODEL_FILE = 'telecom_revenue_prediction_model.joblib'
METADATA_FILE = 'telecom_revenue_prediction_metadata.joblib'
RESULTS_DIR = 'results'
FRONTEND_FILE = 'telecom_revenue_prediction_frontend.html'
REPORT_FILE = os.path.join(RESULTS_DIR, 'telecom_revenue_evaluation_report.txt')

os.makedirs(RESULTS_DIR, exist_ok=True)


def _get_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset not found at '{DATA_FILE}'.")
    df = pd.read_csv(DATA_FILE)
    return df


def _prepare_data(df):
    target = 'Aggregate_Total_Rev'
    df = df.copy()
    # Remove rows with missing target values
    df = df.dropna(subset=[target])
    df.fillna('Unknown', inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target in categorical_cols:
        categorical_cols.remove(target)
    if target in numeric_cols:
        numeric_cols.remove(target)

    imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = imputer.fit_transform(df[categorical_cols])

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    X = df.drop(columns=[target])
    y = df[target].astype(float)
    return X, y, scaler, encoders, numeric_cols, categorical_cols


def _generate_plots(df, y_test, y_pred, model, feature_names):
    """Generate comprehensive visualization plots for telecom revenue analysis."""
    plt.style.use('dark_background')

    # Revenue Distribution Plot
    if 'Aggregate_Total_Rev' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['Aggregate_Total_Rev'], kde=True, color='#60a5fa', ax=ax, bins=30)
        ax.set_title('Telecom Revenue Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Revenue', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, 'revenue_distribution.png'), dpi=100)
        plt.close(fig)

    # Actual vs Predicted Revenue
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.6, color='#34d399', edgecolors='w', linewidth=0.5, s=50)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='#facc15', linewidth=2)
    ax.set_title('Actual vs Predicted Telecom Revenue', fontsize=14, fontweight='bold')
    ax.set_xlabel('Actual Revenue ($)', fontsize=12)
    ax.set_ylabel('Predicted Revenue ($)', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'actual_vs_predicted.png'), dpi=100)
    plt.close(fig)

    # Feature Importance Plot
    if hasattr(model, 'feature_importances_'):
        importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        importance.tail(15).plot.barh(color='#818cf8', ax=ax)
        ax.set_title('Top Feature Importance for Revenue Prediction', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance Score', fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'), dpi=100)
        plt.close(fig)

    # Residual Distribution Plot
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='#fb7185', ax=ax, bins=30)
    ax.set_title('Prediction Residuals Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Residual Value ($)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'residuals_distribution.png'), dpi=100)
    plt.close(fig)

    # Predicted Revenue Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_pred, bins=30, alpha=0.7, color='#06b6d4', label='Predicted', edgecolor='white')
    ax.hist(y_test, bins=30, alpha=0.5, color='#f59e0b', label='Actual', edgecolor='white')
    ax.set_title('Revenue Distribution: Predicted vs Actual', fontsize=14, fontweight='bold')
    ax.set_xlabel('Revenue ($)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'revenue_comparison.png'), dpi=100)
    plt.close(fig)

    # Model Performance Metrics Visualization
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    metrics = ['MAE', 'RMSE', 'R² Score']
    values = [mae, rmse, r2]
    colors = ['#3b82f6', '#10b981', '#f59e0b']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics, values, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=12)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'model_performance.png'), dpi=100)
    plt.close(fig)


def train_revenue_model():
    """Train telecom revenue prediction model."""
    print('[INFO] Training telecom revenue prediction model...')
    df = _get_data()
    X, y, scaler, encoders, numeric_cols, categorical_cols = _prepare_data(df)
    feature_names = X.columns.tolist()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Regressor
    model = RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    report = (
        f"Telecom Revenue Prediction Model Evaluation Report\n"
        f"{'='*50}\n"
        f"Dataset: {DATA_FILE} ({len(df)} samples)\n"
        f"Model Type: Random Forest Regressor\n"
        f"Training Samples: {len(x_train)} | Test Samples: {len(x_test)}\n"
        f"\nPerformance Metrics:\n"
        f"  Mean Absolute Error (MAE): ${mae:,.2f}\n"
        f"  Mean Squared Error (MSE): ${mse:,.2f}\n"
        f"  Root Mean Squared Error (RMSE): ${rmse:,.2f}\n"
        f"  R² Score: {r2:.4f}\n"
        f"\nFeatures Used: {len(feature_names)}\n"
        f"  Numeric Features: {len(numeric_cols)}\n"
        f"  Categorical Features: {len(categorical_cols)}\n"
        f"\nGenerated Visualizations:\n"
        f"  - revenue_distribution.png\n"
        f"  - actual_vs_predicted.png\n"
        f"  - feature_importance.png\n"
        f"  - residuals_distribution.png\n"
        f"  - revenue_comparison.png\n"
        f"  - model_performance.png\n"
    )
    with open(REPORT_FILE, 'w', encoding='utf-8') as fh:
        fh.write(report)

    _generate_plots(df, y_test, y_pred, model, feature_names)

    joblib.dump(model, MODEL_FILE)
    joblib.dump({'scaler': scaler, 'encoders': encoders, 'numeric_cols': numeric_cols, 'categorical_cols': categorical_cols, 'feature_names': feature_names, 'mae': mae, 'rmse': rmse, 'r2': r2}, METADATA_FILE)
    print('[INFO] Training complete. Model saved to:', MODEL_FILE)
    print('[INFO] Metadata saved to:', METADATA_FILE)


def load_assets():
    """Load pre-trained model and metadata."""
    if not os.path.exists(MODEL_FILE) or not os.path.exists(METADATA_FILE):
        return None, None
    model = joblib.load(MODEL_FILE)
    metadata = joblib.load(METADATA_FILE)
    return model, metadata

app = FastAPI(title='Telecom Revenue Prediction')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'], allow_credentials=True)

_model, _metadata = load_assets()

class RevenueRequest(BaseModel):
    """Request model for telecom revenue prediction."""
    network_age: int = Field(24, ge=0, le=3000)
    Aggregate_SMS_Rev: float = Field(50.0, ge=0, le=1000)
    Aggregate_Data_Rev: float = Field(100.0, ge=0, le=1000)
    Aggregate_Data_Vol: float = Field(500.0, ge=0, le=50000)
    Aggregate_Calls: int = Field(200, ge=0, le=5000)
    Aggregate_ONNET_REV: int = Field(10000, ge=0, le=100000)
    Aggregate_OFFNET_REV: int = Field(50000, ge=0, le=200000)
    Aggregate_complaint_count: int = Field(0, ge=0, le=100)
    aug_user_type: str = Field('2G')
    sep_user_type: str = Field('2G')
    aug_fav_a: str = Field('telenor')
    sep_fav_a: str = Field('mobilink')

@app.get('/', response_class=HTMLResponse)
def serve_frontend():
    """Serve the telecom revenue prediction frontend."""
    if os.path.exists(FRONTEND_FILE):
        return HTMLResponse(open(FRONTEND_FILE, 'r', encoding='utf-8').read())
    return HTMLResponse('<h1>Frontend not available.</h1>')

@app.get('/health')
def health():
    """Health check endpoint."""
    return {'ready': _model is not None, 'model_file': os.path.exists(MODEL_FILE), 'metadata_file': os.path.exists(METADATA_FILE)}

@app.post('/predict')
def predict(req: RevenueRequest):
    """Predict telecom revenue based on customer features."""
    global _model, _metadata
    if _model is None or _metadata is None:
        raise HTTPException(status_code=503, detail='Model not loaded. Run with --train first.')

    data = {
        'network_age': float(req.network_age),
        'Aggregate_SMS_Rev': float(req.Aggregate_SMS_Rev),
        'Aggregate_Data_Rev': float(req.Aggregate_Data_Rev),
        'Aggregate_Data_Vol': float(req.Aggregate_Data_Vol),
        'Aggregate_Calls': float(req.Aggregate_Calls),
        'Aggregate_ONNET_REV': float(req.Aggregate_ONNET_REV),
        'Aggregate_OFFNET_REV': float(req.Aggregate_OFFNET_REV),
        'Aggregate_complaint_count': float(req.Aggregate_complaint_count),
        'aug_user_type': req.aug_user_type,
        'sep_user_type': req.sep_user_type,
        'aug_fav_a': req.aug_fav_a,
        'sep_fav_a': req.sep_fav_a,
    }
    df = pd.DataFrame([data], columns=_metadata['feature_names'])
    for col in _metadata['categorical_cols']:
        if col in df.columns:
            le = _metadata['encoders'][col]
            raw_value = str(df.at[0, col])
            if raw_value in le.classes_:
                df.at[0, col] = int(le.transform([raw_value])[0])
            else:
                df.at[0, col] = int(len(le.classes_) // 2)
    df[_metadata['numeric_cols']] = _metadata['scaler'].transform(df[_metadata['numeric_cols']].astype(float))
    X = df[_metadata['feature_names']].to_numpy(dtype=float)
    prediction = float(_model.predict(X)[0])
    return {'predicted_revenue': round(prediction, 2), 'currency': 'USD'}

@app.get('/results/metrics')
def metrics():
    """Get model evaluation metrics and visualization list."""
    if not os.path.exists(REPORT_FILE):
        raise HTTPException(status_code=404, detail='Metrics not found')
    return {'report': open(REPORT_FILE, 'r', encoding='utf-8').read(), 'plots': [
        'revenue_distribution.png', 'actual_vs_predicted.png', 'feature_importance.png', 
        'residuals_distribution.png', 'revenue_comparison.png', 'model_performance.png'
    ]}

@app.get('/results/image/{name}')
def image(name: str):
    """Serve visualization images from results folder."""
    path = os.path.join(RESULTS_DIR, name)
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail='Image not found')

@app.post('/retrain')
def retrain(background_tasks: BackgroundTasks):
    """Trigger model retraining in background."""
    background_tasks.add_task(train_revenue_model)
    return {'status': 'Telecom revenue model retraining started'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run telecom revenue prediction service')
    parser.add_argument('--train', action='store_true', help='Train model before starting')
    parser.add_argument('--port', type=int, default=8000, help='Port to run FastAPI on')
    args = parser.parse_args()

    if args.train or _model is None:
        train_revenue_model()
        _model, _metadata = load_assets()

    import uvicorn
    uvicorn.run('telecom_revenue_data:app', host='127.0.0.1', port=args.port, reload=False)
