# -*- coding: utf-8 -*-
"""
Employee Salary Prediction with FastAPI frontend and model training.

Run locally:
    python employee_salary_prediction.py --train
    python employee_salary_prediction.py

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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

warnings.filterwarnings('ignore')

DATA_FILE = 'job_salary_prediction_dataset.csv'
MODEL_FILE = 'job_salary_prediction_model.joblib'
METADATA_FILE = 'job_salary_prediction_metadata.joblib'
RESULTS_DIR = 'results'
FRONTEND_FILE = 'employee_salary_prediction_frontend.html'
REPORT_FILE = os.path.join(RESULTS_DIR, 'salary_evaluation_report.txt')

os.makedirs(RESULTS_DIR, exist_ok=True)


def _get_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset not found at '{DATA_FILE}'.")
    df = pd.read_csv(DATA_FILE)
    return df


def _prepare_data(df):
    target = 'salary'
    df = df.copy()
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
    plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df['salary'], kde=True, color='#60a5fa', ax=ax)
    ax.set_title('Salary Distribution')
    ax.set_xlabel('Salary')
    ax.set_ylabel('Count')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'salary_distribution.png'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(y_test, y_pred, alpha=0.6, color='#34d399', edgecolors='w', linewidth=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='#facc15')
    ax.set_title('Actual vs Predicted Salary')
    ax.set_xlabel('Actual Salary')
    ax.set_ylabel('Predicted Salary')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'actual_vs_predicted.png'))
    plt.close(fig)

    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    importance.plot.barh(color='#818cf8', ax=ax)
    ax.set_title('Feature Importance')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(x=df['experience_years'], y=df['salary'], hue=df['education_level'], palette='tab10', ax=ax)
    ax.set_title('Experience vs Salary by Education Level')
    ax.set_xlabel('Experience Years')
    ax.set_ylabel('Salary')
    ax.legend(title='Education', bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'experience_vs_salary.png'))
    plt.close(fig)


def train_salary_model():
    print('[INFO] Training salary model...')
    df = _get_data()
    X, y, scaler, encoders, numeric_cols, categorical_cols = _prepare_data(df)
    feature_names = X.columns.tolist()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    report = (
        f"Model evaluation for employee salary prediction\n"
        f"Dataset: {DATA_FILE} ({len(df)} samples)\n"
        f"MAE: {mae:.2f}\n"
        f"RMSE: {rmse:.2f}\n"
        f"R2: {r2:.4f}\n"
    )
    with open(REPORT_FILE, 'w', encoding='utf-8') as fh:
        fh.write(report)

    _generate_plots(df, y_test, y_pred, model, feature_names)

    joblib.dump(model, MODEL_FILE)
    joblib.dump({'scaler': scaler, 'encoders': encoders, 'numeric_cols': numeric_cols, 'categorical_cols': categorical_cols, 'feature_names': feature_names, 'mae': mae, 'rmse': rmse, 'r2': r2}, METADATA_FILE)
    print('[INFO] Training complete. Saved artifacts.')


def load_assets():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(METADATA_FILE):
        return None, None
    model = joblib.load(MODEL_FILE)
    metadata = joblib.load(METADATA_FILE)
    return model, metadata

app = FastAPI(title='Job Salary Prediction')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'], allow_credentials=True)

_model, _metadata = load_assets()

class SalaryRequest(BaseModel):
    job_title: str = Field('Data Analyst')
    experience_years: int = Field(5, ge=0, le=50)
    education_level: str = Field('Bachelor')
    skills_count: int = Field(10, ge=0, le=50)
    industry: str = Field('Technology')
    company_size: str = Field('Medium')
    location: str = Field('USA')
    remote_work: str = Field('No')
    certifications: int = Field(0, ge=0, le=20)

@app.get('/', response_class=HTMLResponse)
def serve_frontend():
    if os.path.exists(FRONTEND_FILE):
        return HTMLResponse(open(FRONTEND_FILE, 'r', encoding='utf-8').read())
    return HTMLResponse('<h1>Frontend not available.</h1>')

@app.get('/health')
def health():
    return {'ready': _model is not None, 'model_file': os.path.exists(MODEL_FILE), 'metadata_file': os.path.exists(METADATA_FILE)}

@app.post('/predict')
def predict(req: SalaryRequest):
    global _model, _metadata
    if _model is None or _metadata is None:
        raise HTTPException(status_code=503, detail='Model not loaded. Run with --train first.')

    data = {
        'job_title': req.job_title,
        'experience_years': float(req.experience_years),
        'education_level': req.education_level,
        'skills_count': float(req.skills_count),
        'industry': req.industry,
        'company_size': req.company_size,
        'location': req.location,
        'remote_work': req.remote_work,
        'certifications': float(req.certifications),
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
    return {'predicted_salary': round(prediction, 2)}

@app.get('/results/metrics')
def metrics():
    if not os.path.exists(REPORT_FILE):
        raise HTTPException(status_code=404, detail='Metrics not found')
    return {'report': open(REPORT_FILE, 'r', encoding='utf-8').read(), 'plots': [
        'salary_distribution.png', 'actual_vs_predicted.png', 'feature_importance.png', 'experience_vs_salary.png'
    ]}

@app.get('/results/image/{name}')
def image(name: str):
    path = os.path.join(RESULTS_DIR, name)
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail='Image not found')

@app.post('/retrain')
def retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_salary_model)
    return {'status': 'retraining started'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run employee salary prediction service')
    parser.add_argument('--train', action='store_true', help='Train model before starting')
    parser.add_argument('--port', type=int, default=8000, help='Port to run FastAPI on')
    args = parser.parse_args()

    if args.train or _model is None:
        train_salary_model()
        _model, _metadata = load_assets()

    import uvicorn
    uvicorn.run('employee_salary_prediction:app', host='127.0.0.1', port=args.port, reload=False)
