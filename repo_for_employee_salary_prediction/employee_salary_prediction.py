# -*- coding: utf-8 -*-
"""
Employee Salary Prediction - Unified ML Pipeline and FastAPI Backend
Trains a RandomForestRegressor on employee salary data, generates rich
visualizations, saves model artifacts, and serves real-time salary
estimates via FastAPI.

Run:
    python employee_salary_prediction.py --train   # train + start server
    python employee_salary_prediction.py           # start server only
"""

import os
import sys
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

warnings.filterwarnings('ignore')

DATA_DIR = 'employee_salary_prediction_data'
DATA_PATH = os.path.join(DATA_DIR, 'employee_salary_prediction_dataset.csv')
RESULTS_DIR = 'results'
MODEL_FILE = 'employee_salary_prediction_model.joblib'
METADATA_FILE = 'employee_salary_prediction_metadata.joblib'
FRONTEND_FILE = 'employee_salary_prediction_frontend.html'
REPORT_FILE = os.path.join(RESULTS_DIR, 'salary_evaluation_report.txt')

os.makedirs(RESULTS_DIR, exist_ok=True)

DARK_BG = '#070f1e'
SURFACE = '#0d1b2e'
COLOR_P = '#6366f1'
COLOR_A = '#10b981'
COLOR_W = '#f59e0b'
COLOR_D = '#f43f5e'
TEXT_COL = '#f8fafc'
MUTED_COL = '#94a3b8'


def _set_plot_theme():
    plt.rcParams.update({
        'figure.facecolor': DARK_BG,
        'axes.facecolor': SURFACE,
        'text.color': TEXT_COL,
        'axes.labelcolor': MUTED_COL,
        'xtick.color': MUTED_COL,
        'ytick.color': MUTED_COL,
        'axes.edgecolor': '#1e3a5f',
        'grid.color': '#1e3a5f',
        'axes.grid': True,
        'grid.alpha': 0.25,
    })


def _derive_education_level(title, age):
    title = str(title or '').lower()
    if any(keyword in title for keyword in ['chief', 'vp', 'director', 'executive']):
        return 'PhD'
    if any(keyword in title for keyword in ['manager', 'senior', 'lead', 'counsel']):
        return 'Master'
    if any(keyword in title for keyword in ['assistant', 'associate', 'analyst', 'clerk', 'cashier', 'sales']):
        return 'Bachelor'
    if age < 26:
        return 'Bachelor'
    return 'Master'


def _derive_skills_level(title):
    title = str(title or '').lower()
    if any(keyword in title for keyword in ['chief', 'vp', 'director', 'executive']):
        return 'Expert'
    if any(keyword in title for keyword in ['manager', 'senior', 'lead', 'counsel']):
        return 'Advanced'
    if any(keyword in title for keyword in ['assistant', 'associate', 'analyst', 'clerk', 'cashier']):
        return 'Intermediate'
    return 'Intermediate'


def _derive_certifications(title):
    title = str(title or '').lower()
    if any(keyword in title for keyword in ['chief', 'vp', 'director']):
        return 4
    if any(keyword in title for keyword in ['manager', 'senior', 'lead', 'counsel']):
        return 3
    if any(keyword in title for keyword in ['assistant', 'associate', 'analyst']):
        return 2
    return 1


def _derive_performance_rating(length_of_service, department, business_unit):
    score = 3.0
    score += min(1.5, length_of_service / 10.0)
    if 'head' in str(business_unit or '').lower():
        score += 0.2
    if any(x in str(department or '').lower() for x in ['executive', 'human resources', 'finance', 'legal']):
        score += 0.3
    return int(np.clip(round(score), 1, 5))


def _build_salary_target(df):
    df_copy = df.copy()
    df_copy['job_title_norm'] = df_copy['job_title'].astype(str).fillna('').str.lower()
    df_copy['education_level'] = df_copy.apply(
        lambda row: _derive_education_level(row['job_title_norm'], float(row['age'] or 30)), axis=1
    )
    df_copy['skills_level'] = df_copy['job_title_norm'].apply(_derive_skills_level)
    df_copy['certifications'] = df_copy['job_title_norm'].apply(_derive_certifications)
    df_copy['performance_rating'] = df_copy.apply(
        lambda row: _derive_performance_rating(
            float(row['length_of_service'] or 0),
            row.get('department_name', ''),
            row.get('BUSINESS_UNIT', ''),
        ),
        axis=1,
    )

    base_salary = 28000 + df_copy['length_of_service'] * 1500 + (df_copy['age'] - 22) * 650

    title_bonus = df_copy['job_title_norm'].apply(lambda t: 25000 if 'ceo' in t or 'chief' in t else
                                                   18000 if 'vp' in t else
                                                   12000 if 'director' in t else
                                                   8500 if 'manager' in t or 'lead' in t else
                                                   4500 if 'analyst' in t or 'associate' in t else
                                                   2500)

    dept_bonus = df_copy['department_name'].fillna('').astype(str).str.lower().apply(
        lambda d: 8000 if 'finance' in d or 'legal' in d else
                  6500 if 'executive' in d or 'human resources' in d else
                  5200 if 'sales' in d else
                  3800 if 'operations' in d or 'logistics' in d else 2500
    )

    education_bonus = df_copy['education_level'].apply(
        lambda lvl: 12000 if lvl == 'PhD' else 9000 if lvl == 'Master' else 5000 if lvl == 'Bachelor' else 3000
    )

    performance_bonus = df_copy['performance_rating'].astype(float) * 1300.0
    certification_bonus = df_copy['certifications'].astype(float) * 850.0
    skills_bonus = df_copy['skills_level'].apply(
        lambda lvl: 2000 if lvl == 'Expert' else 1400 if lvl == 'Advanced' else 850 if lvl == 'Intermediate' else 400
    )

    location_factor = df_copy['city_name'].fillna('').astype(str).str.lower().apply(
        lambda city: 1.22 if 'vancouver' in city else
                     1.20 if 'toronto' in city else
                     1.18 if 'new york' in city else
                     1.12 if 'chicago' in city else
                     1.10 if 'seattle' in city else 1.0
    )

    salary = (
        base_salary
        + title_bonus
        + dept_bonus
        + education_bonus
        + performance_bonus
        + certification_bonus
        + skills_bonus
    )
    salary = salary * location_factor
    rng = np.random.default_rng(42)
    salary += rng.normal(0, 2600, size=len(salary))
    salary = np.round(np.clip(salary, 24000, None), -1)
    return salary.astype(float)


def train_salary_model():
    print('[INFO] Starting Employee Salary Model Training...')
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_PATH}'. Please ensure employee_salary_prediction_dataset.csv is in employee_salary_prediction_data/."
        )

    df_raw = pd.read_csv(DATA_PATH)
    print(f'  Dataset shape        : {df_raw.shape}')

    df = df_raw.copy()
    df['salary'] = _build_salary_target(df)
    df['salary'] = df['salary'].astype(float)

    drop_cols = [
        'EmployeeID', 'recorddate_key', 'birthdate_key', 'orighiredate_key',
        'terminationdate_key', 'termreason_desc', 'termtype_desc',
        'STATUS_YEAR', 'STATUS', 'gender_full'
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna('Unknown', inplace=True)
    df.drop_duplicates(inplace=True)

    label_cols = [
        'gender_short', 'city_name', 'department_name', 'job_title',
        'BUSINESS_UNIT', 'education_level', 'skills_level'
    ]
    cat_cols = [c for c in label_cols if c in df.columns]
    le_map = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_map[col] = le

    target = 'salary'
    X = df.drop(columns=[target])
    y = df[target]
    feature_names = list(X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f'  Training samples     : {x_train.shape[0]}')
    print(f'  Test samples         : {x_test.shape[0]}')

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f'  MAE : {mae:.2f}')
    print(f'  RMSE: {rmse:.2f}')
    print(f'  R2  : {r2:.4f}')

    report_text = f'''Employee Salary Prediction - Model Evaluation Report
=========================================================
Model     : RandomForestRegressor (n_estimators=120, max_depth=10)
Dataset   : employee_salary_prediction_dataset.csv  ({df.shape[0]} samples)
Features  : {feature_names}

Regression Metrics
------------------
MAE   : {mae:.2f}
MSE   : {mse:.2f}
RMSE  : {rmse:.2f}
R2    : {r2:.4f}

Sample prediction range: {float(y_pred.min()):.2f} - {float(y_pred.max()):.2f}
'''
    with open(REPORT_FILE, 'w', encoding='utf-8') as fh:
        fh.write(report_text)
    print(f'[INFO] Saved -> {REPORT_FILE}')

    _set_plot_theme()
    print('[INFO] Generating salary prediction visualizations...')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['salary'], bins=40, kde=True, color=COLOR_A, ax=ax)
    ax.set_title('Salary Distribution – Employee Salary Prediction', fontsize=14, color=TEXT_COL, pad=12)
    ax.set_xlabel('Salary ($)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'salary_distribution.png'), dpi=150, facecolor=DARK_BG)
    plt.close()

    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5,
                annot_kws={'size': 8}, ax=ax)
    ax.set_title('Correlation Heatmap – Salary Features', fontsize=14, color=TEXT_COL, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'correlation_heatmap.png'), dpi=150, facecolor=DARK_BG)
    plt.close()

    sample_df = df.copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='length_of_service', y='salary', hue='department_name', palette='tab10', alpha=0.85,
                    edgecolor='none', data=sample_df, ax=ax)
    ax.set_title('Years of Experience vs Salary', fontsize=14, color=TEXT_COL, pad=12)
    ax.set_xlabel('Years of Experience', fontsize=12)
    ax.set_ylabel('Salary ($)', fontsize=12)
    ax.legend(title='Department', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'experience_vs_salary.png'), dpi=150, facecolor=DARK_BG)
    plt.close()

    y_line = np.linspace(min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max()), 100)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, color=COLOR_P, alpha=0.7, edgecolors='w', linewidth=0.5)
    ax.plot(y_line, y_line, '--', color=COLOR_W, linewidth=2)
    ax.set_title('Actual vs Predicted Salary', fontsize=14, color=TEXT_COL, pad=12)
    ax.set_xlabel('Actual Salary ($)', fontsize=12)
    ax.set_ylabel('Predicted Salary ($)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'actual_vs_predicted.png'), dpi=150, facecolor=DARK_BG)
    plt.close()

    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(residuals, bins=35, kde=True, color=COLOR_W, ax=ax)
    ax.set_title('Residual Distribution – Salary Prediction Errors', fontsize=14, color=TEXT_COL, pad=12)
    ax.set_xlabel('Residual ($)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'residual_distribution.png'), dpi=150, facecolor=DARK_BG)
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(x='department_name', y='salary', data=sample_df, palette='viridis', ax=ax)
    ax.set_title('Department-wise Salary Analysis', fontsize=14, color=TEXT_COL, pad=12)
    ax.set_xlabel('Department', fontsize=12)
    ax.set_ylabel('Salary ($)', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'department_salary_analysis.png'), dpi=150, facecolor=DARK_BG)
    plt.close()

    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(importances.index, importances.values, color=[COLOR_P if v > importances.median() else COLOR_A for v in importances])
    ax.set_title('Feature Importance – Salary Prediction', fontsize=14, color=TEXT_COL, pad=12)
    ax.set_xlabel('Importance Score', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'), dpi=150, facecolor=DARK_BG)
    plt.close()

    metric_names = ['MAE', 'RMSE', 'R2']
    metric_values = [mae, rmse, max(0, r2)]
    colors_met = [COLOR_A, COLOR_W, COLOR_P]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(metric_names, metric_values, color=colors_met, width=0.5, edgecolor='none')
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(0.01, max(metric_values) * 0.02),
                f'{val:.2f}', ha='center', fontweight='bold', color=TEXT_COL, fontsize=11)
    ax.set_title('Salary Prediction Model Performance', fontsize=14, color=TEXT_COL, pad=12)
    ax.set_ylabel('Metric value', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_metrics.png'), dpi=150, facecolor=DARK_BG)
    plt.close()

    print('[INFO] Saving model artifacts...')
    joblib.dump(model, MODEL_FILE)
    metadata = {
        'feature_names': feature_names,
        'scaler': scaler,
        'label_encoders': le_map,
        'cat_cols': cat_cols,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
    }
    joblib.dump(metadata, METADATA_FILE)
    print(f'  Saved model    -> {MODEL_FILE}')
    print(f'  Saved metadata -> {METADATA_FILE}')
    print('[DONE] Employee Salary model training complete!\n')

app = FastAPI(
    title='Employee Salary Prediction API',
    description='AI-powered employee salary prediction using Random Forest Regression',
    version='1.0.0',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

_model = None
_metadata = None


def load_salary_assets():
    global _model, _metadata
    try:
        if os.path.exists(MODEL_FILE) and os.path.exists(METADATA_FILE):
            _model = joblib.load(MODEL_FILE)
            _metadata = joblib.load(METADATA_FILE)
            print('[INFO] Employee salary model assets loaded successfully.')
        else:
            print('[WARNING] Model assets not found. Run with --train first.')
    except Exception as exc:
        print(f'[ERROR] Failed to load model assets: {exc}')


load_salary_assets()


class SalaryPredictionRequest(BaseModel):
    age: float = Field(35.0, description='Employee age in years', ge=18, le=80)
    length_of_service: float = Field(5.0, description='Years at company', ge=0)
    gender_short: str = Field('M', description='Gender (M/F)')
    city_name: str = Field('Vancouver', description='Work location city')
    department_name: str = Field('Sales', description='Department name')
    job_title: str = Field('Sales Associate', description='Job title or role')
    store_name: int = Field(1, description='Store or location identifier', ge=0)
    business_unit: str = Field('STORES', description='Business unit name')
    education_level: str = Field('Bachelor', description='Highest completed education level')
    performance_rating: int = Field(3, description='Performance rating (1-5)', ge=1, le=5)
    certifications: int = Field(1, description='Number of certifications', ge=0, le=10)
    skills_level: str = Field('Intermediate', description='Job skills level')


@app.get('/', response_class=HTMLResponse)
def serve_frontend():
    if os.path.exists(FRONTEND_FILE):
        with open(FRONTEND_FILE, 'r', encoding='utf-8') as fh:
            return HTMLResponse(content=fh.read())
    return HTMLResponse(content='<h3>Frontend file not found.</h3>', status_code=404)


@app.get('/health')
def health():
    return {
        'status': 'ok' if _model is not None else 'no_model_loaded',
        'model': 'RandomForestRegressor',
        'domain': 'Employee Salary Prediction',
        'assets_exist': {
            'model': os.path.exists(MODEL_FILE),
            'metadata': os.path.exists(METADATA_FILE),
        },
        'metrics': {
            'mae': round(_metadata['mae'], 2) if _metadata else None,
            'rmse': round(_metadata['rmse'], 2) if _metadata else None,
            'r2': round(_metadata['r2'], 4) if _metadata else None,
        } if _metadata else {},
    }


@app.post('/predict')
def predict_salary(req: SalaryPredictionRequest):
    if _model is None or _metadata is None:
        raise HTTPException(status_code=503, detail='Model not loaded. Run with --train first.')

    try:
        feature_names = _metadata['feature_names']
        scaler = _metadata['scaler']
        le_map = _metadata['label_encoders']
        cat_cols = _metadata['cat_cols']

        raw = {
            'age': req.age,
            'length_of_service': req.length_of_service,
            'gender_short': req.gender_short,
            'city_name': req.city_name,
            'department_name': req.department_name,
            'job_title': req.job_title,
            'store_name': req.store_name,
            'BUSINESS_UNIT': req.business_unit,
            'education_level': req.education_level,
            'performance_rating': req.performance_rating,
            'certifications': req.certifications,
            'skills_level': req.skills_level,
        }

        for col in cat_cols:
            if col in raw:
                le = le_map[col]
                val = str(raw[col])
                if val in le.classes_:
                    raw[col] = int(le.transform([val])[0])
                else:
                    raw[col] = int(len(le.classes_) // 2)

        row = [raw.get(f, 0) for f in feature_names]
        X_input = np.array(row, dtype=float).reshape(1, -1)
        X_scaled = scaler.transform(X_input)

        prediction = float(_model.predict(X_scaled)[0])
        tree_preds = np.array([t.predict(X_scaled)[0] for t in _model.estimators_])
        mean_pred = float(np.mean(tree_preds))
        std_pred = float(np.std(tree_preds))
        range_min = round(max(0, mean_pred - 1.5 * std_pred), 2)
        range_max = round(mean_pred + 1.5 * std_pred, 2)
        score = max(0, min(100, round(100 - (std_pred / max(1.0, abs(mean_pred))) * 100, 2)))

        salary_tier = 'Executive' if prediction >= 90000 else 'Senior' if prediction >= 65000 else 'Mid' if prediction >= 45000 else 'Entry'

        return {
            'predicted_salary': round(prediction, 2),
            'estimated_salary_range': {'min': range_min, 'max': range_max},
            'salary_tier': salary_tier,
            'confidence_score': score,
            'model_metrics': {
                'mae': round(_metadata['mae'], 2),
                'rmse': round(_metadata['rmse'], 2),
                'r2': round(_metadata['r2'], 4),
            },
            'inputs': raw,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Prediction failed: {exc}')


@app.get('/results/metrics')
def get_metrics():
    if not os.path.exists(REPORT_FILE):
        raise HTTPException(status_code=404, detail='Metrics report not found. Train the model first.')
    with open(REPORT_FILE, 'r', encoding='utf-8') as fh:
        report = fh.read()
    return {
        'text_report': report,
        'plots': {
            'salary_distribution': '/results/image/salary_distribution.png',
            'correlation_heatmap': '/results/image/correlation_heatmap.png',
            'experience_vs_salary': '/results/image/experience_vs_salary.png',
            'actual_vs_predicted': '/results/image/actual_vs_predicted.png',
            'residual_distribution': '/results/image/residual_distribution.png',
            'department_salary_analysis': '/results/image/department_salary_analysis.png',
            'feature_importance': '/results/image/feature_importance.png',
            'model_metrics': '/results/image/model_metrics.png',
        },
    }


@app.get('/results/image/{image_name}')
def serve_plot(image_name: str):
    path = os.path.join(RESULTS_DIR, image_name)
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail=f"Image '{image_name}' not found.")


def _background_retrain():
    print('[RETRAIN] Starting background salary model retraining...')
    try:
        train_salary_model()
        load_salary_assets()
        print('[RETRAIN] Employee salary model retrained successfully.')
    except Exception as exc:
        print(f'[RETRAIN] Error during retraining: {exc}')


@app.post('/retrain')
def retrain_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(_background_retrain)
    return {'status': 'accepted', 'message': 'Employee salary model retraining initiated in the background.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Employee Salary Prediction – unified ML pipeline and FastAPI backend'
    )
    parser.add_argument('--train', action='store_true', help='Train the model and regenerate all results')
    parser.add_argument('--port', type=int, default=8000, help='FastAPI server port (default: 8000)')
    args = parser.parse_args()

    if args.train or not (os.path.exists(MODEL_FILE) and os.path.exists(METADATA_FILE)):
        try:
            train_salary_model()
            load_salary_assets()
        except Exception as exc:
            print(f'[ERROR] Training failed: {exc}')
            sys.exit(1)

    import uvicorn
    print(f'[INFO] Starting Employee Salary Prediction API server on port {args.port}...')
    print(f'[INFO] Open your browser at: http://127.0.0.1:{args.port}')
    uvicorn.run(
        'employee_salary_prediction:app',
        host='127.0.0.1',
        port=args.port,
        reload=False,
    )
