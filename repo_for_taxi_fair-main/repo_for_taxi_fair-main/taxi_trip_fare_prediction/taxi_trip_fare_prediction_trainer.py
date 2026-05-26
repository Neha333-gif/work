# -*- coding: utf-8 -*-
"""taxi_trip_fare_prediction_trainer.py

Train a taxi trip fare regression model, generate evaluation metrics and save visualizations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_taxi_trip_fare_model():
    print('=' * 60)
    print('STARTING TAXI TRIP FARE MODEL TRAINING')
    print('=' * 60)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, '..', 'taxi_trip_fare_prediction_dataset', 'taxi_trip_fare_train.csv')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Please ensure taxi_trip_fare_train.csv is available in taxi_trip_fare_prediction_dataset."
        )

    print('[INFO] Loading taxi trip fare dataset...')
    df = pd.read_csv(DATA_PATH)
    print(f'  Dataset shape: {df.shape}')

    expected_columns = [
        'trip_duration',
        'distance_traveled',
        'num_of_passengers',
        'fare',
        'surge_applied',
        'total_fare'
    ]
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f'Missing expected columns in dataset: {missing}')

    feature_columns = [
        'trip_duration',
        'distance_traveled',
        'num_of_passengers',
        'fare',
        'surge_applied'
    ]
    target_column = 'total_fare'

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f'  Train shape: {X_train.shape}, Test shape: {X_test.shape}')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), feature_columns)
        ],
        remainder='drop'
    )

    model_candidates = {
        'Linear Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ]),
        'Decision Tree': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', DecisionTreeRegressor(max_depth=8, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ))
        ])
    }

    evaluation_results = {}

    for name, pipeline in model_candidates.items():
        print(f'[INFO] Training {name}...')
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        evaluation_results[name] = {
            'pipeline': pipeline,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'predictions': predictions
        }
        print(f'  {name} MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}')

    best_model_name = min(evaluation_results, key=lambda name: evaluation_results[name]['rmse'])
    best_result = evaluation_results[best_model_name]
    best_pipeline = best_result['pipeline']

    pipeline_path = os.path.join(BASE_DIR, 'taxi_trip_fare_prediction_pipeline.joblib')
    metadata_path = os.path.join(BASE_DIR, 'taxi_trip_fare_prediction_metadata.joblib')

    joblib.dump(best_pipeline, pipeline_path)

    metadata = {
        'model_type': best_model_name,
        'feature_names': feature_columns,
        'mae': float(best_result['mae']),
        'mse': float(best_result['mse']),
        'rmse': float(best_result['rmse']),
        'r2': float(best_result['r2']),
        'all_metrics': {
            name: {
                'mae': float(info['mae']),
                'mse': float(info['mse']),
                'rmse': float(info['rmse']),
                'r2': float(info['r2'])
            }
            for name, info in evaluation_results.items()
        }
    }
    joblib.dump(metadata, metadata_path)

    print(f'[OK] Saved best model pipeline ({best_model_name}) to {pipeline_path}')
    print(f'[OK] Saved model metadata to {metadata_path}')

    sns.set_theme(style='whitegrid')

    plt.figure(figsize=(10, 6))
    sns.histplot(df[target_column], bins=28, kde=True, color='#1f77b4')
    plt.title('Taxi Fare Distribution')
    plt.xlabel('Total Fare')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fare_distribution.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 8))
    corr = df[feature_columns + [target_column]].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Taxi Fare Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'correlation_heatmap.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    y_test_series = y_test.reset_index(drop=True)
    predictions = pd.Series(best_result['predictions'], name='predicted_total_fare')
    sns.scatterplot(x=y_test_series, y=predictions, alpha=0.6)
    plt.plot([y_test_series.min(), y_test_series.max()], [y_test_series.min(), y_test_series.max()], color='gray', linestyle='--')
    plt.title('Actual vs Predicted Total Fare')
    plt.xlabel('Actual Total Fare')
    plt.ylabel('Predicted Total Fare')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'actual_vs_predicted.png'), dpi=150)
    plt.close()

    residuals = y_test_series - predictions
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=28, kde=True, color='#2ca02c')
    plt.title('Prediction Residuals Distribution')
    plt.xlabel('Actual - Predicted Fare')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'residuals_distribution.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['distance_traveled'], y=df[target_column], hue=df['surge_applied'], palette=['#1f77b4', '#ff7f0e'], alpha=0.7)
    plt.title('Trip Distance vs Total Fare')
    plt.xlabel('Distance Traveled')
    plt.ylabel('Total Fare')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'distance_vs_fare.png'), dpi=150)
    plt.close()

    try:
        rf_pipeline = evaluation_results['Random Forest']['pipeline']
        rf_model = rf_pipeline.named_steps['regressor']
        importances = rf_model.feature_importances_
        fi = pd.Series(importances, index=feature_columns).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        plt.barh(fi.index, fi.values, color='#4f46e5')
        plt.title('Taxi Fare Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'), dpi=150)
        plt.close()
    except Exception as exc:
        print(f'  [WARNING] Feature importance plot was not created: {exc}')

    plt.figure(figsize=(10, 6))
    model_names = list(evaluation_results.keys())
    rmse_scores = [evaluation_results[name]['rmse'] for name in model_names]
    plt.barh(model_names, rmse_scores, color=['#636efa', '#ff7f0e', '#2ca02c'])
    plt.title('Model RMSE Comparison')
    plt.xlabel('RMSE')
    plt.ylabel('Model')
    for i, score in enumerate(rmse_scores):
        plt.text(score + 0.5, i, f'{score:.2f}', va='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_performance.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(predictions, bins=28, kde=True, color='#d62728')
    plt.title('Predicted Total Fare Distribution')
    plt.xlabel('Predicted Total Fare')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'prediction_distribution.png'), dpi=150)
    plt.close()

    report_text = f"""Taxi Trip Fare Prediction Model Evaluation Report
=================================================

Dataset shape: {df.shape}
Best model: {best_model_name}

Best model metrics:
  MAE : {best_result['mae']:.4f}
  MSE : {best_result['mse']:.4f}
  RMSE: {best_result['rmse']:.4f}
  R2  : {best_result['r2']:.4f}

Detailed model comparison:
"""
    for name, info in evaluation_results.items():
        report_text += (
            f"\n{name}: MAE={info['mae']:.4f}, MSE={info['mse']:.4f}, "
            f"RMSE={info['rmse']:.4f}, R2={info['r2']:.4f}\n"
        )

    report_text += f"\nFeature columns:\n{', '.join(feature_columns)}\n"

    with open(os.path.join(RESULTS_DIR, 'taxi_trip_fare_prediction_evaluation.txt'), 'w', encoding='utf-8') as report_file:
        report_file.write(report_text)

    sample_predictions = X_test.copy().reset_index(drop=True)
    sample_predictions['actual_total_fare'] = y_test_series.values
    sample_predictions['predicted_total_fare'] = predictions.values
    sample_predictions.to_csv(os.path.join(RESULTS_DIR, 'sample_taxi_trip_fare_predictions.csv'), index=False)

    print('[OK] Evaluation report saved successfully.')
    print('[OK] Sample predictions saved successfully.')
    print('\n' + '=' * 60)
    print('TAXI TRIP FARE MODEL TRAINING COMPLETE')
    print('=' * 60 + '\n')


if __name__ == '__main__':
    train_taxi_trip_fare_model()
