# -*- coding: utf-8 -*-
"""credit_risk_analysis_trainer.py

Train a credit risk classification model, generate evaluation metrics and save visualizations.
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


def train_credit_risk_model():
    print('=' * 60)
    print('STARTING CREDIT RISK ANALYSIS MODEL TRAINING')
    print('=' * 60)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, '..', 'credit_risk_analysis_dataset', 'bankloans.csv')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Please ensure bankloans.csv is available in credit_risk_analysis_dataset."
        )

    print('[INFO] Loading credit risk dataset...')
    df = pd.read_csv(DATA_PATH)
    print(f'  Dataset shape: {df.shape}')
    print(f'  Columns: {list(df.columns)}')

    # Drop rows with missing target value
    df = df.dropna(subset=['default'])
    print(f'  Shape after removing null targets: {df.shape}')

    # Identify numerical and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'default' in numeric_cols:
        numeric_cols.remove('default')
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f'  Numeric features: {numeric_cols}')
    print(f'  Categorical features: {categorical_cols}')

    # Prepare features and target
    target_column = 'default'
    feature_columns = numeric_cols + categorical_cols

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Handle any missing values in features
    X = X.fillna(X.mean(numeric_only=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'  Train shape: {X_train.shape}, Test shape: {X_test.shape}')
    print(f'  Train target distribution: {y_train.value_counts().to_dict()}')
    print(f'  Test target distribution: {y_test.value_counts().to_dict()}')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols + categorical_cols)
        ],
        remainder='drop'
    )

    model_candidates = {
        'Logistic Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'Decision Tree': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(max_depth=8, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
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
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
        except:
            roc_auc = 0.0
        
        evaluation_results[name] = {
            'pipeline': pipeline,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'predictions': predictions
        }
        print(f'  {name} Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

    best_model_name = max(evaluation_results, key=lambda name: evaluation_results[name]['f1'])
    best_result = evaluation_results[best_model_name]
    best_pipeline = best_result['pipeline']

    pipeline_path = os.path.join(BASE_DIR, 'credit_risk_analysis_pipeline.joblib')
    metadata_path = os.path.join(BASE_DIR, 'credit_risk_analysis_metadata.joblib')

    joblib.dump(best_pipeline, pipeline_path)

    metadata = {
        'model_type': best_model_name,
        'feature_names': feature_columns,
        'label_encoders': label_encoders,
        'accuracy': float(best_result['accuracy']),
        'precision': float(best_result['precision']),
        'recall': float(best_result['recall']),
        'f1': float(best_result['f1']),
        'roc_auc': float(best_result['roc_auc']),
        'all_metrics': {
            name: {
                'accuracy': float(info['accuracy']),
                'precision': float(info['precision']),
                'recall': float(info['recall']),
                'f1': float(info['f1']),
                'roc_auc': float(info['roc_auc'])
            }
            for name, info in evaluation_results.items()
        }
    }
    joblib.dump(metadata, metadata_path)

    print(f'[OK] Saved best model pipeline ({best_model_name}) to {pipeline_path}')
    print(f'[OK] Saved model metadata to {metadata_path}')

    sns.set_theme(style='whitegrid')

    # Default distribution
    plt.figure(figsize=(10, 6))
    default_counts = df[target_column].value_counts()
    plt.bar(['Non-Default', 'Default'], [default_counts[0], default_counts[1]], color=['#2ca02c', '#d62728'])
    plt.title('Credit Risk Distribution')
    plt.ylabel('Count')
    plt.xlabel('Default Status')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'default_distribution.png'), dpi=150)
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    numeric_data = df[numeric_cols + [target_column]].corr()
    sns.heatmap(numeric_data, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={'label': 'Correlation'})
    plt.title('Credit Risk Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'correlation_heatmap.png'), dpi=150)
    plt.close()

    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, best_result['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Non-Default', 'Default'], yticklabels=['Non-Default', 'Default'])
    plt.title('Confusion Matrix - Best Model')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # Prediction distribution
    plt.figure(figsize=(10, 6))
    predictions_series = pd.Series(best_result['predictions'])
    pred_counts = predictions_series.value_counts()
    plt.bar(['Non-Default', 'Default'], [pred_counts[0], pred_counts[1]], color=['#636efa', '#ff7f0e'])
    plt.title('Model Prediction Distribution')
    plt.ylabel('Count')
    plt.xlabel('Predicted Default Status')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'prediction_distribution.png'), dpi=150)
    plt.close()

    # Feature importance (for tree-based models)
    try:
        rf_pipeline = evaluation_results['Random Forest']['pipeline']
        rf_model = rf_pipeline.named_steps['classifier']
        importances = rf_model.feature_importances_
        fi = pd.Series(importances, index=feature_columns).sort_values(ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        plt.barh(fi.index, fi.values, color='#4f46e5')
        plt.title('Top 10 Feature Importances (Random Forest)')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'), dpi=150)
        plt.close()
    except Exception as exc:
        print(f'  [WARNING] Feature importance plot was not created: {exc}')

    # Model performance comparison
    plt.figure(figsize=(10, 6))
    model_names = list(evaluation_results.keys())
    f1_scores = [evaluation_results[name]['f1'] for name in model_names]
    plt.barh(model_names, f1_scores, color=['#636efa', '#ff7f0e', '#2ca02c'])
    plt.title('Model F1-Score Comparison')
    plt.xlabel('F1-Score')
    plt.ylabel('Model')
    for i, score in enumerate(f1_scores):
        plt.text(score + 0.01, i, f'{score:.3f}', va='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_performance.png'), dpi=150)
    plt.close()

    # Evaluation report
    report_text = f"""Credit Risk Analysis Model Evaluation Report
=============================================

Dataset shape: {df.shape}
Best model: {best_model_name}

Best model metrics:
  Accuracy : {best_result['accuracy']:.4f}
  Precision: {best_result['precision']:.4f}
  Recall   : {best_result['recall']:.4f}
  F1-Score : {best_result['f1']:.4f}
  ROC-AUC  : {best_result['roc_auc']:.4f}

Detailed model comparison:
"""
    for name, info in evaluation_results.items():
        report_text += (
            f"\n{name}:\n"
            f"  Accuracy : {info['accuracy']:.4f}\n"
            f"  Precision: {info['precision']:.4f}\n"
            f"  Recall   : {info['recall']:.4f}\n"
            f"  F1-Score : {info['f1']:.4f}\n"
            f"  ROC-AUC  : {info['roc_auc']:.4f}\n"
        )

    report_text += f"\nFeature columns:\n{', '.join(feature_columns)}\n"

    with open(os.path.join(RESULTS_DIR, 'credit_risk_analysis_evaluation.txt'), 'w', encoding='utf-8') as report_file:
        report_file.write(report_text)

    sample_predictions = X_test.copy().reset_index(drop=True)
    sample_predictions['actual_default'] = y_test.values
    sample_predictions['predicted_default'] = best_result['predictions']
    sample_predictions.to_csv(os.path.join(RESULTS_DIR, 'sample_credit_risk_predictions.csv'), index=False)

    print('[OK] Evaluation report saved successfully.')
    print('[OK] Sample predictions saved successfully.')
    print('\n' + '=' * 60)
    print('CREDIT RISK ANALYSIS MODEL TRAINING COMPLETE')
    print('=' * 60 + '\n')


if __name__ == '__main__':
    train_credit_risk_model()
