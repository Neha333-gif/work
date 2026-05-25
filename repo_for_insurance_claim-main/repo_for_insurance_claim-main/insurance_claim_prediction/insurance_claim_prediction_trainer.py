import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    classification_report,
    roc_curve
)


def train_insurance_claim_model():
    print("="*60)
    print("STARTING INSURANCE CLAIM MODEL TRAINING")
    print("="*60)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, '..', 'insurance_claim_prediction_dataset', 'insurance_claims.csv')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please ensure insurance_claims.csv is in the insurance_claim_prediction_dataset folder.")

    print("[INFO] Loading insurance claim dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Shape: {df.shape}")

    if 'customer_id' in df.columns:
        df = df.drop(columns=['customer_id'])

    if 'insuranceclaim' not in df.columns:
        raise ValueError("Target column 'insuranceclaim' not found in dataset.")

    X = df.drop(columns=['insuranceclaim'])
    y_raw = df['insuranceclaim']
    if y_raw.dtype == 'object':
        y = LabelEncoder().fit_transform(y_raw)
    else:
        y = y_raw.astype(int).values

    target_classes = ['No Claim', 'Claim']

    cat_cols = [c for c in ['region'] if c in X.columns]
    num_cols = [c for c in ['age', 'bmi', 'children', 'charges', 'sex', 'smoker'] if c in X.columns]

    print(f"[INFO] Categorical features: {cat_cols}")
    print(f"[INFO] Numeric features: {num_cols}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ], remainder='drop')

    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=500, class_weight='balanced', random_state=42))
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)
    lr_auc = roc_auc_score(y_test, lr_pipeline.predict_proba(X_test)[:, 1])

    dt_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(max_depth=8, class_weight='balanced', random_state=42))
    ])
    dt_pipeline.fit(X_train, y_train)
    dt_pred = dt_pipeline.predict(X_test)
    dt_auc = roc_auc_score(y_test, dt_pipeline.predict_proba(X_test)[:, 1])

    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])
    rf_pipeline.fit(X_train, y_train)
    y_pred = rf_pipeline.predict(X_test)
    y_pred_proba = rf_pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_classes, zero_division=0)

    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC AUC  : {roc_auc:.4f}")

    pipeline_path = os.path.join(BASE_DIR, 'insurance_claim_prediction_pipeline.joblib')
    metadata_path = os.path.join(BASE_DIR, 'insurance_claim_prediction_metadata.joblib')
    joblib.dump(rf_pipeline, pipeline_path)

    metadata = {
        'model_type': 'Random Forest Classifier',
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': float(roc_auc),
        'feature_names': X.columns.tolist(),
        'categorical_features': cat_cols,
        'numeric_features': num_cols,
        'target_classes': target_classes
    }
    joblib.dump(metadata, metadata_path)

    print(f"[OK] Saved model pipeline to {pipeline_path}")
    print(f"[OK] Saved model metadata to {metadata_path}")

    sns.set_theme(style='whitegrid')

    plt.figure(figsize=(8, 6))
    sns.countplot(x=df['insuranceclaim'], palette=['#ff7f0e', '#1f77b4'])
    plt.title('Insurance Claim Class Distribution')
    plt.xlabel('Insurance Claim')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'claim_class_distribution.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    numeric_df = df[num_cols + ['insuranceclaim']].copy()
    numeric_df['insuranceclaim'] = y
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, square=True)
    plt.title('Insurance Feature Correlations')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'correlation_heatmap.png'), dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#1f77b4', lw=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('Insurance Claim ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_classes, yticklabels=target_classes)
    plt.title('Insurance Claim Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()

    try:
        classifier = rf_pipeline.named_steps['classifier']
        ohe = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat']
        cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
        all_feature_names = num_cols + cat_feature_names
        importances = classifier.feature_importances_
        fi_series = pd.Series(importances, index=all_feature_names).sort_values(ascending=False).head(20)

        plt.figure(figsize=(10, 8))
        sns.barplot(x=fi_series.values, y=fi_series.index, palette='viridis')
        plt.title('Top Insurance Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f"  [WARNING] Could not generate feature importance plot: {e}")

    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, color='#d62728', lw=2)
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'precision_recall_curve.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.histplot(y_pred_proba, bins=20, kde=True, color='#2ca02c')
    plt.title('Claim Approval Probability Distribution')
    plt.xlabel('Approval Probability')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'claim_probability_distribution.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest']
    scores = [accuracy_score(y_test, lr_pred), accuracy_score(y_test, dt_pred), accuracy]
    sns.barplot(x=model_names, y=scores, palette=['#636efa', '#ff7f0e', '#2ca02c'])
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, score in enumerate(scores):
        plt.text(i, score + 0.02, f'{score:.3f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison.png'), dpi=150)
    plt.close()

    report_text = f"""Insurance Claim Prediction Model Evaluation Report
=================================================

Pipeline: Random Forest Classification for insurance claim prediction

Dataset shape: {df.shape}
Target distribution:
{df['insuranceclaim'].value_counts().to_string()}

Random Forest Performance:
  Accuracy : {accuracy:.4f}
  Precision: {precision:.4f}
  Recall   : {recall:.4f}
  F1 Score : {f1:.4f}
  ROC AUC  : {roc_auc:.4f}

Classification Report:
{report}

Comparison Models:
  Logistic Regression ROC AUC: {lr_auc:.4f}
  Decision Tree      ROC AUC: {dt_auc:.4f}
"""
    with open(os.path.join(RESULTS_DIR, 'insurance_claim_prediction_evaluation.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)

    sample_df = df.head(100).copy()
    sample_df['predicted_claim_status'] = [target_classes[int(p)] for p in rf_pipeline.predict(sample_df.drop(columns=['insuranceclaim']))]
    sample_df['claim_approval_probability'] = rf_pipeline.predict_proba(sample_df.drop(columns=['insuranceclaim']))[:, 1]
    sample_df.to_csv(os.path.join(RESULTS_DIR, 'sample_insurance_claim_predictions.csv'), index=False)

    print("[OK] Evaluation report saved successfully.")
    print("[OK] Sample predictions saved successfully.")
    print("\n" + "="*60)
    print("INSURANCE CLAIM MODEL TRAINING COMPLETE")
    print("="*60)


if __name__ == '__main__':
    train_insurance_claim_model()

