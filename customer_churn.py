# -*- coding: utf-8 -*-
"""Customer Churn Prediction Pipeline

This script trains a Random Forest classifier to predict customer churn.
It includes data preprocessing, model training, evaluation, and visualization.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             precision_score, recall_score, f1_score, roc_curve, auc, 
                             roc_auc_score, precision_recall_curve)
from sklearn.ensemble import RandomForestClassifier

# Configure visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define results directory
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'customer_churn_prediction', 'results')
os.makedirs(results_dir, exist_ok=True)

# Load dataset from customer churn data folder
churn_data_path = r"C:\Users\peepl\Downloads\customer churn\WA_Fn-UseC_-Telco-Customer-Churn.csv"

print(f"Loading customer churn dataset from: {churn_data_path}")
df = pd.read_csv(churn_data_path)

print(f"\n=== Customer Churn Dataset Overview ===")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Remove customerID as it's not useful for training
df = df.drop('customerID', axis=1)

# Convert to numeric values, coerce errors to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')

# Extract target variable - Customer Churn
y = df['Churn']
df = df.drop('Churn', axis=1)

print(f"\nChurn distribution before balancing:")
print(y.value_counts())

# One-hot encode categorical features
df = pd.get_dummies(df, drop_first=True)

# Fill null/NaN values in numeric columns with mean
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

# Remove duplicate records
print(f"\nDuplicate records: {df.duplicated().sum()}")
df = df.drop_duplicates()

# Remove outliers in numeric columns (values > 3500 or < 0)
for col in ['TotalCharges', 'MonthlyCharges']:
    median_value = df[col].median()
    df[col] = df[col].apply(lambda x: median_value if x > 3500 or x < 0 else x)

# Prepare features and align with target
X = df.copy()
y_aligned = y.loc[X.index]

print(f"\nFinal dataset shape: {X.shape}")
print(f"Features: {X.columns.tolist()}")

# Train-test split with stratification
x_train, x_test, y_train, y_test = train_test_split(
    X, y_aligned, stratify=y_aligned, test_size=0.2, random_state=42
)

print(f"\n=== Train-Test Split ===")
print(f"Training set: {x_train.shape}")
print(f"Test set: {x_test.shape}")
print(f"Training target: {y_train.shape}")
print(f"Test target: {y_test.shape}")

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

print(f"\n=== After SMOTE Balancing ===")
print(f"Training set after SMOTE: {x_train_smote.shape}")
print("Churn distribution after SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# Scale features
scaler = StandardScaler()
scaler.fit(x_train_smote)

x_train_scaled = scaler.transform(x_train_smote)
x_test_scaled = scaler.transform(x_test)

x_train_scaled = pd.DataFrame(x_train_scaled, columns=X.columns)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=X.columns)

print("\nFeature scaling completed.")

# ============= Model Training =============
print(f"\n=== Training Random Forest Classifier ===")

rf_model = RandomForestClassifier(
    n_estimators=300,
    criterion='gini',
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1)

rf_model.fit(x_train_scaled, y_train_smote)

# Make predictions
y_pred = rf_model.predict(x_test_scaled)
y_pred_proba = rf_model.predict_proba(x_test_scaled)[:, 1]

# ============= Model Evaluation =============
print(f"\n=== Model Performance Metrics ===")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='Yes')
recall = recall_score(y_test, y_pred, pos_label='Yes')
f1 = f1_score(y_test, y_pred, pos_label='Yes')
roc_auc = roc_auc_score(y_test == 'Yes', y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ============= Visualizations =============
print(f"\n=== Generating Visualizations ===")

# 1. Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title('Customer Churn - Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
print(f"Saved: confusion_matrix.png")
plt.close()

# 2. Feature Importance
importance = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importance
}).sort_values(by='importance', ascending=False).head(15)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance_df['feature'], feature_importance_df['importance'], color='steelblue')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 15 Features - Feature Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
print(f"Saved: feature_importance.png")
plt.close()

# 3. Class Distribution
plt.figure(figsize=(8, 6))
churn_counts = y_test.value_counts()
colors = ['#2ecc71', '#e74c3c']
plt.bar(churn_counts.index, churn_counts.values, color=colors, edgecolor='black', linewidth=1.5)
plt.xlabel('Customer Churn Status', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Customer Churn Distribution (Test Set)', fontsize=14, fontweight='bold')
plt.xticks(rotation=0)
for i, v in enumerate(churn_counts.values):
    plt.text(i, v + 5, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
print(f"Saved: class_distribution.png")
plt.close()

# 4. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test == 'Yes', y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Customer Churn - ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
print(f"Saved: roc_curve.png")
plt.close()

# 5. Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test == 'Yes', y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, color='steelblue', lw=2, label='Precision-Recall Curve')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Customer Churn - Precision-Recall Curve', fontsize=14, fontweight='bold')
plt.legend(loc="upper right", fontsize=11)
plt.grid(alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
print(f"Saved: precision_recall_curve.png")
plt.close()

# 6. Correlation Heatmap (top features)
top_features = feature_importance_df.head(10)['feature'].tolist()
correlation_matrix = X[top_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap - Top 10 Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
print(f"Saved: correlation_heatmap.png")
plt.close()

# ============= Save Model and Preprocessors =============
print(f"\n=== Saving Model Artifacts ===")

# Save Random Forest model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'customer_churn_model.joblib')
joblib.dump(rf_model, model_path)
print(f"Saved model to: {model_path}")

# Save scaler
scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'customer_churn_preprocessors.joblib')
joblib.dump(scaler, scaler_path)
print(f"Saved preprocessor (scaler) to: {scaler_path}")

# Save feature names
features_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'customer_churn_features.joblib')
joblib.dump(X.columns.tolist(), features_path)
print(f"Saved feature names to: {features_path}")

# ============= Save Results Report =============
print(f"\n=== Saving Results Report ===")

results_report = f"""
=== CUSTOMER CHURN PREDICTION - RESULTS REPORT ===

Dataset Information:
  Total Records: {len(df) + len(df[df.index.isin(y.index)])}
  Features: {X.shape[1]}
  Target Classes: No Churn / Churn

Data Split:
  Training Set: {x_train.shape[0]} samples
  Test Set: {x_test.shape[0]} samples

Model Configuration:
  Algorithm: Random Forest Classifier
  n_estimators: 300
  max_depth: 10
  criterion: gini
  class_weight: balanced

Performance Metrics (Test Set):
  Accuracy: {accuracy:.4f}
  Precision: {precision:.4f}
  Recall: {recall:.4f}
  F1-Score: {f1:.4f}
  ROC-AUC: {roc_auc:.4f}

Confusion Matrix:
  True Negatives: {cm[0, 0]}
  False Positives: {cm[0, 1]}
  False Negatives: {cm[1, 0]}
  True Positives: {cm[1, 1]}

Class Distribution (Training):
  No Churn: {(y_train == 'No').sum()}
  Churn: {(y_train == 'Yes').sum()}

Class Distribution (After SMOTE):
  No Churn: {(y_train_smote == 'No').sum()}
  Churn: {(y_train_smote == 'Yes').sum()}

Top 5 Important Features:
{feature_importance_df.head(5).to_string(index=False)}

Generated Visualizations:
  - confusion_matrix.png
  - feature_importance.png
  - class_distribution.png
  - roc_curve.png
  - precision_recall_curve.png
  - correlation_heatmap.png

Model Artifacts:
  - customer_churn_model.joblib
  - customer_churn_preprocessors.joblib
  - customer_churn_features.joblib

Generated on: {pd.Timestamp.now()}
"""

results_file = os.path.join(results_dir, 'accuracy_results.txt')
with open(results_file, 'w') as f:
    f.write(results_report)

print(f"Saved results report to: {results_file}")

print("\n=== PIPELINE EXECUTION COMPLETED SUCCESSFULLY ===\n")





