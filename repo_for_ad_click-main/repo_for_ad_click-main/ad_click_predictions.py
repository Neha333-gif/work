# -*- coding: utf-8 -*-
"""
Ad Click Prediction Model Trainer
Automatically processes banner interaction data, trains a lightweight Random Forest Classifier,
generates evaluation visualizations, and saves model assets.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. Setup Directories and Load Data
# ──────────────────────────────────────────────
# Use relative paths or check script location for robustness
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "banner_interactions.csv")

# Set up results directory
RESULTS_DIR = os.path.join(base_dir, "ad_click_prediction", "results")
if not os.path.exists(os.path.join(base_dir, "ad_click_prediction")):
    # Fallback to local results/ if the folder hasn't been renamed yet
    RESULTS_DIR = os.path.join(base_dir, "customer_segmentaion", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"[INFO] Loading dataset from {data_path}...")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset banner_interactions.csv not found at {data_path}")

df = pd.read_csv(data_path)
print(f"  Dataset shape : {df.shape}")
print(f"  Columns       : {list(df.columns)}")

# ──────────────────────────────────────────────
# 2. Preprocessing & Bug Fixes
# ──────────────────────────────────────────────
print("[INFO] Preprocessing features...")

# Fit LabelEncoder on the target
clicks_encoder = LabelEncoder()
df['clicks'] = clicks_encoder.fit_transform(df['clicks'])
y = df['clicks']

# Prepare feature matrix (drop target and high cardinality ID)
x = df.drop(['clicks', 'user_id'], axis=1)

# Store original column data types to fix the imputer coercion bug
original_dtypes = x.dtypes

# Apply imputer (imputer returns a numpy array, which coerces all to object/string)
imputer = SimpleImputer(strategy='most_frequent')
x_imputed = imputer.fit_transform(x)
x = pd.DataFrame(x_imputed, columns=x.columns)

# Cast columns back to their original datatypes to prevent standard numerical variables from being label encoded
for col in x.columns:
    x[col] = x[col].astype(original_dtypes[col])

label_encoders = {}
scalers = {}

# Process each column correctly based on its true datatype
for col in x.columns:
    if x[col].dtype == 'object':
        le = LabelEncoder()
        x[col] = le.fit_transform(x[col])
        label_encoders[col] = le
        print(f"  Categorical column '{col}' encoded with LabelEncoder.")
    else:
        # Cast to numeric to prevent scaling issues
        x[col] = pd.to_numeric(x[col])
        scaler = StandardScaler()
        x[col] = scaler.fit_transform(x[col].values.reshape(-1, 1))
        scalers[col] = scaler
        print(f"  Numerical column '{col}' scaled with StandardScaler.")

# ──────────────────────────────────────────────
# 3. Train / Test Split + SMOTE
# ──────────────────────────────────────────────
print("[INFO] Oversampling minority classes using SMOTE...")
# Use k_neighbors=4 as in original code, setting k_neighbors to accommodate smaller classes
smote = SMOTE(random_state=42, k_neighbors=4)
x_resampled, y_resampled = smote.fit_resample(x, y)

print(f"  Original class distribution: {y.value_counts().to_dict()}")
print(f"  Resampled class distribution: {y_resampled.value_counts().to_dict()}")

x_train, x_test, y_train, y_test = train_test_split(
    x_resampled, y_resampled, test_size=0.2, random_state=42
)
print(f"  Train set size: {x_train.shape[0]}, Test set size: {x_test.shape[0]}")

# ──────────────────────────────────────────────
# 4. Train Lightweight Random Forest Classifier
# ──────────────────────────────────────────────
print("[INFO] Training lightweight Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=50,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(x_train, y_train)

# ──────────────────────────────────────────────
# 5. Make Predictions and Evaluate
# ──────────────────────────────────────────────
print("[INFO] Evaluating model...")
y_pred = rf_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
report = classification_report(y_test, y_pred, target_names=[str(c) for c in clicks_encoder.classes_])

print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")

# Save evaluation report to results folder
metrics_text = f"""AI Ad Click Prediction - Model Evaluation Report
=================================================
Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)
F1-Score  : {f1:.4f}
Precision : {precision:.4f}
Recall    : {recall:.4f}

Classification Report:
{report}
"""
with open(os.path.join(RESULTS_DIR, "accuracy_results.txt"), "w", encoding="utf-8") as f:
    f.write(metrics_text)
print(f"[INFO] Saved metrics report -> {os.path.join(RESULTS_DIR, 'accuracy_results.txt')}")

# ──────────────────────────────────────────────
# 6. Data Visualizations
# ──────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
print("[INFO] Generating data visualizations...")

# 6a. Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=clicks_encoder.classes_, yticklabels=clicks_encoder.classes_)
plt.xlabel('Predicted Clicks')
plt.ylabel('Actual Clicks')
plt.title('Confusion Matrix – Ad Click Prediction')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=120)
plt.close()
print(f"  Saved -> {os.path.join(RESULTS_DIR, 'confusion_matrix.png')}")

# 6b. Feature Importance
plt.figure(figsize=(10, 6))
importances = rf_model.feature_importances_
feature_names = x.columns
indices = np.argsort(importances)[::-1]
sns.barplot(x=importances[indices], y=feature_names[indices], palette="viridis")
plt.xlabel('Relative Feature Importance')
plt.ylabel('Feature Name')
plt.title('Feature Importance – Ad Click Prediction')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=120)
plt.close()
print(f"  Saved -> {os.path.join(RESULTS_DIR, 'feature_importance.png')}")

# 6c. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_df = x.copy()
corr_df['clicks'] = y
corr = corr_df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "correlation_heatmap.png"), dpi=120)
plt.close()
print(f"  Saved -> {os.path.join(RESULTS_DIR, 'correlation_heatmap.png')}")

# 6d. Class Distribution
plt.figure(figsize=(8, 5))
original_class_counts = df['clicks'].value_counts().sort_index()
sns.barplot(x=clicks_encoder.inverse_transform(original_class_counts.index), y=original_class_counts.values, palette="rocket")
plt.xlabel('Number of Clicks')
plt.ylabel('Number of Interactions')
plt.title('Original Clicks Class Distribution')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "class_distribution.png"), dpi=120)
plt.close()
print(f"  Saved -> {os.path.join(RESULTS_DIR, 'class_distribution.png')}")

# 6e. ROC Curve (One-vs-Rest for multiclass)
y_test_bin = label_binarize(y_test, classes=np.unique(y_resampled))
n_classes = y_test_bin.shape[1]
y_score = rf_model.predict_proba(x_test)

plt.figure(figsize=(10, 8))
if n_classes > 2:
    colors = sns.color_palette("husl", n_classes)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'Class {clicks_encoder.classes_[i]} ROC curve (area = {roc_auc:.2f})')
elif n_classes == 2 or len(np.unique(y_resampled)) == 2:
    fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=120)
plt.close()
print(f"  Saved -> {os.path.join(RESULTS_DIR, 'roc_curve.png')}")

# ──────────────────────────────────────────────
# 7. Save Model and Preprocessors
# ──────────────────────────────────────────────
print("[INFO] Saving model and preprocessors...")

# We save the Random Forest model
joblib.dump(rf_model, os.path.join(base_dir, "ad_click_model.joblib"))
print("  Saved model -> ad_click_model.joblib")

# Save all encoders, scalers, and dynamic lists needed by backend and frontend
preprocessor_data = {
    "label_encoders": label_encoders,
    "scalers": scalers,
    "clicks_encoder": clicks_encoder,
    "unique_banners": sorted(list(df['banner_id'].unique())),
    "date_range": {
        "min": str(df['event_date'].min()),
        "max": str(df['event_date'].max()),
        "all_dates": sorted(list(df['event_date'].unique()))
    }
}
joblib.dump(preprocessor_data, os.path.join(base_dir, "ad_click_preprocessors.joblib"))
print("  Saved preprocessors -> ad_click_preprocessors.joblib")

print("\n[DONE] Model training and asset generation completed successfully!")