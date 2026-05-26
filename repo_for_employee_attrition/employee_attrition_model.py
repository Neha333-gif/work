# -*- coding: utf-8 -*-
"""
employee_attrition_model.py
Standalone Employee Attrition Prediction ML Script.

Trains a RandomForestClassifier on the MFG10YearTerminationData dataset,
evaluates performance, generates visualizations, and saves the model artifact.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

DATA_PATH   = os.path.join("employee_attrition_prediction_data", "MFG10YearTerminationData.csv")
RESULTS_DIR = "results"
MODEL_FILE  = "employee_attrition_model.joblib"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Load Dataset
# ──────────────────────────────────────────────────────────────────────────────

print("[INFO] Loading dataset...")
df = pd.read_csv(DATA_PATH)

print(f"  Dataframe shape  : {df.shape}")
print(f"  Columns          : {list(df.columns)}")
print(f"\n  First 3 rows:\n{df.head(3)}\n")
print(f"  Null values:\n{df.isnull().sum()}\n")
print(f"  Target distribution:\n{df['STATUS'].value_counts()}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

print("[INFO] Preprocessing data...")

# Drop columns that would cause label leakage or are non-informative identifiers
drop_cols = [
    'EmployeeID',
    'recorddate_key',
    'birthdate_key',
    'orighiredate_key',
    'terminationdate_key',   # leakage: only filled on termination
    'termreason_desc',       # leakage: only filled on termination
    'termtype_desc',         # leakage: only filled on termination
    'gender_full',           # redundant with gender_short
    'STATUS_YEAR',           # temporal info not useful for prediction
]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# Encode target: ACTIVE=0, TERMINATED=1
df['STATUS'] = (df['STATUS'] == 'TERMINATED').astype(int)

# Encode categorical features
cat_cols = df.select_dtypes(include='object').columns.tolist()
le_map = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_map[col] = le

# Fill any remaining nulls
df.fillna(df.median(numeric_only=True), inplace=True)
df.drop_duplicates(inplace=True)

print(f"  Processed shape  : {df.shape}")
print(f"  Features         : {[c for c in df.columns if c != 'STATUS']}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Feature / Target Split
# ──────────────────────────────────────────────────────────────────────────────

x = df.drop('STATUS', axis=1)
y = df['STATUS']

feature_names = list(x.columns)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Training samples : {x_train.shape[0]}")
print(f"  Test samples     : {x_test.shape[0]}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. SMOTE Oversampling (balance minority class)
# ──────────────────────────────────────────────────────────────────────────────

print("[INFO] Applying SMOTE oversampling...")
smote = SMOTE(random_state=42)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)
print(f"  After SMOTE - Training samples: {x_train_res.shape[0]}")
print(f"  Class distribution after SMOTE:\n  {pd.Series(y_train_res).value_counts().to_dict()}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Train Random Forest Classifier
# ──────────────────────────────────────────────────────────────────────────────

print("[INFO] Training RandomForestClassifier...")
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)
model.fit(x_train_res, y_train_res)

y_pred      = model.predict(x_test)
y_pred_prob = model.predict_proba(x_test)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_prob)

print(f"""
  Accuracy  : {accuracy:.4f}
  Precision : {precision:.4f}
  Recall    : {recall:.4f}
  F1 Score  : {f1:.4f}
  ROC-AUC   : {roc_auc:.4f}
""")
print("[INFO] Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Active', 'Terminated']))

# ──────────────────────────────────────────────────────────────────────────────
# 6. Visualizations
# ──────────────────────────────────────────────────────────────────────────────

DARK_BG   = '#070f1e'
SURFACE   = '#0d1b2e'
COLOR_P   = '#6366f1'
COLOR_A   = '#10b981'
COLOR_W   = '#f59e0b'
COLOR_D   = '#f43f5e'
TEXT_COL  = '#f8fafc'
MUTED_COL = '#94a3b8'

plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor':   SURFACE,
    'text.color':       TEXT_COL,
    'axes.labelcolor':  MUTED_COL,
    'xtick.color':      MUTED_COL,
    'ytick.color':      MUTED_COL,
    'axes.edgecolor':   '#1e3a5f',
    'grid.color':       '#1e3a5f',
})

print("[INFO] Generating visualizations...")

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Active', 'Terminated'],
            yticklabels=['Active', 'Terminated'], ax=ax,
            linewidths=0.5, linecolor='#1e3a5f',
            cbar_kws={'shrink': 0.8})
ax.set_title('Confusion Matrix – Employee Attrition', fontsize=14, color=TEXT_COL, pad=15)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=150, facecolor=DARK_BG)
plt.close()
print(f"  Saved -> {RESULTS_DIR}/confusion_matrix.png")

# 2. Feature Importance
importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
colors = [COLOR_P if v < importances.median() else COLOR_A for v in importances]
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(importances.index, importances.values, color=colors)
ax.set_title('Feature Importance – Drivers of Employee Attrition', fontsize=14, color=TEXT_COL, pad=15)
ax.set_xlabel('Importance Score', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'), dpi=150, facecolor=DARK_BG)
plt.close()
print(f"  Saved -> {RESULTS_DIR}/feature_importance.png")

# 3. ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color=COLOR_P, lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], '--', color=COLOR_D, lw=1.5, label='Random Classifier')
ax.fill_between(fpr, tpr, alpha=0.12, color=COLOR_P)
ax.set_title('ROC Curve – Employee Attrition Prediction', fontsize=14, color=TEXT_COL, pad=15)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.legend(loc='lower right', facecolor=SURFACE, edgecolor='#1e3a5f', labelcolor=TEXT_COL)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'), dpi=150, facecolor=DARK_BG)
plt.close()
print(f"  Saved -> {RESULTS_DIR}/roc_curve.png")

# 4. Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(rec, prec)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(rec, prec, color=COLOR_A, lw=2.5, label=f'PR Curve (AUC = {pr_auc:.3f})')
ax.fill_between(rec, prec, alpha=0.12, color=COLOR_A)
ax.set_title('Precision-Recall Curve – Employee Attrition', fontsize=14, color=TEXT_COL, pad=15)
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.legend(loc='upper right', facecolor=SURFACE, edgecolor='#1e3a5f', labelcolor=TEXT_COL)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'precision_recall_curve.png'), dpi=150, facecolor=DARK_BG)
plt.close()
print(f"  Saved -> {RESULTS_DIR}/precision_recall_curve.png")

# 5. Attrition Class Distribution
orig_df = pd.read_csv(DATA_PATH)
counts  = orig_df['STATUS'].value_counts()
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(counts.index, counts.values, color=[COLOR_A, COLOR_D], width=0.5, edgecolor='none')
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
            f'{val:,}', ha='center', fontsize=13, fontweight='bold', color=TEXT_COL)
ax.set_title('Employee Attrition Class Distribution', fontsize=14, color=TEXT_COL, pad=15)
ax.set_xlabel('Status', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'attrition_distribution.png'), dpi=150, facecolor=DARK_BG)
plt.close()
print(f"  Saved -> {RESULTS_DIR}/attrition_distribution.png")

# 6. Correlation Heatmap
corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt='.2f',
            linewidths=0.5, annot_kws={'size': 8}, ax=ax)
ax.set_title('Feature Correlation Heatmap – Employee Workforce Data', fontsize=14, color=TEXT_COL, pad=15)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'correlation_heatmap.png'), dpi=150, facecolor=DARK_BG)
plt.close()
print(f"  Saved -> {RESULTS_DIR}/correlation_heatmap.png")

# 7. Retention vs Attrition by Department (top 8 departments)
dept_df = pd.read_csv(DATA_PATH)
top_depts = dept_df['department_name'].value_counts().head(8).index
dept_df   = dept_df[dept_df['department_name'].isin(top_depts)]
pivot     = dept_df.groupby(['department_name', 'STATUS']).size().unstack(fill_value=0)
pivot     = pivot.reindex(columns=['ACTIVE', 'TERMINATED'], fill_value=0)
fig, ax   = plt.subplots(figsize=(12, 6))
x_pos     = np.arange(len(pivot))
w         = 0.38
ax.bar(x_pos - w/2, pivot['ACTIVE'],     width=w, color=COLOR_A, label='Active',     alpha=0.9)
ax.bar(x_pos + w/2, pivot['TERMINATED'], width=w, color=COLOR_D, label='Terminated', alpha=0.9)
ax.set_xticks(x_pos)
ax.set_xticklabels(pivot.index, rotation=30, ha='right', fontsize=10)
ax.set_title('Employee Retention vs Attrition by Department', fontsize=14, color=TEXT_COL, pad=15)
ax.set_ylabel('Employee Count', fontsize=12)
ax.legend(facecolor=SURFACE, edgecolor='#1e3a5f', labelcolor=TEXT_COL)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'retention_vs_attrition.png'), dpi=150, facecolor=DARK_BG)
plt.close()
print(f"  Saved -> {RESULTS_DIR}/retention_vs_attrition.png")

# 8. Model Metrics Bar Chart
metric_names  = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
metric_values = [accuracy, precision, recall, f1, roc_auc]
colors_met    = [COLOR_A, COLOR_P, COLOR_W, COLOR_D, '#a855f7']
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(metric_names, metric_values, color=colors_met, width=0.5, edgecolor='none')
for bar, val in zip(bars, metric_values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', fontweight='bold', color=TEXT_COL, fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_title('Employee Attrition Prediction – Model Performance Metrics', fontsize=14, color=TEXT_COL, pad=15)
ax.set_ylabel('Score', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'model_metrics.png'), dpi=150, facecolor=DARK_BG)
plt.close()
print(f"  Saved -> {RESULTS_DIR}/model_metrics.png")

# ──────────────────────────────────────────────────────────────────────────────
# 7. Save Accuracy Report
# ──────────────────────────────────────────────────────────────────────────────

report_text = f"""Employee Attrition Prediction - Model Evaluation Report
=========================================================
Model     : RandomForestClassifier (n_estimators=150, max_depth=10)
Dataset   : MFG10YearTerminationData.csv  ({df.shape[0]} samples after preprocessing)
Features  : {feature_names}

Performance Metrics
-------------------
Accuracy  : {accuracy:.4f}  ({accuracy * 100:.2f}%)
Precision : {precision:.4f}
Recall    : {recall:.4f}
F1 Score  : {f1:.4f}
ROC-AUC   : {roc_auc:.4f}

Classification Report
---------------------
{classification_report(y_test, y_pred, target_names=['Active', 'Terminated'])}

Confusion Matrix
----------------
{confusion_matrix(y_test, y_pred)}
"""

with open(os.path.join(RESULTS_DIR, 'accuracy_results.txt'), 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f"[INFO] Saved -> {RESULTS_DIR}/accuracy_results.txt")

# ──────────────────────────────────────────────────────────────────────────────
# 8. Save Model Artifact
# ──────────────────────────────────────────────────────────────────────────────

joblib.dump(model, MODEL_FILE)
print(f"[INFO] Saved model -> {MODEL_FILE}")
print("[DONE] Employee Attrition Model training complete!")
