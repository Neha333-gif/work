# -*- coding: utf-8 -*-
"""
Disease Prediction Model Trainer
Automatically processes clinical symptom data and trains an XGBoost Classifier with SMOTE.
Saves model assets and logs visualizations to the results/ folder.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - no GUI windows
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. Load Data
# ──────────────────────────────────────────────
DATA_PATH = r"C:\Users\peepl\Downloads\cust_segmentation_zip\customer_segmentaion\disease prediction\disease prediction.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("[INFO] Loading dataset...")
df = pd.read_csv(DATA_PATH)

print(f"  Dataset shape : {df.shape}")
print(f"  Columns       : {list(df.columns)}")

# ──────────────────────────────────────────────
# 2. Preprocessing
# ──────────────────────────────────────────────
print("[INFO] Preprocessing data...")

# Strip whitespace from all string cells
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Extract target
y_raw = df['Disease'].copy()
df.drop('Disease', axis=1, inplace=True)

# Identify symptom columns
symptom_cols = [c for c in df.columns if 'Symptom' in c]

# Gather all unique non-NaN symptoms (sorted)
all_symptoms = set()
for col in symptom_cols:
    all_symptoms.update(df[col].dropna().unique())
all_symptoms = sorted(all_symptoms)

# Build global symptom → integer mapping
# 0 = None / missing, 1-N = specific symptoms
symptom_to_int = {sym: idx + 1 for idx, sym in enumerate(all_symptoms)}
unique_symptoms = all_symptoms  # clean list for frontend dropdown

print(f"  Total unique symptoms: {len(unique_symptoms)}")

# Encode each symptom column using the global map (NaN → 0)
X = df[symptom_cols].copy()
for col in symptom_cols:
    X[col] = X[col].map(lambda v: symptom_to_int.get(v, 0) if pd.notna(v) else 0)

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)

# ──────────────────────────────────────────────
# 3. Train / Test Split + SMOTE
# ──────────────────────────────────────────────
print("[INFO] Splitting dataset and applying SMOTE...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"  Training size after SMOTE : {X_train_res.shape}")
print(f"  Test size                 : {X_test.shape}")

# ──────────────────────────────────────────────
# 4. Train XGBoost Classifier
# ──────────────────────────────────────────────
print("[INFO] Training XGBoost classifier...")
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist",
    n_jobs=2,
    eval_metric="mlogloss",
    use_label_encoder=False
)
xgb_model.fit(X_train_res, y_train_res)

# ──────────────────────────────────────────────
# 5. Evaluate
# ──────────────────────────────────────────────
print("[INFO] Evaluating model...")
y_pred = xgb_model.predict(X_test)

accuracy  = accuracy_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall    = recall_score(y_test, y_pred, average='weighted')
report    = classification_report(y_test, y_pred, target_names=le.classes_)

print(f"\n  Accuracy  : {accuracy:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print("\n  Classification Report:\n", report)

# Save metrics to text file
metrics_text = f"""AI Disease Prediction - Model Evaluation Report
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
print(f"[INFO] Metrics saved -> {RESULTS_DIR}/accuracy_results.txt")

# ──────────────────────────────────────────────
# 6. Visualizations  (saved to results/, no plt.show())
# ──────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")

# --- 6a. Confusion Matrix ---
print("[INFO] Generating confusion matrix plot...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(22, 18))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=le.classes_, yticklabels=le.classes_,
    linewidths=0.3, linecolor='grey'
)
plt.xlabel('Predicted Label', fontsize=11)
plt.ylabel('True Label', fontsize=11)
plt.title('Confusion Matrix – Disease Prediction (XGBoost)', fontsize=14, pad=16)
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"[INFO] Saved -> {RESULTS_DIR}/confusion_matrix.png")

# --- 6b. Disease Distribution ---
print("[INFO] Generating disease distribution plot...")
disease_series = pd.Series(le.inverse_transform(y_test), name="Disease")
plt.figure(figsize=(14, 10))
order = disease_series.value_counts().index.tolist()
sns.countplot(
    y=disease_series, order=order,
    palette="coolwarm"
)
plt.xlabel("Number of Records", fontsize=12)
plt.ylabel("Disease", fontsize=12)
plt.title("Disease Distribution – Test Set", fontsize=14, pad=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "disease_distribution.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"[INFO] Saved -> {RESULTS_DIR}/disease_distribution.png")

# --- 6c. Symptom Frequency ---
print("[INFO] Generating symptom frequency plot...")
all_symptom_values = []
for col in symptom_cols:
    # Reload original values from raw df (before encoding)
    raw_df_col = pd.read_csv(DATA_PATH)[col].dropna().str.strip()
    all_symptom_values.extend(raw_df_col.tolist())

symptom_counts = pd.Series(all_symptom_values).value_counts().head(30)
plt.figure(figsize=(14, 9))
sns.barplot(x=symptom_counts.values, y=symptom_counts.index, palette="viridis")
plt.xlabel("Frequency", fontsize=12)
plt.ylabel("Symptom", fontsize=12)
plt.title("Top 30 Most Frequent Symptoms", fontsize=14, pad=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "symptom_frequency.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"[INFO] Saved -> {RESULTS_DIR}/symptom_frequency.png")

# ──────────────────────────────────────────────
# 7. Save Model Assets
# ──────────────────────────────────────────────
print("[INFO] Saving model assets...")
joblib.dump(xgb_model, "xgboost_disease_model.joblib")
print("  Saved -> xgboost_disease_model.joblib")

joblib.dump(le, "disease_label_encoder.joblib")
print("  Saved -> disease_label_encoder.joblib")

symptom_encoder_data = {
    "symptom_to_int": symptom_to_int,
    "unique_symptoms": unique_symptoms
}
joblib.dump(symptom_encoder_data, "symptom_encoder.joblib")
print("  Saved -> symptom_encoder.joblib")

print("\n[DONE] Training complete! All assets saved.")
print(f"  Model    : xgboost_disease_model.joblib")
print(f"  Encoder  : disease_label_encoder.joblib")
print(f"  Symptoms : symptom_encoder.joblib")
print(f"  Results  : {RESULTS_DIR}/")