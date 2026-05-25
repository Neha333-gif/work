# -*- coding: utf-8 -*-
"""
Email Spam Prediction Trainer

Pipeline:
- Load SMS/Email spam dataset (SMS Spam Collection-like CSV)
- TF-IDF vectorization
- Train a lightweight classifier (Logistic Regression)
- Evaluate and save metrics
- Produce visualizations: confusion matrix, class distribution, ROC, feature importance, correlation heatmap
- Save model artifacts expected by the FastAPI backend
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
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
warnings_imported = False
try:
    import warnings
    warnings.filterwarnings('ignore')
    warnings_imported = True
except Exception:
    pass


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'emial_spam_dataset', 'spam.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

print('[INFO] Loading dataset...')
df = pd.read_csv(DATA_PATH, encoding='latin-1')

print(f'  Dataset shape : {df.shape}')
print(f'  Columns       : {list(df.columns)}')

# The SMS Spam Collection uses columns: v1 -> label (ham/spam), v2 -> text
label_col = 'v1' if 'v1' in df.columns else 'label'
text_col = 'v2' if 'v2' in df.columns else 'text'

df = df[[label_col, text_col]].dropna()
df.columns = ['label', 'text']
df['text'] = df['text'].astype(str).str.strip()
df['label'] = df['label'].astype(str).str.strip()

# Encode labels
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])  # ham -> 0, spam -> 1

X_text = df['text']
y = df['label_enc']

print('[INFO] Vectorizing text with TF-IDF...')
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
X_vec = vectorizer.fit_transform(X_text)

print('[INFO] Splitting data...')
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)

print('[INFO] Training Logistic Regression classifier...')
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

print('[INFO] Evaluating model...')
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
report    = classification_report(y_test, y_pred, target_names=le.classes_)

print('\n  Accuracy  : {0:.4f}'.format(accuracy))
print('  Precision : {0:.4f}'.format(precision))
print('  Recall    : {0:.4f}'.format(recall))
print('  F1-Score  : {0:.4f}\n'.format(f1))
print('  Classification Report:\n', report)

# Save textual metrics
metrics_text = f"""Email Spam Prediction - Model Evaluation Report
=================================================
Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)
Precision : {precision:.4f}
Recall    : {recall:.4f}
F1-Score  : {f1:.4f}

Classification Report:
{report}
"""
with open(os.path.join(RESULTS_DIR, 'accuracy_results.txt'), 'w', encoding='utf-8') as f:
    f.write(metrics_text)
print(f"[INFO] Metrics saved -> {RESULTS_DIR}/accuracy_results.txt")

sns.set_theme(style='darkgrid', palette='muted')

# Confusion matrix
print('[INFO] Generating confusion matrix...')
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()

# Class distribution (test set)
print('[INFO] Generating class distribution plot...')
plt.figure(figsize=(6,4))
pd.Series(le.inverse_transform(y_test), name='label').value_counts().plot(kind='bar', color=['#10b981','#f43f5e'])
plt.title('Class Distribution (Test Set)')
plt.xlabel('Label')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'class_distribution.png'), dpi=150)
plt.close()

# ROC curve
if y_proba is not None:
    print('[INFO] Generating ROC curve...')
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}', color='#6366f1')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'), dpi=150)
    plt.close()

# Feature importance / coefficients for Logistic Regression
print('[INFO] Generating feature importance (top coefficients)...')
feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_[0]
top_pos_idx = np.argsort(coefs)[-20:][::-1]
top_neg_idx = np.argsort(coefs)[:20]

top_features = [(feature_names[i], float(coefs[i])) for i in np.concatenate([top_pos_idx, top_neg_idx])]
top_df = pd.DataFrame(top_features, columns=['feature','coef'])

plt.figure(figsize=(10,8))
sns.barplot(x='coef', y='feature', data=top_df.head(20), palette='vlag')
plt.title('Top 20 Feature Coefficients (positive = spam)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'), dpi=150)
plt.close()

# Correlation heatmap of top features (presence matrix)
print('[INFO] Generating correlation heatmap for top features...')
top_feature_names = [f for f,_ in top_features[:15]]
if len(top_feature_names) > 1:
    # Build a dense matrix for top features over the entire dataset
    X_top = vectorizer.transform(df['text']).tocsc()[:, [list(feature_names).index(f) for f in top_feature_names]]
    top_df_features = pd.DataFrame(X_top.toarray(), columns=top_feature_names)
    corr = top_df_features.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap (Top Predictor Words)')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'correlation_heatmap.png'), dpi=150)
    plt.close()

# Save model assets expected by the API
print('[INFO] Saving model and pipeline assets...')
joblib.dump(model, os.path.join(BASE_DIR, 'email_spam_model.joblib'))
joblib.dump(vectorizer, os.path.join(BASE_DIR, 'email_spam_vectorizer.joblib'))
joblib.dump(le, os.path.join(BASE_DIR, 'email_spam_label_encoder.joblib'))

metadata = {
    'model_type': 'TF-IDF + Logistic Regression',
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'num_features': len(feature_names),
}
# Build top spam indicators for frontend explainability
max_abs = float(np.max(np.abs(coefs))) if len(coefs) > 0 else 1.0
spam_indices = np.argsort(coefs)[-50:][::-1]  # top positive coeffs (spam indicators)
top_spam_indicators = []
for i in spam_indices:
    word = feature_names[i]
    importance = float(abs(coefs[i]) / max_abs) if max_abs > 0 else 0.0
    top_spam_indicators.append({
        'word': word,
        'importance': importance,
        'coef': float(coefs[i])
    })

metadata['top_spam_indicators'] = top_spam_indicators
joblib.dump(metadata, os.path.join(BASE_DIR, 'email_spam_metadata.joblib'))

# Save sample predictions for frontend
print('[INFO] Saving sample predictions...')
sample_df = df.sample(n=min(200, len(df)), random_state=42).copy()
sample_vec = vectorizer.transform(sample_df['text'])
sample_df['prediction_prob'] = model.predict_proba(sample_vec)[:,1]
sample_df['prediction'] = le.inverse_transform(model.predict(sample_vec))
sample_df.to_csv(os.path.join(BASE_DIR, 'sample_spam_predictions.csv'), index=False)

print('\n[DONE] Training complete. Assets saved in project root and results/.')
print('  Model         : email_spam_model.joblib')
print('  Vectorizer    : email_spam_vectorizer.joblib')
print('  LabelEncoder  : email_spam_label_encoder.joblib')
print('  Metadata      : email_spam_metadata.joblib')
print(f'  Results (plots & report): {RESULTS_DIR}')