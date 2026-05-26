# -*- coding: utf-8 -*-
"""insurance_claim_prediction.ipynb

Standalone insurance claim classification demo script.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib

import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'insurance_claim_prediction', 'insurance_claim_prediction_dataset', 'insurance_claims.csv')

print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f" dataframe shape : {df.shape}")

if 'customer_id' in df.columns:
    df = df.drop(columns=['customer_id'])

if 'insuranceclaim' not in df.columns:
    raise ValueError("Target column 'insuranceclaim' not found in dataset.")

X = df.drop(columns=['insuranceclaim'])
y = df['insuranceclaim']

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col])
    else:
        X[col] = StandardScaler().fit_transform(X[col].values.reshape(-1, 1))

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"x_train : {x_train.shape}")
print(f"x_test : {x_test.shape}")
print(f"y_train : {y_train.shape}")
print(f"y_test : {y_test.shape}")

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(x_train, y_train_encoded)

y_pred_encoded = model.predict(x_test)

accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
report = classification_report(y_test_encoded, y_pred_encoded, target_names=le.classes_)

print(f"Accuracy: {accuracy}\n")
print("Classification Report:\n", report)

output_path = os.path.join(BASE_DIR, 'insurance_claim_prediction', 'insurance_claim_prediction_model.joblib')
joblib.dump(model, output_path)
print(f"Model saved as {output_path}")
