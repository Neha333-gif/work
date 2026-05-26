# -*- coding: utf-8 -*-
"""taxi_trip_fare_model.py

Standalone taxi trip fare regression demo script.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'taxi_trip_fare_prediction_dataset', 'taxi_trip_fare_train.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'taxi_trip_fare_prediction_model.joblib')

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. Please copy the taxi fare dataset into taxi_trip_fare_prediction_dataset."
    )

print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"Dataframe shape: {df.shape}")

required_columns = [
    'trip_duration',
    'distance_traveled',
    'num_of_passengers',
    'fare',
    'surge_applied',
    'total_fare'
]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

X = df[['trip_duration', 'distance_traveled', 'num_of_passengers', 'fare', 'surge_applied']]
y = df['total_fare']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"x_train: {x_train.shape}")
print(f"x_test: {x_test.shape}")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

pipeline = Pipeline([
    ('scaler', scaler),
    ('regressor', model)
])
joblib.dump(pipeline, MODEL_PATH)
print(f"Model saved as {MODEL_PATH}")
