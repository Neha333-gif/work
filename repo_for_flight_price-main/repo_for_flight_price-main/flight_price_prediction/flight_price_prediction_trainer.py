import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_flight_price_model():
    print("="*60)
    print("STARTING FLIGHT PRICE PREDICTION MODEL TRAINING (RANDOM FOREST)")
    print("="*60)
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, '..', 'flight_price_dataset', 'Clean_Dataset.csv')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please ensure Clean_Dataset.csv is in the flight_price_dataset folder.")
        
    print("[INFO] Loading flight price dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Shape: {df.shape}")
    
    # 1. Drop index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    # 2. Split features and target
    target = 'price'
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")
        
    X = df.drop(columns=[target])
    y = df[target]
    
    # 3. Identify categorical and numeric columns
    cat_cols = ['airline', 'flight', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']
    num_cols = ['duration', 'days_left']
    
    # Recalculate columns actually present
    cat_cols = [c for c in cat_cols if c in X.columns]
    num_cols = [c for c in num_cols if c in X.columns]
    
    print(f"[INFO] Categorical features: {cat_cols}")
    print(f"[INFO] Numeric features: {num_cols}")
    
    # 4. Train-Test Split (80-20)
    print("[INFO] Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # 5. Define Preprocessing Pipeline
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    
    # 6. Define the Lightweight Random Forest Model
    print("\n[INFO] Initializing Random Forest Regressor Pipeline...")
    rf_model = RandomForestRegressor(
        n_estimators=30,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', rf_model)
    ])
    
    # Train baseline model for comparison
    print("[INFO] Training Baseline Linear Regression for comparison...")
    baseline_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    baseline_pipeline.fit(X_train, y_train)
    baseline_pred = baseline_pipeline.predict(X_test)
    baseline_r2 = r2_score(y_test, baseline_pred)
    
    # Fit the Random Forest Model
    print("[INFO] Training Random Forest Regressor...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("[INFO] Evaluating Random Forest Regressor performance...")
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"  R2 Score : {r2:.4f}")
    print(f"  MAE      : {mae:.2f}")
    print(f"  RMSE     : {rmse:.2f}")
    
    # Save the pipeline and metadata
    pipeline_path = os.path.join(BASE_DIR, 'flight_price_prediction_pipeline.joblib')
    metadata_path = os.path.join(BASE_DIR, 'flight_price_prediction_metadata.joblib')
    
    joblib.dump(pipeline, pipeline_path)
    
    metadata = {
        'model_type': 'Random Forest Regressor',
        'r2': float(r2),
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'features': X.columns.tolist(),
        'categorical_features': cat_cols,
        'numeric_features': num_cols
    }
    joblib.dump(metadata, metadata_path)
    print(f"[OK] Saved model pipeline to {pipeline_path}")
    print(f"[OK] Saved model metadata to {metadata_path}")
    
    # 8. Generate and save visualizations
    print("\n[INFO] Generating visualizations...")
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Price Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(y, kde=True, color='#2b6ef6', bins=50)
    plt.title('Flight Ticket Price Distribution', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('Price (INR)', fontsize=12)
    plt.ylabel('Density / Count', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'price_distribution.png'), dpi=150)
    plt.close()
    
    # Plot 2: Actual vs Predicted
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.3, color='#0f62fe')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Actual vs. Predicted Price (Random Forest)', fontsize=14, fontweight='bold')
    plt.xlabel('Actual Price (INR)', fontsize=12)
    plt.ylabel('Predicted Price (INR)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'actual_vs_predicted.png'), dpi=150)
    plt.close()
    
    # Plot 3: Residuals Distribution
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='#00a4ff', bins=50)
    plt.title('Prediction Error (Residuals) Distribution', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('Residual (Actual - Predicted Price)', fontsize=12)
    plt.ylabel('Density / Count', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'residuals.png'), dpi=150)
    plt.close()
    
    # Plot 4: Feature Importance
    try:
        regressor = pipeline.named_steps['regressor']
        ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
        cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
        all_feature_names = num_cols + cat_feature_names
        
        importances = regressor.feature_importances_
        if len(importances) == len(all_feature_names):
            fi_series = pd.Series(importances, index=all_feature_names).sort_values(ascending=False).head(20)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(x=fi_series.values, y=fi_series.index, palette='viridis')
            plt.title('Top 20 Feature Importances (Random Forest)', fontsize=15, fontweight='bold', pad=15)
            plt.xlabel('F-Score / Importance Weight', fontsize=12)
            plt.ylabel('Feature Name', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'), dpi=150)
            plt.close()
            print("  Generated feature_importance.png successfully.")
    except Exception as e:
        print(f"  [WARNING] Could not plot feature importance: {e}")
            
    # Plot 5: Correlation Heatmap (numerical features and target)
    plt.figure(figsize=(8, 6))
    corr = df[['duration', 'days_left', 'price']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, square=True)
    plt.title('Correlation Heatmap', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'correlation_heatmap.png'), dpi=150)
    plt.close()
    
    # Plot 6: Model Comparison Bar Chart (Random Forest vs Baseline)
    plt.figure(figsize=(10, 6))
    model_names = ['Linear Regression Baseline', 'Random Forest Regressor']
    r2_scores = [baseline_r2, r2]
    bars = plt.bar(model_names, r2_scores, color=['#a0aec0', '#3182ce'], width=0.4)
    plt.title('Model R2 Score Comparison', fontsize=15, fontweight='bold', pad=15)
    plt.ylabel('R2 Score (Higher is Better)', fontsize=12)
    plt.ylim(0, 1.05)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.02, f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison.png'), dpi=150)
    plt.close()
    
    # Save a detailed evaluation report
    report_text = f"""Flight Price Prediction Model Evaluation Report
=================================================
Trained on Easemytrip Dataset (300,153 rows)
Model Selected: Random Forest Regressor (Lightweight ML Model)

Performance metrics of Random Forest Regressor model:
------------------------------------------
R2 Score (Coefficient of Determination): {metadata['r2']:.4f}
Mean Absolute Error (MAE)            : INR {metadata['mae']:.2f}
Mean Squared Error (MSE)             : {metadata['mse']:.2f}
Root Mean Squared Error (RMSE)       : INR {metadata['rmse']:.2f}

Model Comparison:
---------------------------
Linear Regression Baseline   -> R2: {baseline_r2:.4f}
Random Forest Regressor      -> R2: {r2:.4f} | MAE: INR {mae:.2f} | RMSE: INR {rmse:.2f}
"""
    with open(os.path.join(RESULTS_DIR, 'flight_price_prediction_evaluation.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"[OK] Evaluation report saved successfully.")
    
    # 9. Save a small sample dataset with predictions for frontend testing/preview
    sample_df = df.head(100).copy()
    sample_df['predicted_price'] = pipeline.predict(sample_df.drop(columns=['price']))
    sample_df.to_csv(os.path.join(RESULTS_DIR, 'sample_flight_price_predictions.csv'), index=False)
    print(f"[OK] Generated sample predictions CSV.")
    
    print("\n" + "="*60)
    print("MODEL TRAINING PIPELINE SUCCESSFULLY COMPLETED")
    print("="*60)

if __name__ == '__main__':
    train_flight_price_model()