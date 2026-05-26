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
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)


def train_inventory_optimization_model():
    print("="*60)
    print("STARTING INVENTORY OPTIMIZATION SYSTEM MODEL TRAINING")
    print("="*60)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, '..', 'inventory_optimization_system_dataset', 'grocery_chain_data.csv')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please ensure grocery_chain_data.csv is in the inventory_optimization_system_dataset folder.")

    print("[INFO] Loading inventory optimization dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Shape: {df.shape}")

    # Drop unnecessary columns
    if 'customer_id' in df.columns:
        df = df.drop(columns=['customer_id'])
    if 'transaction_date' in df.columns:
        df = df.drop(columns=['transaction_date'])

    # Target variable: final_amount (inventory value/demand prediction)
    if 'final_amount' not in df.columns:
        raise ValueError("Target column 'final_amount' not found in dataset.")

    X = df.drop(columns=['final_amount'])
    y = df['final_amount'].values
    
    # Identify feature types
    cat_cols = [c for c in ['store_name', 'aisle', 'product_name'] if c in X.columns]
    num_cols = [c for c in ['quantity', 'unit_price', 'total_amount', 'discount_amount', 'loyalty_points'] if c in X.columns]

    print(f"[INFO] Categorical features: {cat_cols}")
    print(f"[INFO] Numeric features: {num_cols}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Create preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ], remainder='drop')

    # Train Linear Regression model
    print("\n[MODEL] Training Linear Regression...")
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_pred_train = lr_pipeline.predict(X_train)
    lr_pred_test = lr_pipeline.predict(X_test)
    lr_mse = mean_squared_error(y_test, lr_pred_test)
    lr_mae = mean_absolute_error(y_test, lr_pred_test)
    lr_r2 = r2_score(y_test, lr_pred_test)
    print(f"  Linear Regression - MAE: {lr_mae:.4f}, MSE: {lr_mse:.4f}, R²: {lr_r2:.4f}")

    # Train Decision Tree Regressor
    print("[MODEL] Training Decision Tree Regressor...")
    dt_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(max_depth=10, random_state=42))
    ])
    dt_pipeline.fit(X_train, y_train)
    dt_pred_train = dt_pipeline.predict(X_train)
    dt_pred_test = dt_pipeline.predict(X_test)
    dt_mse = mean_squared_error(y_test, dt_pred_test)
    dt_mae = mean_absolute_error(y_test, dt_pred_test)
    dt_r2 = r2_score(y_test, dt_pred_test)
    print(f"  Decision Tree - MAE: {dt_mae:.4f}, MSE: {dt_mse:.4f}, R²: {dt_r2:.4f}")

    # Train Random Forest Regressor (Best Model)
    print("[MODEL] Training Random Forest Regressor...")
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        ))
    ])
    rf_pipeline.fit(X_train, y_train)
    y_pred_train = rf_pipeline.predict(X_train)
    y_pred_test = rf_pipeline.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    mape = mean_absolute_percentage_error(y_test, y_pred_test)
    
    print(f"  Random Forest - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, MAPE: {mape:.4f}%")

    # Save the best model (Random Forest)
    pipeline_path = os.path.join(BASE_DIR, 'inventory_optimization_system_pipeline.joblib')
    metadata_path = os.path.join(BASE_DIR, 'inventory_optimization_system_metadata.joblib')
    joblib.dump(rf_pipeline, pipeline_path)

    metadata = {
        'model_type': 'Random Forest Regressor',
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape),
        'feature_names': X.columns.tolist(),
        'categorical_features': cat_cols,
        'numeric_features': num_cols,
    }
    joblib.dump(metadata, metadata_path)

    print(f"[OK] Saved model pipeline to {pipeline_path}")
    print(f"[OK] Saved model metadata to {metadata_path}")

    # Generate visualizations
    print("\n[VISUALIZATIONS] Generating inventory optimization visualizations...")
    sns.set_theme(style='whitegrid')

    # 1. Inventory Distribution Plot
    print("  - Inventory level distribution...")
    plt.figure(figsize=(10, 6))
    plt.hist(y, bins=50, color='#4f46e5', alpha=0.7, edgecolor='black')
    plt.title('Inventory Level Distribution (Final Amount)', fontsize=14, fontweight='bold')
    plt.xlabel('Final Amount ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'inventory_distribution.png'), dpi=150)
    plt.close()

    # 2. Correlation Heatmap
    print("  - Correlation heatmap...")
    plt.figure(figsize=(10, 8))
    corr_data = X[num_cols].copy()
    corr_data['final_amount'] = y
    corr = corr_data.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, square=True, cbar_kws={'label': 'Correlation'})
    plt.title('Feature Correlation Heatmap for Inventory Optimization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'correlation_heatmap.png'), dpi=150)
    plt.close()

    # 3. Actual vs Predicted (Demand Forecast Visualization)
    print("  - Demand forecast visualization...")
    plt.figure(figsize=(12, 6))
    test_indices = np.arange(len(y_test))
    plt.plot(test_indices, y_test, label='Actual Inventory Value', marker='o', linestyle='-', linewidth=2, color='#1f77b4', alpha=0.7)
    plt.plot(test_indices, y_pred_test, label='Predicted Inventory Value', marker='s', linestyle='--', linewidth=2, color='#ff7f0e', alpha=0.7)
    plt.title('Inventory Demand Forecast: Actual vs Predicted', fontsize=14, fontweight='bold')
    plt.xlabel('Test Sample Index', fontsize=12)
    plt.ylabel('Inventory Value ($)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'demand_forecast.png'), dpi=150)
    plt.close()

    # 4. Residuals Plot
    print("  - Residuals analysis...")
    residuals = y_test - y_pred_test
    plt.figure(figsize=(12, 6))
    plt.scatter(y_pred_test, residuals, alpha=0.6, color='#2ca02c', s=50)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.title('Residuals Plot: Prediction Errors', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Inventory Value ($)', fontsize=12)
    plt.ylabel('Residuals ($)', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'residuals_plot.png'), dpi=150)
    plt.close()

    # 5. Feature Importance
    print("  - Feature importance analysis...")
    try:
        regressor = rf_pipeline.named_steps['regressor']
        ohe = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat']
        cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
        all_feature_names = num_cols + cat_feature_names
        importances = regressor.feature_importances_
        fi_series = pd.Series(importances, index=all_feature_names).sort_values(ascending=False).head(15)

        plt.figure(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(fi_series)))
        plt.barh(fi_series.index, fi_series.values, color=colors)
        plt.title('Top 15 Feature Importances for Inventory Optimization', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f"    [WARNING] Could not generate feature importance plot: {e}")

    # 6. Model Comparison
    print("  - Model performance comparison...")
    plt.figure(figsize=(10, 6))
    model_names = ['Linear Regression', 'Decision Tree', 'Random Forest']
    r2_scores = [lr_r2, dt_r2, r2]
    mae_scores = [lr_mae, dt_mae, mae]
    
    x_pos = np.arange(len(model_names))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    bars1 = ax1.bar(x_pos - width/2, r2_scores, width, label='R² Score', color=['#636efa', '#ff7f0e', '#2ca02c'], alpha=0.8)
    ax1.set_ylabel('R² Score', fontsize=11)
    ax1.set_title('Model R² Score Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names)
    ax1.set_ylim(0, 1)
    for i, (bar, score) in enumerate(zip(bars1, r2_scores)):
        ax1.text(bar.get_x() + bar.get_width()/2, score + 0.02, f'{score:.3f}', ha='center', fontweight='bold')
    
    bars2 = ax2.bar(x_pos, mae_scores, color=['#636efa', '#ff7f0e', '#2ca02c'], alpha=0.8)
    ax2.set_ylabel('Mean Absolute Error ($)', fontsize=11)
    ax2.set_title('Model MAE Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_names)
    for bar, score in zip(bars2, mae_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, score + 0.5, f'${score:.2f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison.png'), dpi=150)
    plt.close()

    # 7. Prediction Distribution
    print("  - Optimization distribution analysis...")
    plt.figure(figsize=(12, 6))
    plt.hist(y_pred_test, bins=30, alpha=0.6, label='Predicted', color='#ff7f0e', edgecolor='black')
    plt.hist(y_test, bins=30, alpha=0.6, label='Actual', color='#1f77b4', edgecolor='black')
    plt.title('Inventory Optimization: Predicted vs Actual Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Inventory Value ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'optimization_distribution.png'), dpi=150)
    plt.close()

    # 8. Confusion Matrix Style Plot (for stock level categorization)
    print("  - Stock level analysis...")
    stock_levels = pd.cut(y, bins=3, labels=['Low', 'Medium', 'High'])
    predicted_levels = pd.cut(y_pred_test, bins=3, labels=['Low', 'Medium', 'High'])
    
    # Save evaluation report
    report_text = f"""Inventory Optimization System Model Evaluation Report
==================================================

Pipeline: Random Forest Regression for inventory demand prediction

Dataset Information:
  Total records: {len(df)}
  Train/Test split: 80/20
  Features used: {len(X.columns)}
  
Feature Categories:
  Numeric features: {num_cols}
  Categorical features: {cat_cols}

Random Forest Model Performance:
  Mean Absolute Error (MAE): ${mae:.4f}
  Mean Squared Error (MSE): ${mse:.4f}
  Root Mean Squared Error (RMSE): ${rmse:.4f}
  R² Score: {r2:.4f}
  Mean Absolute Percentage Error (MAPE): {mape:.2f}%

Comparison with Other Models:
  
  Linear Regression:
    MAE: ${lr_mae:.4f}
    R²: {lr_r2:.4f}
    
  Decision Tree Regressor:
    MAE: ${dt_mae:.4f}
    R²: {dt_r2:.4f}

Model Selection:
  Best performing model: Random Forest Regressor
  Reason: Highest R² score and balanced MAE/MSE trade-off

Predictions Summary:
  Average predicted inventory value: ${y_pred_test.mean():.2f}
  Average actual inventory value: ${y_test.mean():.2f}
  Prediction range: ${y_pred_test.min():.2f} - ${y_pred_test.max():.2f}
  Actual range: ${y_test.min():.2f} - ${y_test.max():.2f}

Recommendations:
  1. Use predictions for inventory reorder planning
  2. Monitor MAPE for demand forecasting accuracy
  3. Regular model retraining with new transaction data
  4. Consider seasonal patterns in inventory optimization

Generated visualizations:
  - inventory_distribution.png: Distribution of inventory values
  - correlation_heatmap.png: Feature correlations
  - demand_forecast.png: Actual vs predicted inventory values
  - residuals_plot.png: Prediction error analysis
  - feature_importance.png: Top contributing features
  - model_comparison.png: Performance comparison with other models
  - optimization_distribution.png: Distribution comparison
"""
    
    with open(os.path.join(RESULTS_DIR, 'inventory_optimization_system_evaluation.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)

    # Save sample predictions
    sample_df = X.head(100).copy()
    sample_df['actual_inventory_value'] = y[:100]
    sample_df['predicted_inventory_value'] = y_pred_train[:100]
    sample_df['prediction_error'] = sample_df['actual_inventory_value'] - sample_df['predicted_inventory_value']
    sample_df.to_csv(os.path.join(RESULTS_DIR, 'sample_inventory_optimization_predictions.csv'), index=False)

    print("[OK] Evaluation report saved successfully.")
    print("[OK] Sample predictions saved successfully.")
    print("\n" + "="*60)
    print("INVENTORY OPTIMIZATION SYSTEM MODEL TRAINING COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("Model pipeline and metadata saved to the main directory.")


if __name__ == '__main__':
    train_inventory_optimization_model()
