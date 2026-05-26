# Credit Risk Analysis

A machine learning project for predicting credit risk and loan default probability using a Random Forest classifier.

## Overview

This project builds and deploys a credit risk classification model that predicts whether a loan applicant is likely to default based on their financial and personal information. The model achieves high accuracy by comparing multiple algorithms and selecting the best performer.

## Features

- **Multiple Model Algorithms**: Logistic Regression, Decision Tree, and Random Forest
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics
- **Interactive Web Frontend**: User-friendly HTML interface for real-time predictions
- **FastAPI Backend**: RESTful API for model serving and batch predictions
- **Data Visualizations**: Correlation heatmaps, confusion matrices, feature importance plots
- **Model Artifacts**: Serialized pipeline and metadata for reproducibility

## Dataset

**Source**: Bank Loan Default Dataset (`bankloans.csv`)

- Contains loan application records with financial and demographic features
- Target variable: `default` (0 = Non-Default, 1 = Default)
- Features: age, income, credit score, loan amount, employment history, and more

## Project Structure

```
credit_risk_analysis/
├── credit_risk_analysis_trainer.py       # Model training script
├── credit_risk_analysis_api.py           # FastAPI backend server
├── credit_risk_analysis_frontend.html    # Web interface
├── credit_risk_analysis_pipeline.joblib  # Trained model artifact
├── credit_risk_analysis_metadata.joblib  # Model metadata and metrics
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
├── credit_risk_analysis_dataset/
│   └── bankloans.csv                     # Training dataset
└── results/
    ├── default_distribution.png          # Class distribution chart
    ├── correlation_heatmap.png           # Feature correlation matrix
    ├── confusion_matrix.png              # Model confusion matrix
    ├── feature_importance.png            # Top features plot
    ├── model_performance.png             # Model comparison chart
    ├── prediction_distribution.png       # Prediction distribution
    ├── sample_credit_risk_predictions.csv # Test predictions
    └── credit_risk_analysis_evaluation.txt # Evaluation report
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd credit_risk_analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training the Model

Run the trainer to build and evaluate the model:

```bash
python credit_risk_analysis_trainer.py
```

This will:
- Load and preprocess the dataset
- Train three different models
- Select the best performer
- Generate evaluation metrics
- Create visualizations
- Save model artifacts

### 2. Starting the Backend API

Launch the FastAPI server:

```bash
python credit_risk_analysis_api.py
```

The API will be available at `http://127.0.0.1:8000`

### 3. Using the Web Interface

Open `credit_risk_analysis_frontend.html` in a web browser to:
- Input applicant financial information
- Get real-time risk predictions
- View probability scores and risk classifications
- Reset the form for new predictions

## API Endpoints

### Health Check
- **GET** `/health` - Server status and model information

### Model Metadata
- **GET** `/metadata` - Model type and performance metrics

### Single Prediction
- **POST** `/predict` - Predict risk for a single applicant
  ```json
  {
    "features": {
      "age": 35,
      "income": 75,
      "loan_amount": 50,
      "loan_duration": 24,
      "credit_score": 700,
      "employment_years": 5,
      "existing_debts": 15,
      "loan_purpose": "auto"
    }
  }
  ```

### Batch Prediction
- **POST** `/batch_predict` - Predict risk for multiple applicants

### Results
- **GET** `/results/list` - List all available result files
- **GET** `/results/{file_name}` - Download specific result file

## Model Performance

The trained model achieves:
- **Accuracy**: ~85-92% (depends on train/test split)
- **Precision**: ~0.80+
- **Recall**: ~0.75+
- **F1-Score**: ~0.77+
- **ROC-AUC**: ~0.90+

## Technology Stack

- **Backend**: Python, FastAPI, Uvicorn
- **ML Framework**: scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Model Serialization**: Joblib
- **Frontend**: HTML5, CSS3, JavaScript

## Risk Levels

The model classifies applicants into three risk categories:

- **LOW RISK** (0): Applicant unlikely to default
- **MEDIUM RISK** (0.5-0.7 probability): Moderate default risk
- **HIGH RISK** (>0.7 probability): High default probability

## Example Usage

```python
# Load trained model
import joblib
from credit_risk_analysis_api import pipeline

# Create prediction
sample = {
    'age': 35,
    'income': 75,
    'loan_amount': 50,
    'loan_duration': 24,
    'credit_score': 700,
    'employment_years': 5,
    'existing_debts': 15,
    'loan_purpose': 'auto'
}

prediction = pipeline.predict([sample])
probability = pipeline.predict_proba([sample])
```

## Future Enhancements

- Add SHAP values for model interpretability
- Implement hyperparameter tuning with GridSearchCV
- Add data drift detection
- Create monitoring dashboard
- Deploy as containerized microservice
- Add unit tests and CI/CD pipeline

## License

MIT License - See LICENSE file for details

## Author

Data Science Team

## Support

For issues or questions, please create an issue in the repository.
