# Product Recommendation System

Unified ML pipeline and FastAPI backend for recommending beauty products from customer interaction data.

## Dataset

`product_recommendation_engine/ratings_Beauty.csv`

## Quick Start

```bash
pip install -r requirements.txt
python product_recommendation_system.py --train
python product_recommendation_system.py --port 8000
```

Open `http://127.0.0.1:8000/` for the frontend UI.

## Models

- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier

The best model (highest F1 score) is saved automatically.

## Outputs

All artifacts are written to `results/`:

- Evaluation reports (`accuracy_results.txt`, `evaluation_report.txt`)
- Recommendations CSV (`product_recommendation_outputs.csv`)
- Visualizations (product popularity, correlation heatmap, feature importance, etc.)
- Model files (`product_recommendation_system_model.joblib`, `product_recommendation_system_metadata.joblib`)

## API Endpoints

- `GET /` - Frontend UI
- `GET /health` - Health check
- `POST /predict` - Get product recommendations
- `GET /results/metrics` - Evaluation report and plot paths
- `GET /results/image/{image_name}` - Serve generated plots
- `POST /retrain` - Retrain model in background
