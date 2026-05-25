# 🎯 Customer Segmentation Project Setup Guide

This is a complete ML pipeline for customer segmentation using K-means clustering, with FastAPI backend and interactive web frontend - similar to the retail sales prediction structure.

## 📦 Project Structure

```
customer-segmentation-project/
├── Train.csv                              # Training dataset
├── Test.csv                               # Test dataset
├── train_model.py                         # Model training script
├── main.py                                # FastAPI backend
├── customer_segmentation_frontend.html    # Interactive web UI
├── kmeans_model.joblib                    # Trained model (generated)
├── scaler.joblib                          # Feature scaler (generated)
├── imputer.joblib                         # Data imputer (generated)
├── label_encoders.joblib                  # Categorical encoders (generated)
├── cluster_info.joblib                    # Cluster metadata (generated)
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt contents:**
```
fastapi==0.104.1
uvicorn==0.24.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.2
python-multipart==0.0.6
```

### 2. Prepare Your Data

Make sure your `Train.csv` and `Test.csv` files are in the project directory.

**Expected CSV structure (Train.csv):**
```
annual_income,spending_score,age,recency,frequency,monetary,... (other features)
50000,50,35,30,5,500
75000,75,28,15,10,1500
30000,25,55,60,2,200
```

### 3. Train the Model

```bash
python train_model.py
```

This will:
- Load and preprocess your data
- Determine optimal number of clusters (using elbow method)
- Train K-means clustering model
- Save all transformers and the trained model
- Generate clustering metrics

**Output files created:**
- ✓ `kmeans_model.joblib` - Trained K-means model
- ✓ `scaler.joblib` - StandardScaler for numerical features
- ✓ `imputer.joblib` - SimpleImputer for missing values
- ✓ `label_encoders.joblib` - LabelEncoders for categorical features
- ✓ `numeric_cols.joblib` - List of numeric column names
- ✓ `categorical_cols.joblib` - List of categorical column names
- ✓ `cluster_info.joblib` - Cluster statistics and metadata
- ✓ `clustered_data.csv` - Original data with cluster assignments

### 4. Start FastAPI Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: **http://localhost:8000**

**API Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 5. Open the Web Interface

Simply open `customer_segmentation_frontend.html` in your web browser, or access it from a local web server:

```bash
# Python 3
python -m http.server 8080
```

Then visit: **http://localhost:8080/customer_segmentation_frontend.html**

---

## 📊 API Endpoints

### Health Check
```bash
GET /health
```
Response: `{"status": "ok", "model": "Customer Segmentation K-means"}`

### Get Model Info
```bash
GET /info
```
Returns cluster statistics, silhouette scores, and segment characteristics.

### Segment Single Customer
```bash
POST /segment
Content-Type: application/json

{
  "annual_income": 50000,
  "spending_score": 50,
  "age": 35,
  "recency": 30,
  "frequency": 5,
  "monetary": 500
}
```

Response:
```json
{
  "cluster": 1,
  "segment_name": "Regular Customers",
  "description": "Consistent, moderate engagement",
  "color": "#4ECDC4",
  "distance_to_center": 2.34
}
```

### Batch Segment Multiple Customers
```bash
POST /segment/batch
Content-Type: application/json

[
  {"annual_income": 50000, "spending_score": 50, "age": 35, "recency": 30, "frequency": 5, "monetary": 500},
  {"annual_income": 75000, "spending_score": 75, "age": 28, "recency": 15, "frequency": 10, "monetary": 1500}
]
```

### Get All Segments
```bash
GET /segments
```

---

## 🎨 Web Interface Features

### Tab 1: Single Customer Analysis
- Enter customer details
- Get instant segment classification
- View distance to cluster center
- Real-time API connection status

### Tab 2: Batch Analysis
- Upload CSV file with customer data
- Process multiple customers at once
- Download CSV template
- Drag & drop file support
- View results in table format

### Tab 3: Segments Information
- View all customer segments
- See segment characteristics
- Display cluster quality metrics
- Color-coded segment badges

---

## 🔧 Customization

### Modify Segment Characteristics

Edit `main.py` in the `SEGMENT_CHARACTERISTICS` dictionary:

```python
SEGMENT_CHARACTERISTICS = {
    0: {
        "name": "Your Custom Name",
        "description": "Description of this segment",
        "color": "#FF6B6B"
    },
    # Add more segments...
}
```

### Adjust Input Fields

If your dataset has different columns, modify:

1. **In train_model.py**: Adjust the data loading and feature selection
2. **In main.py**: Update the `SegmentRequest` model:
   ```python
   class SegmentRequest(BaseModel):
       your_feature_1: float
       your_feature_2: int
       # Add your fields...
   ```

3. **In HTML frontend**: Update the input form fields in the "Single Customer" tab

### Change Number of Clusters

In `train_model.py`, modify:
```python
optimal_k = 4  # Change this number
```

---

## 📈 Understanding Model Performance

### Silhouette Score
- Range: -1 to 1
- Higher is better (closer to 1)
- Measures how similar points are to their own cluster vs other clusters

### Davies-Bouldin Index
- Range: 0 to infinity
- Lower is better
- Measures average similarity ratio of each cluster with its most similar cluster

### Elbow Method
The training script automatically plots the elbow curve to help you determine the optimal number of clusters visually.

---

## 🐛 Troubleshooting

### Issue: "Failed to load a required model or transformer"
**Solution:** Ensure you've run `train_model.py` to generate all joblib files.

### Issue: "Connection refused" error in frontend
**Solution:** Make sure FastAPI server is running on the correct URL. Update API URL in frontend settings.

### Issue: "Invalid categorical value provided"
**Solution:** The input value for a categorical field isn't in the training data. Check your Train.csv for valid values.

### Issue: CSV batch upload fails
**Solution:** Ensure CSV format matches expected columns and order (check template).

---

## 📋 Expected Train.csv Format

Minimal example with required features:
```csv
annual_income,spending_score,age,recency,frequency,monetary
50000,50,35,30,5,500
75000,75,28,15,10,1500
30000,25,55,60,2,200
100000,90,25,5,15,3000
45000,40,50,45,3,300
```

You can add more features - the model will use all numeric columns.

---

## 🔐 Security Notes

- The CORS middleware allows all origins (`allow_origins=["*"]`) - Change this in production:
  ```python
  allow_origins=["http://localhost:3000", "https://yourdomain.com"]
  ```

- The API has no authentication - Add authentication middleware in production

---

## 📚 Next Steps

1. **Data Exploration:** Analyze Train.csv to understand customer features
2. **Feature Engineering:** Create new meaningful features
3. **Model Tuning:** Experiment with different K values and preprocessing
4. **Deployment:** Deploy to cloud (Heroku, AWS, Google Cloud, etc.)
5. **Integration:** Connect with your business systems

---

## 🎓 Learning Resources

- K-means Clustering: https://scikit-learn.org/stable/modules/clustering.html#k-means
- FastAPI: https://fastapi.tiangolo.com/
- Joblib: https://joblib.readthedocs.io/
- Scikit-learn Preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review FastAPI documentation: http://localhost:8000/docs
3. Check console/terminal for error messages
4. Verify data format and API connection

---

## 📝 License

This project is provided as-is for educational and business purposes.

---

## 🚀 Production Checklist

- [ ] Run `train_model.py` with complete dataset
- [ ] Test all API endpoints with `requests` library
- [ ] Update segment characteristics based on business insights
- [ ] Add authentication to FastAPI
- [ ] Implement logging and monitoring
- [ ] Set up environment variables for configuration
- [ ] Test frontend with different browsers
- [ ] Deploy API to production server
- [ ] Set up HTTPS/SSL certificate
- [ ] Configure CORS for your domain
- [ ] Set up automated model retraining
- [ ] Document API changes

---

**Version:** 1.0.0  
**Last Updated:** 2026  
**Framework:** FastAPI + K-means Clustering
