# 🎯 Customer Segmentation Project - Step-by-Step Implementation Guide

## Phase 1: Setup (5 minutes)

### Step 1.1: Prepare Project Directory
```bash
mkdir customer-segmentation-project
cd customer-segmentation-project
```

### Step 1.2: Copy Your Files
Place these files in your project directory:
- `Train.csv` - Your training dataset
- `Test.csv` - Your test dataset

### Step 1.3: Create Python Files
Download and save these three Python/Web files:
- `train_model.py` - Model training script
- `main.py` - FastAPI backend server
- `customer_segmentation_frontend.html` - Web interface
- `requirements.txt` - Dependencies list

### Step 1.4: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Phase 2: Model Training (10-15 minutes)

### Step 2.1: Inspect Your Data
First, take a quick look at your data:
```python
import pandas as pd
df = pd.read_csv('Train.csv')
print(df.head())
print(df.info())
print(df.describe())
```

### Step 2.2: Run Training Script
```bash
python train_model.py
```

**What happens:**
1. Loads your Train.csv file
2. Handles missing values
3. Encodes categorical features
4. Scales numerical features
5. Finds optimal number of clusters (2-10)
6. Trains K-means model
7. Saves 8 joblib files (model + transformers)
8. Outputs cluster statistics

**Check console output for:**
- ✓ Dataset shape
- ✓ Optimal K value selected
- ✓ Silhouette Score (higher is better, -1 to 1)
- ✓ Davies-Bouldin Index (lower is better)
- ✓ Cluster size distribution

**Generated files to verify:**
```
✓ kmeans_model.joblib          (200 KB - 2 MB)
✓ scaler.joblib                (1-5 KB)
✓ imputer.joblib               (1-5 KB)
✓ label_encoders.joblib        (10-50 KB)
✓ cluster_info.joblib          (1 KB)
✓ numeric_cols.joblib          (1 KB)
✓ categorical_cols.joblib      (1 KB)
✓ clustered_data.csv           (same size as input)
```

---

## Phase 3: Start API Server (5 minutes)

### Step 3.1: Launch FastAPI
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

### Step 3.2: Verify API is Running

Open in your browser:
- http://localhost:8000/health → Should show `{"status": "ok"}`
- http://localhost:8000/docs → Interactive API documentation
- http://localhost:8000/redoc → Alternative documentation

### Step 3.3: Test API with cURL

```bash
# Test single customer segmentation
curl -X POST http://localhost:8000/segment \
  -H "Content-Type: application/json" \
  -d '{
    "annual_income": 50000,
    "spending_score": 50,
    "age": 35,
    "recency": 30,
    "frequency": 5,
    "monetary": 500
  }'
```

Expected response:
```json
{
  "cluster": 1,
  "segment_name": "Regular Customers",
  "description": "Consistent, moderate engagement",
  "color": "#4ECDC4",
  "distance_to_center": 2.34
}
```

---

## Phase 4: Use Web Interface (5 minutes)

### Step 4.1: Open Frontend
Simply open `customer_segmentation_frontend.html` in your web browser.

Or run a simple server:
```bash
# In a new terminal, from project directory
python -m http.server 8080
```

Then visit: `http://localhost:8080/customer_segmentation_frontend.html`

### Step 4.2: Configure API URL
In the web interface:
1. Go to "Single Customer" tab
2. Set "API Base URL" to: `http://localhost:8000`
3. Click "Analyze Customer" to test

### Step 4.3: Analyze Customers

**Tab 1: Single Customer**
- Enter individual customer details
- Get instant segment prediction
- See distance to cluster center

**Tab 2: Batch Analysis**
- Upload CSV file with multiple customers
- Download template CSV first
- Upload and process
- View all results

**Tab 3: Segments Info**
- Click "Load Segment Info"
- View all cluster characteristics
- See model performance metrics
- Understand segment distribution

---

## Phase 5: Customization (Optional)

### Step 5.1: Modify Input Fields

If your CSV has different columns, update **main.py**:

```python
# Find this class
class SegmentRequest(BaseModel):
    annual_income: float = Field(...)
    spending_score: int = Field(...)
    age: int = Field(...)
    # Replace with your actual columns
    
# Change to match your columns:
class SegmentRequest(BaseModel):
    customer_lifetime_value: float = Field(...)
    purchase_frequency: int = Field(...)
    days_since_signup: int = Field(...)
```

Also update in **train_model.py** the feature selection logic.

### Step 5.2: Rename Customer Segments

In **main.py**, find and edit:
```python
SEGMENT_CHARACTERISTICS = {
    0: {"name": "Premium Customers", ...},
    1: {"name": "Regular Customers", ...},
    2: {"name": "At-Risk Customers", ...},
    3: {"name": "New Customers", ...},
}
```

### Step 5.3: Change Number of Clusters

In **train_model.py**:
```python
optimal_k = 4  # Change this to 3, 5, 6, etc.
```

Then retrain:
```bash
python train_model.py
```

---

## Phase 6: Testing & Validation

### Step 6.1: Test with Sample Data

Create `test_customers.csv`:
```csv
annual_income,spending_score,age,recency,frequency,monetary
60000,65,30,20,7,800
90000,85,26,8,14,2200
35000,30,58,70,1,100
```

Upload via web interface's Batch Analysis tab.

### Step 6.2: Validate Model Quality

Check `cluster_info.joblib` metrics:

```python
import joblib
info = joblib.load('cluster_info.joblib')
print(f"Silhouette Score: {info['silhouette_score']:.3f}")
print(f"Cluster Sizes: {info['cluster_sizes']}")
```

**Good signs:**
- Silhouette Score > 0.5 (excellent)
- Balanced cluster sizes (not all in one cluster)
- Davies-Bouldin < 1.5 (good separation)

### Step 6.3: Analyze Clustered Data

```python
import pandas as pd
df_clustered = pd.read_csv('clustered_data.csv')
print(df_clustered.groupby('Cluster').size())
print(df_clustered.groupby('Cluster').mean())
```

---

## Troubleshooting Guide

### ❌ Problem: "ModuleNotFoundError: No module named 'fastapi'"
**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### ❌ Problem: "FileNotFoundError: xg_boost_model.joblib"
**Solution:** Run training first
```bash
python train_model.py
```

### ❌ Problem: "Connection refused" in browser
**Solution:** Start API server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### ❌ Problem: API returns "Invalid categorical value"
**Solution:** Check your input matches training data values

### ❌ Problem: "Port 8000 is already in use"
**Solution:** Use different port
```bash
uvicorn main:app --port 8001
```

---

## Common Use Cases

### Use Case 1: Customer Marketing Segments
Upload customer list → Get segment classification → Create targeted campaigns

### Use Case 2: Product Recommendations
Know customer segment → Recommend products for that segment

### Use Case 3: Churn Risk Detection
Analyze "At-Risk" segment → Create retention campaigns

### Use Case 4: Revenue Optimization
Identify "Premium" segment → Increase engagement with high-value customers

---

## File Reference

| File | Purpose | When to Edit |
|------|---------|--------------|
| `train_model.py` | Train K-means model | Different data columns, change K |
| `main.py` | API server | Add endpoints, modify fields, update segments |
| `customer_segmentation_frontend.html` | Web UI | Change colors, add fields, modify layout |
| `requirements.txt` | Dependencies | Add/remove libraries (rarely) |

---

## Performance Tips

1. **Faster Training:** Remove unnecessary features from Train.csv
2. **Better Segments:** Ensure good data quality, handle outliers
3. **API Speed:** Run on local machine, optimize with caching
4. **Frontend:** Cache API responses to reduce requests

---

## Next Steps After Setup

1. ✓ Train model with your actual data
2. ✓ Validate segment quality with business team
3. ✓ Adjust segment names to match your business
4. ✓ Create marketing strategies for each segment
5. ✓ Monitor segment changes over time
6. ✓ Schedule monthly retraining with new data
7. → Deploy to production (AWS, Heroku, etc.)

---

## Quick Command Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train_model.py

# Start API server
uvicorn main:app --reload

# Start local web server
python -m http.server 8080

# Test API
curl -X GET http://localhost:8000/health

# View API docs
open http://localhost:8000/docs
```

---

## Estimated Timeline

| Phase | Time | Status |
|-------|------|--------|
| Setup | 5 min | ⏳ |
| Training | 15 min | ⏳ |
| API Server | 2 min | ⏳ |
| Testing | 5 min | ⏳ |
| **Total** | **~30 min** | ✓ |

---

## Key Metrics to Track

- **Silhouette Score:** Cluster quality (higher is better)
- **Davies-Bouldin Index:** Cluster separation (lower is better)
- **Cluster Sizes:** Distribution across segments
- **Distance to Center:** Customer's closeness to cluster center

---

## Success Criteria

✓ Model trains without errors  
✓ API server runs on http://localhost:8000  
✓ Web interface loads in browser  
✓ Can analyze single customer  
✓ Can upload batch CSV file  
✓ Segment info displays correctly  

---

**You're all set! Happy segmenting! 🚀**
