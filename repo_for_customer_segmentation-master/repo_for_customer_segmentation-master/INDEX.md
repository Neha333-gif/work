# 📦 Customer Segmentation Project - Complete Package

## 📋 Project Overview

This is a **complete, production-ready customer segmentation system** built on the same architecture as your retail sales prediction project. It uses **K-means clustering** to automatically segment customers into meaningful groups.

**Framework:** FastAPI + K-means (scikit-learn)  
**Architecture:** Model Training → REST API → Web Interface  
**Time to Deploy:** ~30 minutes  

---

## 📂 All Files in This Package

### Core Application Files

| File | Size | Purpose | Status |
|------|------|---------|--------|
| `train_model.py` | 4.5 KB | Train K-means clustering model | ✓ Ready |
| `main.py` | 6.3 KB | FastAPI backend server | ✓ Ready |
| `customer_segmentation_frontend.html` | 21 KB | Interactive web interface | ✓ Ready |
| `requirements.txt` | 153 B | Python dependencies | ✓ Ready |

### Documentation Files

| File | Size | Purpose |
|------|------|---------|
| `README.md` | 8.7 KB | Complete reference guide |
| `SETUP_GUIDE.md` | 9.0 KB | Step-by-step setup instructions |
| `PROJECT_COMPARISON.md` | 9.5 KB | Comparison with retail sales project |
| `INDEX.md` | This file | Project overview & quick reference |

---

## 🚀 Quick Start (5 Steps)

### 1️⃣ **Prepare Files**
```bash
# Create project directory
mkdir customer-segmentation
cd customer-segmentation

# Copy all 4 core files here:
# - train_model.py
# - main.py
# - customer_segmentation_frontend.html
# - requirements.txt

# Place your data files:
# - Train.csv (your training dataset)
# - Test.csv (your test dataset)
```

### 2️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3️⃣ **Train Model**
```bash
python train_model.py
```
This creates 8 joblib files automatically.

### 4️⃣ **Start API Server**
```bash
uvicorn main:app --reload
```

### 5️⃣ **Open Web Interface**
Simply open `customer_segmentation_frontend.html` in your browser, or:
```bash
python -m http.server 8080
# Then visit: http://localhost:8080/customer_segmentation_frontend.html
```

**That's it! You're done! 🎉**

---

## 📖 Documentation Guide

### For First-Time Setup
**→ Start with:** `SETUP_GUIDE.md`
- Complete step-by-step walkthrough
- Expected outputs at each stage
- Troubleshooting common issues
- Real command examples to copy-paste

### For Understanding Architecture
**→ Read:** `README.md`
- What each file does
- How components interact
- API endpoint reference
- Customization options

### For Comparing Approaches
**→ Check:** `PROJECT_COMPARISON.md`
- How this differs from retail sales project
- When to use clustering vs regression
- File-by-file differences
- Migration guide if needed

---

## 🎯 Key Features

### Web Interface Capabilities

**Tab 1: Single Customer Analysis**
- Enter customer details in real-time
- Get instant segment classification
- View distance to cluster center
- See segment characteristics

**Tab 2: Batch Customer Processing**
- Upload CSV file with multiple customers
- Download template CSV first
- Process 100s of customers at once
- View all results in table format
- Drag & drop file support

**Tab 3: Segment Information**
- View all customer segments
- See cluster quality metrics
- Understand segment sizes
- Color-coded segment badges

### API Features

**5 REST Endpoints:**
```
GET  /health              → Server status
GET  /info                → Model metrics & segments
POST /segment             → Single customer segmentation
POST /segment/batch       → Batch customer segmentation
GET  /segments            → All segment definitions
```

**Automated API Documentation:**
- Interactive Swagger UI: `http://localhost:8000/docs`
- ReDoc Alternative: `http://localhost:8000/redoc`

---

## 📊 What Gets Created

After running `train_model.py`, you'll have:

### Model Files (Auto-generated)
```
kmeans_model.joblib       ← Trained clustering model
scaler.joblib             ← Feature scaling transformer
imputer.joblib            ← Missing value handler
label_encoders.joblib     ← Categorical encoders
cluster_info.joblib       ← Cluster statistics
numeric_cols.joblib       ← Feature names
categorical_cols.joblib   ← Feature names
clustered_data.csv        ← Original data with cluster labels
```

### What They Do
- `kmeans_model.joblib` - Makes predictions (cluster assignments)
- `scaler.joblib` - Normalizes input features
- `imputer.joblib` - Handles missing values
- `label_encoders.joblib` - Converts text to numbers
- `cluster_info.joblib` - Stores model performance metrics

---

## 🔧 Customization Examples

### Change Number of Clusters
In `train_model.py`, line ~60:
```python
optimal_k = 4  # Change to 3, 5, 6, etc.
```

### Rename Customer Segments
In `main.py`, lines ~30-40:
```python
SEGMENT_CHARACTERISTICS = {
    0: {"name": "VIP Customers", "description": "..."},
    1: {"name": "Regular Users", "description": "..."},
    # Customize as needed
}
```

### Add/Change Input Fields
In `main.py`, class `SegmentRequest`:
```python
class SegmentRequest(BaseModel):
    annual_income: float = Field(...)
    spending_score: int = Field(...)
    # Add your own fields matching your CSV
```

Also update in `train_model.py` data loading section.

---

## 📈 Model Quality Metrics

The system automatically calculates:

**Silhouette Score** (0.0 to 1.0)
- Measures cluster quality
- Higher = better separated clusters
- Value > 0.5 is good
- Value > 0.7 is excellent

**Davies-Bouldin Index** (0 to ∞)
- Measures cluster separation
- Lower = better defined clusters
- Value < 1.0 is good
- Value < 0.5 is excellent

**Cluster Distribution**
- Shows size of each customer segment
- Ideally balanced (not all in one cluster)
- If imbalanced, consider adjusting K

---

## 🐛 Troubleshooting Quick Reference

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run: `pip install -r requirements.txt` |
| Model files not found | Run: `python train_model.py` |
| API connection fails | Check URL in frontend, start server |
| Port 8000 in use | Use: `uvicorn main:app --port 8001` |
| CSV upload fails | Check CSV format matches template |
| Slow predictions | Run on local machine, not remote |

See `SETUP_GUIDE.md` for detailed troubleshooting.

---

## 📱 API Testing Examples

### Using Python requests library
```python
import requests

# Single customer
response = requests.post(
    'http://localhost:8000/segment',
    json={
        'annual_income': 50000,
        'spending_score': 50,
        'age': 35,
        'recency': 30,
        'frequency': 5,
        'monetary': 500
    }
)
print(response.json())

# Batch
customers = [
    {'annual_income': 50000, 'spending_score': 50, ...},
    {'annual_income': 75000, 'spending_score': 75, ...},
]
response = requests.post(
    'http://localhost:8000/segment/batch',
    json=customers
)
print(response.json())
```

### Using cURL
```bash
curl -X POST http://localhost:8000/segment \
  -H "Content-Type: application/json" \
  -d '{"annual_income":50000,"spending_score":50,"age":35,"recency":30,"frequency":5,"monetary":500}'
```

---

## 🎓 Learning Objectives

After completing this project, you'll understand:

✓ K-means clustering algorithm  
✓ Unsupervised machine learning  
✓ Feature scaling and normalization  
✓ Building REST APIs with FastAPI  
✓ Model serialization with joblib  
✓ Batch processing in ML  
✓ Web interface development  
✓ Model deployment patterns  

---

## 📊 Expected Output Examples

### Training Output
```
Dataset shape: (200, 6)
Optimal K found: 4
Silhouette Score: 0.658
Davies-Bouldin Index: 0.847
Cluster Sizes: {0: 45, 1: 52, 2: 48, 3: 55}
✓ Model saved as kmeans_model.joblib
```

### API Response
```json
{
  "cluster": 1,
  "segment_name": "Regular Customers",
  "description": "Consistent, moderate engagement",
  "color": "#4ECDC4",
  "distance_to_center": 2.34
}
```

### Batch Response
```json
{
  "count": 3,
  "results": [
    {"index": 0, "cluster": 1, "segment_name": "Regular Customers", ...},
    {"index": 1, "cluster": 0, "segment_name": "Premium Customers", ...},
    {"index": 2, "cluster": 2, "segment_name": "At-Risk Customers", ...}
  ]
}
```

---

## 🚀 Next Steps

After successful setup:

1. **Validate Results**
   - Analyze generated `clustered_data.csv`
   - Verify cluster characteristics make business sense

2. **Customize Segments**
   - Rename segments based on your domain
   - Update segment descriptions
   - Adjust colors in frontend

3. **Test Thoroughly**
   - Try various input combinations
   - Test batch upload with large files
   - Verify API performance

4. **Deploy**
   - Move to production server
   - Set up HTTPS/SSL
   - Configure CORS for your domain
   - Add authentication

5. **Monitor**
   - Track segment stability over time
   - Schedule monthly retraining
   - Monitor prediction accuracy

---

## 📋 File Contents Summary

### train_model.py (~180 lines)
- Loads Train.csv
- Handles missing values
- Encodes categorical features
- Scales numerical features
- Finds optimal K (2-10)
- Trains K-means model
- Calculates quality metrics
- Saves 8 joblib files

### main.py (~200 lines)
- FastAPI application setup
- CORS configuration
- Model loading from joblib
- 5 API endpoints
- Preprocessing pipeline
- Single and batch prediction
- Segment characteristics

### customer_segmentation_frontend.html (~600 lines)
- 3-tab interface
- Single customer form
- Batch CSV upload
- Segment info display
- Real-time API integration
- Error handling
- Result visualization

### requirements.txt (9 lines)
- fastapi==0.104.1
- uvicorn==0.24.0
- pandas==2.0.3
- numpy==1.24.3
- scikit-learn==1.3.0
- joblib==1.3.2
- And 3 more essential libraries

---

## 💾 System Requirements

**Minimum:**
- Python 3.7+
- 512 MB RAM
- 100 MB disk space
- Any OS (Windows, Mac, Linux)

**Recommended:**
- Python 3.9+
- 2+ GB RAM
- 500 MB disk space
- Modern browser (Chrome, Firefox, Safari)

---

## 📞 Getting Help

1. **For setup issues** → See `SETUP_GUIDE.md`
2. **For API help** → Visit http://localhost:8000/docs
3. **For customization** → Check `README.md` customization section
4. **For architecture questions** → Read `PROJECT_COMPARISON.md`
5. **For errors** → Check troubleshooting in `SETUP_GUIDE.md`

---

## ✅ Success Checklist

- [ ] All 4 core files copied to project directory
- [ ] `requirements.txt` dependencies installed
- [ ] `Train.csv` and `Test.csv` in project directory
- [ ] `python train_model.py` completed successfully
- [ ] 8 joblib files created in project directory
- [ ] `uvicorn main:app --reload` running without errors
- [ ] Can access http://localhost:8000/docs
- [ ] Can open `customer_segmentation_frontend.html` in browser
- [ ] Can enter customer details and get segmentation result
- [ ] Can upload CSV and process batch
- [ ] Can view segment information

**All checked? You're ready to use the system! 🎉**

---

## 📚 Additional Resources

- **K-means Clustering:** https://scikit-learn.org/stable/modules/clustering.html#k-means
- **FastAPI Tutorial:** https://fastapi.tiangolo.com/
- **Joblib Docs:** https://joblib.readthedocs.io/
- **scikit-learn Preprocessing:** https://scikit-learn.org/stable/modules/preprocessing.html

---

## 📝 Version Information

- **Project Version:** 1.0.0
- **Created:** May 2026
- **Framework:** FastAPI + scikit-learn
- **Python:** 3.7+
- **License:** Open source

---

## 🎯 Key Takeaways

This project demonstrates:

1. **Complete ML Pipeline** - From raw data to production API
2. **Best Practices** - Proper preprocessing, serialization, error handling
3. **User-Friendly** - Web interface for non-technical users
4. **Scalable** - Handles single and batch predictions
5. **Maintainable** - Clear code structure, comprehensive documentation

It's a **template for any ML classification/clustering project** you want to build.

---

## 🎓 What You've Learned

✓ How machine learning models are trained and saved  
✓ How to build REST APIs for model serving  
✓ How to create interactive web interfaces  
✓ How to handle batch processing  
✓ Industry best practices for ML deployment  

---

## 🚀 Ready to Begin?

1. **Start with:** `SETUP_GUIDE.md` (step-by-step)
2. **Reference:** `README.md` (detailed docs)
3. **Compare:** `PROJECT_COMPARISON.md` (if needed)
4. **Deploy:** Follow production checklist in README

**Estimated total time: 30 minutes from download to live system!**

---

**Happy Segmenting! 🎯**

For the best experience, follow the steps in `SETUP_GUIDE.md` exactly as written.
