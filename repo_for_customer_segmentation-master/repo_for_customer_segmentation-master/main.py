import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app for Customer Segmentation
app = FastAPI(title="Customer Segmentation API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and transformers
try:
    kmeans_model = joblib.load('kmeans_model.joblib')
    scaler = joblib.load('scaler.joblib')
    imputer = joblib.load('imputer.joblib')
    label_encoders = joblib.load('label_encoders.joblib')
    numeric_cols = joblib.load('numeric_cols.joblib')
    categorical_cols = joblib.load('categorical_cols.joblib')
    cluster_info = joblib.load('cluster_info.joblib')
except FileNotFoundError as e:
    raise RuntimeError(f"Failed to load required model or transformer: {e}. Make sure all joblib files are present.")

# Define segment characteristics
SEGMENT_CHARACTERISTICS = {
    0: {"name": "Premium Customers", "description": "High value, frequent buyers", "color": "#FF6B6B"},
    1: {"name": "Regular Customers", "description": "Consistent, moderate engagement", "color": "#4ECDC4"},
    2: {"name": "At-Risk Customers", "description": "Low activity, needs attention", "color": "#FFE66D"},
    3: {"name": "New Customers", "description": "Recent, high potential", "color": "#95E1D3"},
}

class SegmentRequest(BaseModel):
    """Define request schema - matches the actual Train.csv columns"""
    Gender: str = Field(..., description="Gender (Male/Female)")
    Ever_Married: str = Field(default="No", description="Ever Married (Yes/No)")
    Age: int = Field(..., description="Age of customer")
    Graduated: str = Field(default="No", description="Graduated (Yes/No)")
    Profession: str = Field(default="Artist", description="Profession")
    Work_Experience: float = Field(default=1.0, description="Work Experience in years")
    Spending_Score: str = Field(default="Low", description="Spending Score (Low/Average/High)")
    Family_Size: float = Field(default=2.0, description="Family Size")
    Var_1: str = Field(default="Cat_6", description="Anonymized category Var_1")

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok", "model": "Customer Segmentation K-means"}

@app.get("/info")
def get_info():
    """Get model information"""
    return {
        "model_type": "K-means Clustering",
        "n_clusters": cluster_info['n_clusters'],
        "silhouette_score": float(cluster_info['silhouette_score']),
        "davies_bouldin_score": float(cluster_info['davies_bouldin_score']),
        "cluster_sizes": cluster_info['cluster_sizes'],
        "segments": SEGMENT_CHARACTERISTICS
    }

def preprocess_input(data: SegmentRequest) -> pd.DataFrame:
    """Preprocess raw input data for prediction"""
    try:
        # Create DataFrame from input with correct columns and order
        input_df = pd.DataFrame({
            'Gender': [data.Gender],
            'Ever_Married': [data.Ever_Married],
            'Age': [data.Age],
            'Graduated': [data.Graduated],
            'Profession': [data.Profession],
            'Work_Experience': [data.Work_Experience],
            'Spending_Score': [data.Spending_Score],
            'Family_Size': [data.Family_Size],
            'Var_1': [data.Var_1]
        })
        
        # Encode categorical variables using the loaded label_encoders
        for col in categorical_cols:
            le = label_encoders[col]
            val = str(input_df[col].iloc[0])
            if val not in le.classes_:
                val = 'nan'
            input_df[col] = le.transform([val])
        
        # Impute missing values for numeric columns
        input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
        
        # Scale all features (scaler expects all 9 columns in correct order)
        input_scaled = scaler.transform(input_df)
        
        return input_scaled
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preprocessing input: {str(e)}")

@app.post("/segment")
def segment_customer(req: SegmentRequest):
    """Predict customer segment for a single customer"""
    try:
        X_processed = preprocess_input(req)
        
        # Predict cluster
        cluster = kmeans_model.predict(X_processed)[0]
        
        # Get segment characteristics
        segment_info = SEGMENT_CHARACTERISTICS.get(
            cluster,
            {"name": f"Segment {cluster}", "description": "Unknown segment", "color": "#999999"}
        )
        
        return {
            "cluster": int(cluster),
            "segment_name": segment_info["name"],
            "description": segment_info["description"],
            "color": segment_info["color"],
            "distance_to_center": float(np.min(
                np.sqrt(np.sum((X_processed - kmeans_model.cluster_centers_) ** 2, axis=1))
            ))
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")

@app.post("/segment/batch")
def segment_batch(requests: list[SegmentRequest]):
    """Predict customer segments for multiple customers"""
    try:
        processed_inputs = []
        for req in requests:
            try:
                processed_inputs.append(preprocess_input(req)[0])
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error processing batch: {str(e)}")
        
        X_batch_processed = np.array(processed_inputs)
        
        # Predict clusters for batch
        clusters = kmeans_model.predict(X_batch_processed)
        distances = np.min(
            np.sqrt(np.sum((X_batch_processed[:, np.newaxis, :] - kmeans_model.cluster_centers_) ** 2, axis=2)),
            axis=1
        )
        
        results = []
        for i, (cluster, distance) in enumerate(zip(clusters, distances)):
            segment_info = SEGMENT_CHARACTERISTICS.get(
                cluster,
                {"name": f"Segment {cluster}", "description": "Unknown segment", "color": "#999999"}
            )
            
            results.append({
                "index": i,
                "cluster": int(cluster),
                "segment_name": segment_info["name"],
                "description": segment_info["description"],
                "color": segment_info["color"],
                "distance_to_center": float(distance)
            })
        
        return {
            "count": len(requests),
            "results": results
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch segmentation failed: {str(e)}")

@app.get("/segments")
def get_segments():
    """Get all segment characteristics"""
    return {
        "total_segments": cluster_info['n_clusters'],
        "segments": SEGMENT_CHARACTERISTICS
    }

from fastapi.responses import HTMLResponse

def get_strategy(slug: str) -> str:
    strategies = {
        "premium": "Premium customers represent your highest value group with high spending scores. Focus on exclusive VIP product offerings, personalized premier client support, early access to new collections, and retention/fidelity perks to maximize customer lifetime value (LTV).",
        "regular": "Regular customers are the core backbone of your business, demonstrating moderate, consistent engagement. Target them with tailored loyalty programs, dynamic product recommendations matching their past interests, and incremental upselling campaigns to nudge them into premium behavior.",
        "at-risk": "At-Risk customers show low activity and spend scores. Prioritize them with urgent win-back email campaigns, highly enticing time-sensitive discounts, and feedback surveys to understand their pain points and prevent customer churn.",
        "new": "New customers are highly prospective with recent sign-ups and high growth potential. Focus on a warm welcome onboarding sequence, introductory discounts for second purchases, educational content about your products, and customer success touchpoints."
    }
    return strategies.get(slug, "Standard marketing engagement and product feedback campaigns.")

def generate_table_rows(samples: list) -> str:
    rows = []
    for s in samples:
        rows.append(f"""
        <tr>
          <td>{s.get('Gender', 'N/A')}</td>
          <td>{s.get('Ever_Married', 'N/A')}</td>
          <td>{int(s.get('Age', 0))}</td>
          <td>{s.get('Graduated', 'N/A')}</td>
          <td>{s.get('Profession', 'N/A')}</td>
          <td>{f"{s.get('Work_Experience'):.0f} yrs" if pd.notna(s.get('Work_Experience')) else 'N/A'}</td>
          <td>{s.get('Spending_Score', 'N/A')}</td>
          <td>{f"{s.get('Family_Size'):.0f}" if pd.notna(s.get('Family_Size')) else 'N/A'}</td>
          <td>{s.get('Var_1', 'N/A')}</td>
        </tr>
        """)
    return "\n".join(rows) if rows else "<tr><td colspan='9' style='text-align: center;'>No sample data available.</td></tr>"

@app.get("/view/{segment_name}", response_class=HTMLResponse)
def view_segment_dashboard(segment_name: str):
    segment_name = segment_name.lower().strip()
    slug_map = {
        "premium": 0,
        "regular": 1,
        "at-risk": 2,
        "new": 3
    }
    
    if segment_name not in slug_map:
        raise HTTPException(status_code=404, detail=f"Segment '{segment_name}' not found. Choose from: premium, regular, at-risk, new.")
        
    cluster_id = slug_map[segment_name]
    char = SEGMENT_CHARACTERISTICS[cluster_id]
    
    # Load clustered data dynamically
    try:
        df = pd.read_csv('clustered_data.csv')
        segment_df = df[df['Cluster'] == cluster_id]
        
        # Calculate dynamic metrics
        total_size = len(segment_df)
        avg_age = float(segment_df['Age'].mean())
        avg_exp = float(segment_df['Work_Experience'].dropna().mean()) if 'Work_Experience' in segment_df else 0.0
        avg_fam = float(segment_df['Family_Size'].dropna().mean()) if 'Family_Size' in segment_df else 0.0
        
        # Top profession
        top_prof = "Unknown"
        if 'Profession' in segment_df and not segment_df['Profession'].dropna().empty:
            top_prof = str(segment_df['Profession'].dropna().mode().iloc[0])
            
        # Top spending score
        top_spend = "Unknown"
        if 'Spending_Score' in segment_df and not segment_df['Spending_Score'].dropna().empty:
            top_spend = str(segment_df['Spending_Score'].dropna().mode().iloc[0])
            
        # Top 15 sample customers
        sample_customers = segment_df.head(15).to_dict(orient='records')
        
    except Exception as e:
        total_size = cluster_info['cluster_sizes'].get(str(cluster_id), 0)
        avg_age = 0.0
        avg_exp = 0.0
        avg_fam = 0.0
        top_prof = "N/A"
        top_spend = "N/A"
        sample_customers = []
        
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>{char['name']} Segment Dashboard</title>
      <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
      <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
          font-family: 'Outfit', sans-serif;
          background-color: #0f172a;
          color: #f8fafc;
          min-height: 100vh;
          padding: 40px 20px;
        }}
        .container {{
          max-width: 1200px;
          margin: 0 auto;
        }}
        .back-link {{
          display: inline-block;
          margin-bottom: 20px;
          color: #94a3b8;
          text-decoration: none;
          font-weight: 500;
          transition: color 0.2s;
        }}
        .back-link:hover {{
          color: {char['color']};
        }}
        header {{
          text-align: center;
          margin-bottom: 40px;
          border-bottom: 2px solid #1e293b;
          padding-bottom: 24px;
        }}
        .badge {{
          display: inline-block;
          padding: 6px 14px;
          background-color: {char['color']}22;
          border: 1px dashed {char['color']};
          color: {char['color']};
          border-radius: 50px;
          font-size: 14px;
          font-weight: 700;
          margin-bottom: 12px;
          text-transform: uppercase;
          letter-spacing: 0.1em;
        }}
        h1 {{
          font-size: 38px;
          font-weight: 700;
          color: #ffffff;
          margin-bottom: 10px;
          text-shadow: 0 0 20px {char['color']}33;
        }}
        .tagline {{
          font-size: 16px;
          color: #94a3b8;
        }}
        .grid {{
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          gap: 20px;
          margin-bottom: 40px;
        }}
        .kpi-card {{
          background: #1e293b88;
          border: 1px solid #334155;
          border-radius: 16px;
          padding: 24px;
          text-align: center;
          box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
          backdrop-filter: blur(5px);
          transition: transform 0.2s, border-color 0.2s;
        }}
        .kpi-card:hover {{
          transform: translateY(-4px);
          border-color: {char['color']};
        }}
        .kpi-label {{
          font-size: 13px;
          color: #94a3b8;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          margin-bottom: 8px;
        }}
        .kpi-value {{
          font-size: 28px;
          font-weight: 700;
          color: #ffffff;
        }}
        .section-card {{
          background: #1e293b88;
          border: 1px solid #334155;
          border-radius: 16px;
          padding: 30px;
          margin-bottom: 40px;
          box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
          backdrop-filter: blur(5px);
        }}
        h2 {{
          font-size: 20px;
          font-weight: 600;
          color: #ffffff;
          margin-bottom: 20px;
          border-left: 4px solid {char['color']};
          padding-left: 12px;
        }}
        .table-container {{
          overflow-x: auto;
        }}
        table {{
          width: 100%;
          border-collapse: collapse;
          text-align: left;
        }}
        th {{
          padding: 12px 16px;
          border-bottom: 2px solid #334155;
          color: #94a3b8;
          font-weight: 600;
          font-size: 14px;
        }}
        td {{
          padding: 12px 16px;
          border-bottom: 1px solid #334155;
          color: #cbd5e1;
          font-size: 14px;
        }}
        tr:hover td {{
          background-color: #33415533;
        }}
        .strategy-grid {{
          display: grid;
          grid-template-columns: 1fr;
          gap: 20px;
        }}
        .strategy-item {{
          background: #0f172a88;
          border-left: 4px solid {char['color']};
          border-radius: 0 12px 12px 0;
          padding: 16px 20px;
        }}
        .strategy-title {{
          font-weight: 600;
          color: #ffffff;
          margin-bottom: 6px;
        }}
        .strategy-desc {{
          font-size: 14px;
          color: #94a3b8;
          line-height: 1.5;
        }}
      </style>
    </head>
    <body>
      <div class="container">
        <a href="http://localhost:8080/customer_segmentation_frontend.html" class="back-link">← Back to Main Analyzer</a>
        
        <header>
          <div class="badge">{segment_name} segment</div>
          <h1>{char['name']}</h1>
          <p class="tagline">{char['description']}</p>
        </header>
        
        <div class="grid">
          <div class="kpi-card">
            <div class="kpi-label">Segment Size</div>
            <div class="kpi-value">{total_size:,}</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Average Age</div>
            <div class="kpi-value">{avg_age:.1f} yrs</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Avg Work Exp</div>
            <div class="kpi-value">{avg_exp:.1f} yrs</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Avg Family Size</div>
            <div class="kpi-value">{avg_fam:.1f}</div>
          </div>
        </div>
        
        <div class="grid" style="grid-template-columns: 1fr 1fr;">
          <div class="kpi-card">
            <div class="kpi-label">Top Profession</div>
            <div class="kpi-value" style="font-size: 22px; color: {char['color']};">{top_prof}</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Top Spending Score</div>
            <div class="kpi-value" style="font-size: 22px; color: {char['color']};">{top_spend}</div>
          </div>
        </div>
        
        <div class="section-card">
          <h2>Strategic Business Recommendations</h2>
          <div class="strategy-grid">
            <div class="strategy-item">
              <div class="strategy-title">Targeted Outreach & Campaigns</div>
              <div class="strategy-desc">{get_strategy(segment_name)}</div>
            </div>
          </div>
        </div>
        
        <div class="section-card">
          <h2>Representative Customers (Sample of 15)</h2>
          <div class="table-container">
            <table>
              <thead>
                <tr>
                  <th>Gender</th>
                  <th>Ever Married</th>
                  <th>Age</th>
                  <th>Graduated</th>
                  <th>Profession</th>
                  <th>Work Experience</th>
                  <th>Spending Score</th>
                  <th>Family Size</th>
                  <th>Var_1</th>
                </tr>
              </thead>
              <tbody>
                {generate_table_rows(sample_customers)}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </body>
    </html>
    """
    return html_content
