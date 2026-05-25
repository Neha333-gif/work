import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib
import warnings

warnings.filterwarnings("ignore")

def train_and_save_model():
    """
    Train a K-means clustering model for customer segmentation
    """
    print("Loading data...")
    # Load your dataset (Train.csv)
    df = pd.read_csv("Train.csv")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Create a copy for processing
    df_processed = df.copy()
    # Drop ID and Segmentation columns if they exist as they are not clustering features
    cols_to_drop = [c for c in ['ID', 'Segmentation'] if c in df_processed.columns]
    if cols_to_drop:
        df_processed.drop(columns=cols_to_drop, inplace=True)
        print(f"Dropped non-features: {cols_to_drop}")
    
    # Display basic info
    print(f"\nDataset Info:")
    print(df_processed.info())
    print(f"\nBasic Statistics:")
    print(df_processed.describe())
    
    # Handle missing values
    print("\nHandling missing values...")
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    
    # Identify numeric and categorical columns
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    # Encode categorical variables
    label_encoders = {}
    df_encoded = df_processed.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded {col}")
    
    # Impute numeric columns
    df_encoded[numeric_cols] = imputer.fit_transform(df_encoded[numeric_cols])
    
    # Scale all features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)
    
    # Determine optimal number of clusters using elbow method
    print("\nDetermining optimal number of clusters...")
    inertias = []
    silhouette_scores = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette Score={silhouette_scores[-1]:.3f}")
    
    # Use optimal k (you can adjust this based on your analysis)
    optimal_k = 4  # Default, can be changed based on elbow method
    print(f"\nUsing optimal K={optimal_k}")
    
    # Train final model
    print(f"\nTraining K-means model with {optimal_k} clusters...")
    kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans_model.fit_predict(X_scaled)
    
    # Calculate metrics
    silhouette = silhouette_score(X_scaled, clusters)
    davies_bouldin = davies_bouldin_score(X_scaled, clusters)
    
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"Davies-Bouldin Index: {davies_bouldin:.3f}")
    
    # Add cluster labels to original dataframe
    df_processed['Cluster'] = clusters
    print(f"\nCluster Distribution:")
    print(df_processed['Cluster'].value_counts().sort_index())
    
    # Save model and transformers
    print("\nSaving model and transformers...")
    joblib.dump(kmeans_model, 'kmeans_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(imputer, 'imputer.joblib')
    joblib.dump(label_encoders, 'label_encoders.joblib')
    joblib.dump(numeric_cols, 'numeric_cols.joblib')
    joblib.dump(categorical_cols, 'categorical_cols.joblib')
    
    print("[OK] Model saved as kmeans_model.joblib")
    print("[OK] Scaler saved as scaler.joblib")
    print("[OK] Imputer saved as imputer.joblib")
    print("[OK] Label encoders saved as label_encoders.joblib")
    print("[OK] Feature info saved as numeric_cols.joblib and categorical_cols.joblib")
    
    # Save cluster info
    cluster_info = {
        'n_clusters': optimal_k,
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'cluster_sizes': df_processed['Cluster'].value_counts().to_dict()
    }
    joblib.dump(cluster_info, 'cluster_info.joblib')
    
    # Save clustered dataset
    df_processed.to_csv('clustered_data.csv', index=False)
    print("[OK] Clustered dataset saved as clustered_data.csv")
    
    return kmeans_model, scaler, imputer, label_encoders, optimal_k

if __name__ == "__main__":
    train_and_save_model()
