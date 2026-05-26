import pandas as pd
import os

csv_path = "job_salary_dataset/Clean_Dataset.csv"
if not os.path.exists(csv_path):
    print("CSV does not exist!")
else:
    df = pd.read_csv(csv_path, nrows=100)
    print("Columns:", df.columns.tolist())
    print("\nHead:\n", df.head())
    
    # Let's inspect object columns
    full_df = pd.read_csv(csv_path)
    print("\nShape of dataset:", full_df.shape)
    for col in full_df.columns:
        if full_df[col].dtype == "object":
            print(f"Column '{col}' unique values (first 10):", full_df[col].unique()[:10].tolist())
        else:
            print(f"Column '{col}' numeric range: {full_df[col].min()} to {full_df[col].max()}")
