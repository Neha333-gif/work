import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_and_save_model():
    print("="*60)
    print("STARTING FAKE NEWS DETECTION NLP MODEL TRAINING")
    print("="*60)
    
    # 1. Load Datasets
    fake_path = "fake news detection dataset/fake.csv"
    true_path = "fake news detection dataset/true.csv"
    
    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        print(f"Error: Datasets not found at {fake_path} or {true_path}")
        return
        
    print("Loading Fake News dataset...")
    df_fake = pd.read_csv(fake_path)
    df_fake['label'] = 0
    
    print("Loading True News dataset...")
    df_true = pd.read_csv(true_path)
    df_true['label'] = 1
    
    print(f"Fake news count: {len(df_fake)}, True news count: {len(df_true)}")
    
    # 2. Combine and Shuffle
    df = pd.concat([df_fake, df_true], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Handle missing values in text and title
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    
    # Combine title and text for NLP context
    print("Preprocessing text data...")
    df['content'] = df['title'] + " " + df['text']
    
    # 3. Train-Test Split
    X = df['content']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # 4. TF-IDF Vectorization
    print("Vectorizing text using TF-IDF (unigrams & bigrams, max 5000 features)...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 5. Train Model (Logistic Regression for Explainability & High Performance)
    print("Training Logistic Regression Classifier...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)
    
    # 6. Evaluation
    print("Evaluating model performance...")
    y_pred = model.predict(X_test_vec)
    
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()  # Convert to list for JSON serialization
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(np.array(cm))
    
    # 7. Extract Word Coefficients for Explainable AI
    print("Extracting feature importances/coefficients...")
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    # Create word-to-coefficient mapping
    word_coef_map = {word: float(coef) for word, coef in zip(feature_names, coefficients)}
    
    # Sort features
    sorted_features = sorted(zip(feature_names, coefficients), key=lambda x: x[1])
    
    # Top 50 Fake news indicators (highly negative coefficients)
    top_fake_words = [{"word": word, "coef": float(coef)} for word, coef in sorted_features[:50]]
    # Top 50 True news indicators (highly positive coefficients)
    top_true_words = [{"word": word, "coef": float(coef)} for word, coef in sorted_features[-50:]][::-1]
    
    # 8. Save Model and Pipeline Assets
    print("Saving trained pipeline assets...")
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    joblib.dump(model, 'logistic_regression_model.joblib')
    joblib.dump(word_coef_map, 'word_coefficients.joblib')
    
    # Save training metadata
    metadata = {
        "model_type": "TF-IDF + Logistic Regression",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "top_fake_words": top_fake_words,
        "top_true_words": top_true_words,
        "num_features": len(feature_names),
        "train_size": len(X_train),
        "test_size": len(X_test)
    }
    joblib.dump(metadata, 'model_metadata.joblib')
    
    # Save a small sample dataset with predictions for frontend testing/preview
    sample_df = df.head(100).copy()
    sample_df_vec = vectorizer.transform(sample_df['content'])
    sample_df['prediction_prob'] = model.predict_proba(sample_df_vec)[:, 1]
    sample_df['prediction'] = model.predict(sample_df_vec)
    sample_df.drop(columns=['content'], inplace=True)
    sample_df.to_csv('sample_news_predictions.csv', index=False)
    
    print("[OK] Assets saved successfully:")
    print(" - tfidf_vectorizer.joblib")
    print(" - logistic_regression_model.joblib")
    print(" - word_coefficients.joblib")
    print(" - model_metadata.joblib")
    print(" - sample_news_predictions.csv")
    print("="*60)

if __name__ == "__main__":
    train_and_save_model()
