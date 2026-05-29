import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pickle

# Load cleaned data
df = pd.read_csv("data/emails_cleaned.csv")

# Text combination
df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_text = vectorizer.fit_transform(df["text"])

# STRUCTURAL FEATURES
X_struct = df[["url_count", "body_length", "subject_length"]].values

# Combine (TF-IDF + numbers)
X = hstack([X_text, X_struct])
y = df["label"]

#Oversampling
print("\nBefore Oversampling:", dict(zip(*np.unique(y, return_counts=True))))

ros = RandomOverSampler(random_state=42)
X, y = ros.fit_resample(X, y)

print("After Oversampling:", dict(zip(*np.unique(y, return_counts=True))))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)

# Predict & evaluate
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print("\n===== STRUCTURE + TEXT RESULTS =====")
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

# Save improved model
with open("models/phishing_detector_v3.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizer_v3.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nBalanced model saved as v3")
