import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Load data
df = pd.read_csv("data/emails_cleaned.csv")

# Combine text features (subject + body)
df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # limit for speed
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print("\n===== RESULTS =====")
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

# Save model + vectorizer
with open("models/phishing_detector.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel saved in /models/")
