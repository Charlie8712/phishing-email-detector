import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

# Load cleaned dataset (before oversampling)
df = pd.read_csv("data/emails_cleaned.csv")
before_counts = df["label"].value_counts()

# Load the trained model + vectorizer
with open("models/phishing_detector_v3.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer_v3.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Recreate feature matrix for full dataset
df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
X_text = vectorizer.transform(df["text"])
X_struct = df[["url_count", "body_length", "subject_length"]].values

from scipy.sparse import hstack
X = hstack([X_text, X_struct])
y_true = df["label"]

# Predict
y_pred = model.predict(X)
y_scores = model.predict_proba(X)[:, 1]

# ==========================
# 1. CONFUSION MATRIX PLOT
# ==========================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legit", "Phishing"],
            yticklabels=["Legit", "Phishing"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("graphs/confusion_matrix", dpi=300)
plt.close()

# ==========================
# 2. CLASS DISTRIBUTION
# ==========================
after_counts = pd.Series({0: 164972, 1: 164972})  # from oversampling

plt.figure(figsize=(7, 5))
before_counts.plot(kind="bar")
plt.title("Class Distribution Before Oversampling")
plt.xlabel("Class (0=Legit, 1=Phishing)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("graphs/class_distribution_before", dpi=300)
plt.close()

plt.figure(figsize=(7, 5))
after_counts.plot(kind="bar")
plt.title("Class Distribution After Oversampling")
plt.xlabel("Class (0=Legit, 1=Phishing)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("graphs/class_distribution_after", dpi=300)
plt.close()

print("Graphs generated:")
print("- confusion_matrix")
print("- class_distribution_before")
print("- class_distribution_after")