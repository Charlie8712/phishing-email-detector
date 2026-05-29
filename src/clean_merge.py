import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# FIRST TIME ONLY — download stopwords
nltk.download('stopwords')

ham = pd.read_csv("data/ham_clean.csv")
phish = pd.read_csv("data/phishing_clean.csv")

# CLEANING FUNCTIONS

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)  # replace URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)         # remove symbols
    text = re.sub(r"\s+", " ", text)                 # collapse spaces

    # Remove stopwords (the, and, is, etc.)
    stop_words = set(stopwords.words("english"))
    words = [w for w in text.split() if w not in stop_words]

    return " ".join(words)

# Apply cleaning
ham["subject"] = ham["subject"].apply(clean_text)
ham["body"] = ham["body"].apply(clean_text)
phish["subject"] = phish["subject"].apply(clean_text)
phish["body"] = phish["body"].apply(clean_text)

# STRUCTURAL FEATURES

def count_urls(text):
    return len(re.findall(r"http\S+|www\S+", text, flags=re.IGNORECASE))

ham["url_count"] = ham["body"].apply(count_urls)
phish["url_count"] = phish["body"].apply(count_urls)

ham["subject_length"] = ham["subject"].str.len()
phish["subject_length"] = phish["subject"].str.len()

ham["body_length"] = ham["body"].str.split().str.len()
phish["body_length"] = phish["body"].str.split().str.len()

# MERGE & SAVE DATASET

combined = pd.concat([ham, phish], ignore_index=True)
combined = combined.sample(frac=1, random_state=42)  # shuffle

combined.to_csv("data/emails_cleaned.csv", index=False)
print("Dataset created → data/emails_cleaned.csv")
print(combined.head())