import os
import pandas as pd

PHISH_DIR = "data/phishing/"
all_rows = []

for file in os.listdir(PHISH_DIR):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(PHISH_DIR, file), encoding="latin-1")

        # Standardize missing columns
        subject = df.columns[df.columns.str.contains("subject", case=False, regex=True)]
        body = df.columns[df.columns.str.contains("body|text|content|email", case=False, regex=True)]

        if len(body) == 0:
            print(f"No body column found in {file}, skipping.")
            continue

        df_clean = pd.DataFrame()
        df_clean["subject"] = df[subject[0]] if len(subject) > 0 else ""
        df_clean["body"] = df[body[0]]
        df_clean["label"] = 1  # phishing

        all_rows.append(df_clean)

phishing_df = pd.concat(all_rows, ignore_index=True)
phishing_df.to_csv("data/phishing_clean.csv", index=False)

print("Saved phishing emails → data/phishing_clean.csv")