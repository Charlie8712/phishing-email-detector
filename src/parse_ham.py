import os
import pandas as pd

HAM_DIR = "data/ham/"

emails = []

for filename in os.listdir(HAM_DIR):
    file_path = os.path.join(HAM_DIR, filename)

    try:
        with open(file_path, "r", encoding="latin-1") as f:
            content = f.read()

            # Extract subject if present
            subject = ""
            body = content

            if "Subject:" in content:
                parts = content.split("Subject:", 1)
                subject = parts[1].split("\n", 1)[0].strip()
                body = parts[1].split("\n", 1)[1].strip()

            emails.append({
                "subject": subject,
                "body": body,
                "label": 0   # 0 = ham (legit)
            })

    except:
        print(f"Error reading: {filename}")
        continue

df = pd.DataFrame(emails)
df.to_csv("data/ham_clean.csv", index=False)

print("Saved ham emails → data/ham_clean.csv")