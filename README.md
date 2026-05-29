# Phishing Email Detection System

## Overview

This project implements a machine learning–based phishing email detection system designed to classify emails as either **phishing** or **legitimate** using a combination of **natural language processing (NLP)** and **structural email analysis**.

The system is built to detect modern phishing attempts by analyzing both the **textual content** of emails and their **structural characteristics**, improving detection robustness compared to approaches that rely on a single feature type.

---

## Key Capabilities

- Hybrid feature extraction (text + structure)
- TF-IDF-based natural language processing
- Detection of suspicious structural patterns (e.g., URL frequency, email length features)
- Dataset balancing using oversampling techniques
- Supervised classification using Logistic Regression
- Full evaluation pipeline with performance metrics and visualization

---

## Methodology

The detection pipeline follows a structured machine learning workflow:

1. **Data Ingestion & Parsing**
   - Emails are collected from multiple public phishing datasets and legitimate email sources.
   - Data is standardized into a unified format (subject, body, label).

2. **Preprocessing**
   - Text normalization (lowercasing, noise removal)
   - Feature merging of subject and body content

3. **Feature Engineering**
   - TF-IDF vectorization for textual representation
   - Structural feature extraction:
     - URL count per email
     - Subject length
     - Body length

4. **Class Imbalance Handling**
   - Random oversampling applied to balance dataset distribution

5. **Model Training**
   - Logistic Regression classifier trained on combined feature set
   - Stratified train-test split ensures balanced evaluation

6. **Evaluation**
   - Accuracy, precision, recall, and F1-score
   - Confusion matrix analysis
   - Visual performance comparisons

---

## Results

The final model demonstrates strong classification performance:

- **Accuracy:** ~99.9%
- High precision and recall across both phishing and legitimate classes
- Low false positive and false negative rates
- Strong generalization after dataset balancing

The results indicate that combining textual and structural signals significantly improves phishing detection effectiveness compared to single-feature approaches.

---

## Visual Outputs

The project includes the following evaluation visualizations:

- Class distribution before balancing
- Class distribution after oversampling
- Confusion matrix analysis

These visualizations provide insight into dataset imbalance handling and model performance behavior.

---

## Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Logistic Regression
- Imbalanced-learn

---

## Project Structure
src/ # Data preprocessing, feature extraction, training scripts
data/ # Cleaned email datasets
graphs/ # Evaluation visualizations
models/ # Trained model artifacts


---

## Key Takeaways

- Hybrid feature engineering improves phishing detection reliability
- Structural email patterns provide strong signals when combined with NLP features
- Class imbalance handling is critical for realistic performance evaluation
- Lightweight models such as Logistic Regression can still achieve strong results with proper feature design

---

## Future Improvements

- Real-time email scanning integration
- Deep learning-based classification models
- Additional metadata features (sender domain analysis, header inspection)
- Deployment as a browser extension or API service
