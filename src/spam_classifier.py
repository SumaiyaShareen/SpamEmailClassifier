# ---------------------------
# SpamGuard AI - Explainable Spam Classifier (UPGRADED)
# ---------------------------

import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

nltk.download('stopwords')

# ---------------------------
# Load Dataset
# ---------------------------
dataset_path = 'spam.csv'
if not os.path.isfile(dataset_path):
    dataset_path = os.path.join('Dataset', 'spam.csv')

df = pd.read_csv(dataset_path, encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ---------------------------
# Preprocessing
# ---------------------------
stop_words = set(stopwords.words('english'))
important_words = {'win','free','cash','prize','click','offer'}
stop_words = stop_words - important_words

stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(preprocess)

# ---------------------------
# Feature Extraction
# ---------------------------
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
X = tfidf.fit_transform(df['clean_text'])
y = df['label']

# ---------------------------
# Train Model
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# ---------------------------
# Evaluation
# ---------------------------
y_pred = model.predict(X_test)

print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------
# 🔥 UNIQUE FEATURE 1: Explainable AI
# ---------------------------
spam_keywords = ['free', 'win', 'cash', 'prize', 'click', 'offer', 'urgent', 'limited']

def explain_email(text):
    found = []
    text_lower = text.lower()
    for word in spam_keywords:
        if word in text_lower:
            found.append(word)
    return found

# ---------------------------
# 🔥 UNIQUE FEATURE 2: Prediction with Score + Explanation
# ---------------------------
def predict_email(text):
    clean = preprocess(text)
    vec = tfidf.transform([clean])

    prob = model.predict_proba(vec)[0]
    spam_prob = prob[1] * 100

    prediction = "Spam" if spam_prob > 50 else "Ham"

    # Risk level
    if spam_prob > 80:
        risk = "HIGH ⚠️"
    elif spam_prob > 50:
        risk = "MEDIUM ⚠️"
    else:
        risk = "LOW ✅"

    # Explainability
    reasons = explain_email(text)

    print("\n---------------------------")
    print("📩 Prediction:", prediction)
    print(f"📊 Spam Probability: {spam_prob:.2f}%")
    print("🚨 Risk Level:", risk)

    if reasons:
        print(" Why Spam Detected:")
        for r in reasons:
            print(" - contains keyword:", r)
    else:
        print(" No strong spam keywords found")

    print("---------------------------\n")



    # ---------------------------
# Save Model for UI (IMPORTANT FIX)
# ---------------------------
import joblib

joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "tfidf.pkl")

print("Model and TF-IDF saved successfully!")