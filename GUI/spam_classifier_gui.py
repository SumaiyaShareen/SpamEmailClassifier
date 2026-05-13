import tkinter as tk
from tkinter import messagebox
import joblib
import re
import nltk
import os

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---------------------------
# CHECK MODELS
# ---------------------------
if not os.path.exists("model.pkl") or not os.path.exists("tfidf.pkl"):
    messagebox.showerror("Error", "model.pkl ya tfidf.pkl missing hai!\nPehle model train karo.")
    exit()

# ---------------------------
# LOAD MODEL
# ---------------------------
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
important_words = {'win','free','cash','prize','click','offer'}
stop_words = stop_words - important_words
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# ---------------------------
# PREDICTION
# ---------------------------
def predict(text):
    clean = preprocess(text)
    vec = tfidf.transform([clean])

    prob = model.predict_proba(vec)[0][1] * 100

    if prob > 50:
        result = "🚨 SPAM"
    else:
        result = "✅ HAM"

    risk = "HIGH ⚠️" if prob > 80 else "MEDIUM ⚠️" if prob > 50 else "LOW ✅"

    return result, prob, risk

# ---------------------------
# BUTTON ACTION
# ---------------------------
def check_email():
    text = input_box.get("1.0", tk.END).strip()

    if not text:
        messagebox.showwarning("Warning", "Please enter email text!")
        return

    result, prob, risk = predict(text)

    result_label.config(text=result)
    detail_label.config(text=f"Spam Probability: {prob:.2f}%")
    risk_label.config(text=f"Risk Level: {risk}")

# ---------------------------
# UI DESIGN
# ---------------------------
root = tk.Tk()
root.title("SpamGuard AI")
root.geometry("800x500")
root.configure(bg="#0f172a")

# TITLE
tk.Label(root,
         text="SpamGuard AI",
         font=("Arial", 26, "bold"),
         bg="#0f172a",
         fg="#38bdf8").pack(pady=10)

# INPUT BOX
input_box = tk.Text(root,
                    height=8,
                    width=85,
                    bg="#1e293b",
                    fg="white",
                    insertbackground="white")
input_box.pack(pady=15)

# BUTTON
tk.Button(root,
          text="Analyze Email",
          command=check_email,
          bg="#38bdf8",
          fg="black",
          font=("Arial", 12, "bold")).pack()

# RESULT
result_label = tk.Label(root,
                        text="Result: ---",
                        font=("Arial", 18, "bold"),
                        bg="#0f172a",
                        fg="white")
result_label.pack(pady=15)

detail_label = tk.Label(root,
                        text="Spam Probability: ---",
                        bg="#0f172a",
                        fg="gray")
detail_label.pack()

risk_label = tk.Label(root,
                      text="Risk Level: ---",
                      bg="#0f172a",
                      fg="orange")
risk_label.pack()

root.mainloop()