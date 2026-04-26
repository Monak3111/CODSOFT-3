# 💳 Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using classification algorithms and a Streamlit web app for real-time predictions.

---

## 📌 Project Overview

This project uses transaction data (PCA-transformed features V1–V28, Time, Amount) to predict whether a transaction is **fraudulent or legitimate**.

* `0` → Legit transaction
* `1` → Fraud transaction

---

## 🚀 Features

* Data preprocessing and scaling
* Handling imbalanced dataset using SMOTE
* Machine Learning model training
* Real-time prediction using Streamlit UI
* Probability-based fraud detection

---

## 🧠 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Streamlit
* Joblib

---

## 📂 Project Structure

```
fraud_detection/
│
├── app.py               # Streamlit web app
├── creditcard.csv       # Dataset
├── fraud_model.pkl      # Trained model
├── scaler.pkl           # Scaler for Amount
└── README.md
```

---

## ⚙️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train model (optional if already trained)

Run your training script to generate `.pkl` files.

### 3. Start Streamlit app

```bash
streamlit run app.py
```

---

## 📊 Model Workflow

1. Load dataset
2. Scale `Amount` feature
3. Balance dataset using SMOTE
4. Train classification model
5. Save model and scaler
6. Deploy using Streamlit

---

## 🎯 Output

The app predicts:

* ✅ Legit Transaction
* 🚨 Fraud Transaction (with probability score)

---

## 📌 Future Improvements

* Use XGBoost or Deep Learning models
* Add API using FastAPI
* Deploy on cloud (AWS / Render / HuggingFace)

---

## 👨‍💻 Author

Built for learning and showcasing machine learning deployment skills.

---
