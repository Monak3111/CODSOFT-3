import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💳 Credit Card Fraud Detection App")

st.write("Enter transaction details:")

# Input fields
time = st.number_input("Time", value=0.0)

v_features = []
for i in range(1, 29):
    v = st.number_input(f"V{i}", value=0.0)
    v_features.append(v)

amount = st.number_input("Amount", value=0.0)

# Scale amount
amount_scaled = scaler.transform([[amount]])[0][0]

# Prediction button
if st.button("Predict Fraud"):
    features = np.array([time] + v_features + [amount_scaled]).reshape(1, -1)

    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f"🚨 Fraud Transaction Detected (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Legit Transaction (Probability of fraud: {prob:.2f})")