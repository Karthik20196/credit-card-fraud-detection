import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and column names
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="Fraud Detector", layout="centered")

st.title("💳 Smart Credit Card Fraud Detector")
st.markdown("Upload your transaction data as a CSV and get instant fraud predictions!")

uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.subheader("📋 Raw Uploaded Data")
    st.write(data.head())

    if set(columns).issubset(data.columns):
        # Preprocess: scale 'Time' and 'Amount'
        data[['Time', 'Amount']] = scaler.transform(data[['Time', 'Amount']])

        # Predict
        predictions = model.predict(data[columns])
        data['Fraud Prediction'] = predictions

        st.subheader("🔍 Prediction Results")
        st.write(data[['Time', 'Amount', 'Fraud Prediction']].head(10))

        st.success(f"✅ Found {sum(predictions)} fraudulent transactions.")

        # Download button
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", csv, "fraud_predictions.csv", "text/csv")

    else:
        st.error("Uploaded file does not have required columns!")
