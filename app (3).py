import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model_churn.pkl")

st.set_page_config(
    page_title="Telco Customer Churn Prediction - A11.2022.14554",
    layout="centered"
)

st.title("📊 Telco Customer Churn Prediction")
st.write("""
Aplikasi ini digunakan untuk memprediksi apakah seorang pelanggan
berpotensi **churn (berhenti berlangganan)** berdasarkan data layanan dan tagihan.
""")

st.sidebar.header("Input Data Pelanggan")

def user_input():
    tenure = st.sidebar.number_input("Tenure (bulan)", 0, 72, 12)
    monthly = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1500.0)

    data = {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }
    return pd.DataFrame([data])

input_df = user_input()

st.subheader("📌 Data Input")
st.write(input_df)

if st.button("🔍 Predict Churn"):
    prediction = model.predict(input_df)
    prob = model.predict_proba(input_df)

    if prediction[0] == 1:
        st.error("❌ Pelanggan BERISIKO CHURN")
    else:
        st.success("✅ Pelanggan TIDAK CHURN")

    st.write("### Probabilitas Prediksi")
    st.write(f"Churn: {prob[0][1]*100:.2f}%")
    st.write(f"No Churn: {prob[0][0]*100:.2f}%")

st.markdown("---")
st.caption("UAS Bengkel Koding Data Science | Awaludin Gymnastiar (A11.2022.14414)")
