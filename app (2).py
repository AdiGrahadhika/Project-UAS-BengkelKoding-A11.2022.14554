import streamlit as st
import pandas as pd
import joblib

# Konfigurasi Halaman
st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

# 1. Fungsi Pemuatan Model (Pastikan nama file di GitHub sama persis!)
@st.cache_resource
def load_model():
    # Pastikan file 'best_model_churn.pkl' sudah di-upload ke GitHub dalam satu folder dengan app.py
    return joblib.load('best_model_churn.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error: Model tidak ditemukan. Pastikan 'best_model_churn.pkl' ada di GitHub. Detail: {e}")
    st.stop()

# --- HEADER ---
st.title("📊 Telco Customer Churn Prediction App - A11.2022.14554")
st.markdown("Aplikasi ini memprediksi kemungkinan pelanggan berhenti berlangganan (Churn).")
st.divider()

# --- SIDEBAR (INPUT) ---
st.sidebar.header("📝 Input Fitur Pelanggan")

def user_input():
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    senior = st.sidebar.selectbox("Senior Citizen (1=Ya, 0=Tidak)", [0, 1])
    tenure = st.sidebar.slider("Tenure (Bulan)", 0, 72, 12)
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    monthly = st.sidebar.number_input("Monthly Charges ($)", value=70.0)
    total = st.sidebar.number_input("Total Charges ($)", value=800.0)
    
    data = {
        'gender': gender,
        'SeniorCitizen': senior,
        'tenure': tenure,
        'Contract': contract,
        'InternetService': internet,
        'MonthlyCharges': monthly,
        'TotalCharges': total
    }
    return pd.DataFrame([data])

input_df = user_input()

# --- MAIN PANEL (HASIL) ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Detail Data Input")
    st.write(input_df)

with col2:
    st.subheader("Hasil Prediksi")
    if st.button("Jalankan Prediksi"):
        # Lakukan prediksi
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1]
        
        if prediction[0] == 1:
            st.error(f"Hasil: **CHURN** (Peluang: {probability:.2%})")
        else:
            st.success(f"Hasil: **TIDAK CHURN** (Peluang: {1-probability:.2%})")

st.divider()
st.info("Catatan: Pastikan urutan kolom input sesuai dengan saat Anda melatih model.")