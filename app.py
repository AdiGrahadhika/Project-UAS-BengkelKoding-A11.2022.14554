import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Konfigurasi Halaman
st.set_page_config(page_title="Telco Churn App", layout="wide")

# 1. Pemuatan Model [cite: 196]
@st.cache_resource
def load_model():
    return joblib.load('best_model_churn.pkl')

model = load_model()

# Header Aplikasi
st.title("📊 Telco Customer Churn Prediction")
st.markdown("""
Aplikasi ini menggunakan model Machine Learning untuk memprediksi apakah pelanggan akan berhenti berlangganan (Churn). 
Silakan masukkan data pelanggan pada sidebar untuk melihat hasil prediksi.
""")

st.divider()

# 2. Form atau Input Fitur (Sidebar) [cite: 197]
st.sidebar.header("📝 Input Data Pelanggan")
def user_input_features():
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    senior = st.sidebar.selectbox("Senior Citizen", (0, 1))
    tenure = st.sidebar.slider("Tenure (Bulan)", 0, 72, 12)
    contract = st.sidebar.selectbox("Contract", ("Month-to-month", "One year", "Two year"))
    internet = st.sidebar.selectbox("Internet Service", ("DSL", "Fiber optic", "No"))
    monthly = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, value=65.0)
    total = st.sidebar.number_input("Total Charges ($)", min_value=0.0, value=500.0)
    payment = st.sidebar.selectbox("Payment Method", 
                                  ("Electronic check", "Mailed check", "Bank transfer", "Credit card"))
    
    data = {
        'gender': gender, 'SeniorCitizen': senior, 'tenure': tenure,
        'Contract': contract, 'InternetService': internet,
        'MonthlyCharges': monthly, 'TotalCharges': total, 'PaymentMethod': payment
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Tampilan Utama
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Detail Input Pelanggan")
    st.write(input_df)

# 3 & 4. Proses dan Tampilan Hasil Prediksi [cite: 198, 199]
with col2:
    st.subheader("Hasil Prediksi")
    if st.button("Prediksi Churn"):
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        if prediction[0] == 1:
            st.error("🚨 Pelanggan diprediksi: **CHURN**")
        else:
            st.success("✅ Pelanggan diprediksi: **TIDAK CHURN**")
            
        st.write(f"Probabilitas Churn: **{prediction_proba[0][1]:.2%}**")

# 5. Elemen Pendukung (Visualisasi/Penjelasan) [cite: 200]
st.divider()
st.subheader("ℹ️ Penjelasan Fitur Utama")
expander = st.expander("Klik untuk melihat penjelasan fitur")
expander.write("""
- **Tenure**: Menunjukkan berapa lama pelanggan telah menggunakan layanan dalam hitungan bulan[cite: 122].
- **Monthly Charges**: Biaya yang dibebankan kepada pelanggan setiap bulan[cite: 128].
- **Total Charges**: Akumulasi biaya selama pelanggan berlangganan[cite: 128].
- **Contract**: Jenis kontrak yang dimiliki (bulanan, tahunan, atau dua tahunan)[cite: 128].
""")

# Contoh visualisasi sederhana sebagai elemen pendukung
st.subheader("📈 Analisis Singkat")
fig, ax = plt.subplots()
ax.bar(["Monthly", "Total (Scaled/100)"], [input_df['MonthlyCharges'][0], input_df['TotalCharges'][0]/100])
st.pyplot(fig)