
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 2. Load the pre-trained model, scaler, and feature list
# Ensure these files are available in the current working directory
model = joblib.load('best_model_churn.pkl')
scaler = joblib.load('scaler.pkl')
model_features = joblib.load('model_features.pkl')

# 3. Define numerical and categorical features based on the original dataset and preprocessing steps
# These lists are based on the original dataframe structure before one-hot encoding
numerical_cols_to_scale_for_prediction = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
categorical_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod'
]

# 4. Set up the Streamlit application title and description
st.title("Telco Customer Churn Prediction")
st.write("Enter customer details to predict if they will churn.")

# 5. Create a user input form for all 19 features
with st.form("churn_predictor_form"):
    st.subheader("Customer Information")

    # Numerical Inputs
    senior_citizen = st.selectbox("Senior Citizen?", [0, 1], help="0: No, 1: Yes")
    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges ($")", min_value=0.0, max_value=150.0, value=50.0, step=0.01)
    total_charges = st.number_input("Total Charges ($")", min_value=0.0, max_value=10000.0, value=500.0, step=0.01)

    # Categorical Inputs
    gender = st.selectbox("Gender", ['Female', 'Male'])
    partner = st.selectbox("Partner", ['Yes', 'No'])
    dependents = st.selectbox("Dependents", ['Yes', 'No'])
    phone_service = st.selectbox("Phone Service", ['Yes', 'No'])
    multiple_lines = st.selectbox("Multiple Lines", ['No phone service', 'No', 'Yes'])
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox("Online Security", ['No', 'Yes', 'No internet service'])
    online_backup = st.selectbox("Online Backup", ['No', 'Yes', 'No internet service'])
    device_protection = st.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])
    tech_support = st.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
    streaming_tv = st.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'])
    streaming_movies = st.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'])
    contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
    payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

    submitted = st.form_submit_button("Predict Churn")

# 6. Implement the prediction logic
if submitted:
    # Create a dictionary from user inputs
    user_input = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    # Create a DataFrame from the user input
    input_df = pd.DataFrame([user_input])

    # Preprocessing the input for prediction
    # Initialize a new DataFrame with all expected columns from model_features, filled with zeros
    processed_input = pd.DataFrame(0, index=[0], columns=model_features)

    # Directly assign numerical features
    for col in numerical_cols_to_scale_for_prediction:
        if col in input_df.columns:
            processed_input[col] = input_df[col]

    # One-Hot Encode categorical features, handling drop_first=True
    for col in categorical_cols:
        if col in input_df.columns:
            value = input_df[col].iloc[0]
            # For binary columns where 'No' is the dropped category (e.g., Partner_Yes)
            if value == 'Yes' and f"{col}_Yes" in model_features:
                processed_input[f"{col}_Yes"] = 1
            elif value == 'Male' and f"{col}_Male" in model_features:
                processed_input[f"{col}_Male"] = 1
            # For other categorical columns that result in multiple OHE columns
            elif value != 'No' and f"{col}_{value}" in model_features:
                processed_input[f"{col}_{value}"] = 1
            elif value == 'No phone service' and f"MultipleLines_No phone service" in model_features:
                processed_input[f"MultipleLines_No phone service"] = 1
            elif value == 'No internet service':
                # Handle cases like OnlineSecurity_No internet service
                if f"{col}_No internet service" in model_features:
                    processed_input[f"{col}_No internet service"] = 1


    # Apply scaling to the numerical columns using the loaded scaler
    processed_input[numerical_cols_to_scale_for_prediction] = scaler.transform(
        processed_input[numerical_cols_to_scale_for_prediction]
    )

    # Ensure the order of columns is the same as model_features
    processed_input = processed_input[model_features]

    # Make prediction and get probability
    prediction = model.predict(processed_input)
    probability = model.predict_proba(processed_input)[0]

    churn_probability = probability[1] # Probability of churn (class 1)

    # Display the prediction result
    st.write("### Prediction Result:")
    if prediction[0] == 1:
        st.error(f"**Customer is likely to CHURN!** (Probability: {churn_probability:.2%})")
    else:
        st.success(f"**Customer is likely to STAY!** (Probability: {1 - churn_probability:.2%})")
