# -*- coding: utf-8 -*-

# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Model Loading ---

MODEL_PATH = 'artifacts/best_pipeline.joblib' 


@st.cache_resource
def load_model(path):
    """Loads the pre-trained model pipeline from disk."""
    try:
        pipeline = joblib.load(path)
        return pipeline
    except FileNotFoundError:
        st.error(f"Error: Model file '{path}' not found. Please ensure you have run `python train.py` successfully.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


model_pipeline = load_model(MODEL_PATH)



st.set_page_config(page_title="Telco Customer Churn Predictor", layout="centered")

st.title("📞 Telco Customer Churn Prediction App")
st.markdown("Enter the customer's profile and service details below to predict their likelihood of **Churn** (leaving the company).")
st.markdown("---")


with st.form(key='churn_form'):
    
    
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Profile")
        gender = st.selectbox('Gender', ['Male', 'Female'])
        senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
        partner = st.selectbox('Partner', ['No', 'Yes'])
        dependents = st.selectbox('Dependents', ['No', 'Yes'])
        
    with col2:
        st.subheader("⏱️ Tenure & Contract")
        tenure = st.slider('Tenure (Months)', 1, 72, 24)
        contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
        paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
        payment_method = st.selectbox('Payment Method', [
            'Electronic check', 'Mailed check', 
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ])

    with col3:
        st.subheader("💳 Charges")
        monthly_charges = st.number_input('Monthly Charges ($)', min_value=18.0, max_value=150.0, value=70.0, step=0.5)
        total_charges = st.number_input('Total Charges ($)', min_value=0.0, value=1500.0, step=5.0)
        
        # --- Phone Service Details ---
        st.subheader("📞 Phone Service")
        phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
        multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])

    st.markdown("---")
    st.subheader("🌐 Internet Services")
    col4, col5 = st.columns(2)
    
    with col4:
        internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
        online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
        online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
        device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
    
    with col5:
        tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
        streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
        streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
        
    st.markdown("---")
    submitted = st.form_submit_button("Predict Churn")



if submitted:
    
    input_data = {
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


    input_df = pd.DataFrame([input_data])

   
    try:

        probabilities = model_pipeline.predict_proba(input_df)
        churn_prob = probabilities[0][1]

        # 4. Display Results
        st.subheader("📊 Prediction Result")

        st.metric(label="Churn Probability", value=f"{churn_prob*100:.1f}%")

        if churn_prob > 0.5:
            st.error(f"🚨 **HIGH RISK**: This customer is likely to churn. The probability is {churn_prob*100:.1f}%.")
            st.balloons()
        else:
            st.success(f"✅ **LOW RISK**: This customer is likely to stay. The probability of churn is {churn_prob*100:.1f}%.")
            
        st.markdown(f"***Model's Raw Prediction:*** *The probability of class 'Yes' (Churn) is `{churn_prob:.4f}`.*")

    except Exception as e:
        st.error(f"An error occurred during prediction. Please check your model pipeline and input features: {e}")