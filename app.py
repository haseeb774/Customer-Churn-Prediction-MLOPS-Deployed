import streamlit as st
import pandas as pd
import joblib

model = joblib.load("churn_prediction_model.pkl")
feature_order = joblib.load("feature_order.pkl")

st.title("📊 Customer Churn Prediction App")


gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("SeniorCitizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total_charges = tenure * monthly_charges  # auto calculate

internet = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("DeviceProtection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"])

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("PaperlessBilling", ["Yes", "No"])
payment = st.selectbox("PaymentMethod", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
phone_service = st.selectbox("PhoneService", ["Yes", "No"])


input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "InternetService": [internet],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment],
    "PhoneService": [phone_service]
})


input_data = input_data.reindex(columns=feature_order)

if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1 or prob<=20:
        st.error(f"⚠️ Customer is likely to CHURN (probability {prob:.2%})")
    else:
        st.success(f"✅ Customer is likely to STAY (probability {prob:.2%})")
