import streamlit as st
import requests

st.sidebar.title("Navigation")
mode = st.sidebar.radio("Go to", ["Analysis Mode", "Prediction Mode"])

API_URL = "http://127.0.0.1:8000"

if mode == "Analysis Mode":
    st.title("📈 Exploratory Data Analysis")
    # Fetch image list from API
    response = requests.get(f"{API_URL}/get_analysis_images")
    if response.status_code == 200:
        images = response.json()["images"]
        for img_name in images:
            st.subheader(f"Plot: {img_name}")
            st.image(f"{API_URL}/image/{img_name}")

elif mode == "Prediction Mode":
    st.title("🔮 Churn Prediction")
    
    with st.form("user_input"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure", 0, 100, 1)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        
        with col2:
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            monthly = st.number_input("Monthly Charges", value=50.0)
            total = st.number_input("Total Charges", value=100.0)
            # Add other fields as per your RawCustomerData schema...

        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": "Yes", # Default or add input
            "MultipleLines": "No", "InternetService": internet, "OnlineSecurity": security,
            "OnlineBackup": backup, "DeviceProtection": "No", "TechSupport": tech,
            "StreamingTV": "No", "StreamingMovies": "No", "Contract": contract,
            "PaperlessBilling": "Yes", "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total
        }
        
        res = requests.post(f"{API_URL}/predict", json=payload)
        if res.status_code == 200:
            result = res.json()
            if result["churn"] == 1:
                st.error(f"Customer likely to Churn (Prob: {result['probability']:.2f})")
            else:
                st.success(f"Customer likely to Stay (Prob: {result['probability']:.2f})")