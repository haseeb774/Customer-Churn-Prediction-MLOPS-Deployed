from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI()

# Load trained model
model = joblib.load("outputs/model.pkl")

# Path to analysis images
ANALYSIS_PATH = "outputs/analysis"

class RawCustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# Mapping Dictionary for Encoding
MAPPING = {
    "gender": {"Female": 0.0, "Male": 1.0},
    "Partner": {"No": 0.0, "Yes": 1.0},
    "Dependents": {"No": 0.0, "Yes": 1.0},
    "PhoneService": {"No": 0.0, "Yes": 1.0},
    "MultipleLines": {"No": 0.0, "Yes": 1.0, "No phone service": 0.0},
    "InternetService": {"No": 0.0, "DSL": 1.0, "Fiber optic": 2.0},
    "OnlineSecurity": {"No": 0.0, "Yes": 1.0, "No internet service": 0.0},
    "OnlineBackup": {"No": 0.0, "Yes": 1.0, "No internet service": 0.0},
    "DeviceProtection": {"No": 0.0, "Yes": 1.0, "No internet service": 0.0},
    "TechSupport": {"No": 0.0, "Yes": 1.0, "No internet service": 0.0},
    "StreamingTV": {"No": 0.0, "Yes": 1.0, "No internet service": 0.0},
    "StreamingMovies": {"No": 0.0, "Yes": 1.0, "No internet service": 0.0},
    "Contract": {"Month-to-month": 0.0, "One year": 1.0, "Two year": 2.0},
    "PaperlessBilling": {"No": 0.0, "Yes": 1.0}
}

@app.get("/get_analysis_images")
def get_images():
    # Return a list of .jpg files in the folder
    images = [f for f in os.listdir(ANALYSIS_PATH) if f.endswith(".jpg")]
    return {"images": images}

@app.get("/image/{image_name}")
def serve_image(image_name: str):
    return FileResponse(os.path.join(ANALYSIS_PATH, image_name))

@app.post("/predict")
def predict(data: RawCustomerData):
    d = data.dict()
    
    # 1. Transform categorical data using the Mapping
    processed_data = {}
    for key, value in d.items():
        if key in MAPPING:
            processed_data[key] = MAPPING[key].get(value, 0.0)
        else:
            processed_data[key] = value

    # 2. Handle One-Hot Encoding for PaymentMethod manually
    pm = d['PaymentMethod']
    processed_data['PaymentMethod_Bank transfer (automatic)'] = 1.0 if pm == "Bank transfer (automatic)" else 0.0
    processed_data['PaymentMethod_Credit card (automatic)'] = 1.0 if pm == "Credit card (automatic)" else 0.0
    processed_data['PaymentMethod_Electronic check'] = 1.0 if pm == "Electronic check" else 0.0
    processed_data['PaymentMethod_Mailed check'] = 1.0 if pm == "Mailed check" else 0.0
    
    # Remove the original PaymentMethod key
    del processed_data['PaymentMethod']

    # 3. Create DataFrame and Predict
    df = pd.DataFrame([processed_data])
    prediction = model.predict(df)
    prob = model.predict_proba(df)[:, 1]

    return {"churn": int(prediction[0]), "probability": float(prob[0])}