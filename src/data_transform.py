import pandas as pd
import sys
from src.logging import logging
from src.exception import CustomException
class Datatransform:
    def __init__(self, data):
        self.data = data

    
    def transform_data(self):
        try:
            df = self.data
            df["gender"] = df["gender"].map({"Female": 0,"Male": 1})

            
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            df.dropna(inplace=True)


            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})


            if "customerID" in df.columns:
                df.drop(columns="customerID", inplace=True)
            if "CLV_proxy" in df.columns:
                df.drop(columns="CLV_proxy", inplace=True)

            
            df["Partner"] = df["Partner"].map({"Yes": 1, "No": 0})
            df["Dependents"] = df["Dependents"].map({"Yes": 1, "No": 0})
            df["PhoneService"] = df["PhoneService"].map({"Yes": 1, "No": 0})
            df["MultipleLines"] = df["MultipleLines"].replace({"Yes": 1, "No": 0, "No phone service": 0})
            df["InternetService"] = df["InternetService"].replace({"DSL": 1, "Fiber optic": 2, "No": 0})
            df["Contract"] = df["Contract"].replace({"Month-to-month": 0, "One year": 1, "Two year": 2})
            df["PaperlessBilling"] = df["PaperlessBilling"].map({"Yes": 1, "No": 0})
            services = ["OnlineBackup","OnlineSecurity","StreamingMovies",
                        "StreamingTV","DeviceProtection","TechSupport"]

            for col in services:
                df[col] = df[col].replace({"Yes": 1, "No": 0, "No internet service": 0})

            #  one hot encode for df["PaymentMethod"] through scikit learn
            from sklearn.preprocessing import OneHotEncoder
            import numpy as np

            encoder = OneHotEncoder(sparse_output=False)

            encoded_data = encoder.fit_transform(df[["PaymentMethod"]])
            df = df.drop(columns=["PaymentMethod"])
            df = pd.concat([df, pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(["PaymentMethod"]))], axis=1)
            df = df.dropna()
            logging.info("data_transformation succeesfuly completw")
            df.to_csv("data/processed/processed_churn.csv")
            return df
            

            
        except Exception as e:
            logging.info("error occured in datatransform")
            raise CustomException(e,sys)
            