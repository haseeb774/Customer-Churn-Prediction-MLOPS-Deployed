import pandas as pd
def transform_data(data):
    df = data
    df["gender"] = df["gender"].map({"Female": 0,"Male": 1})

    
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)


    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})


    if "customerID" in df.columns:
        df.drop(columns="customerID", inplace=True)
    if "CLV_proxy" in df.columns:
        df.drop(columns="CLV_proxy", inplace=True)

    drop_features = ["MultipleLines", "StreamingMovies"]
    df = df.drop(columns=[c for c in drop_features if c in df.columns])
    return df

    
