from src.data_ingest import data_import
from src.data_transform import transform_data
from src.model_train import train_model

if __name__ == "__main__":
    # Use 'r' for the Windows path to avoid SyntaxWarnings
    path = r"D:\ML projects\churn\data\WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    # 1. Ingest
    data = data_import(path)
    
    # 2. Transform
    transformer = transform_data(data)
    
    # 3. Train
    train_model(transformer)