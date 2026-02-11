import sys
import os

# Ensure the project root is in the python path so 'src' can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from src.data_ingest import DataIngest
from src.data_transform import Datatransform
from src.model_train import TrainModel

def run_pipeline():
    # 1. Ingest
    path = r"D:\ML projects\churn\data\WA_Fn-UseC_-Telco-Customer-Churn.csv"
    print("--- Starting Ingestion ---")
    ingestor = DataIngest(path)
    raw_data = ingestor.data_import()
    
    # 2. Transform
    print("--- Starting Transformation ---")
    transformer = Datatransform(raw_data)
    transformed_data = transformer.transform_data()
    
    # 3. Train
    print("--- Starting Training & MLflow Logging ---")
    trainer = TrainModel(transformed_data)
    trainer.train_model()
    print("--- Pipeline Execution Complete ---")

if __name__ == "__main__":
    run_pipeline()