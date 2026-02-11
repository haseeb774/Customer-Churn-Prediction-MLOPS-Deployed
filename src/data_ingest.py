import pandas as pd
import sys
from src.logging import logging
from src.exception import CustomException
class DataIngest:
    def __init__(self, file_path):
        self.file_path = file_path

    def data_import(self):
        try:
            df = pd.read_csv(self.file_path)
            logging.info("file is load and returned from data_ingest")
            return df
            
            
        except Exception as e:
            logging.info("error occured in data_ingest")
            raise CustomException(e,sys)
            