import pandas as pd
def data_import(file_path):
    df = pd.read_csv(file_path)
    return df
    