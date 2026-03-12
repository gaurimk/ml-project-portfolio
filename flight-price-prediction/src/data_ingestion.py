import pandas as pd
import os

class DataIngestion:
    
    def __init__(self):
        self.train_path = "data/Data_Train.xlsx"
        self.artifacts_path = "artifacts/raw_data.csv"
    
    def initiate_data_ingestion(self):
        
        df = pd.read_excel(self.train_path)
        
        os.makedirs("artifacts", exist_ok=True)
        
        df.to_csv(self.artifacts_path, index=False)
        
        print("Data ingestion completed")
        
        return df