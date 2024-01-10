import os
import sys
import pandas as pd
import numpy as np
from src.exceptions import CustomException
from src.logger import logging
from dataclasses import dataclass


@dataclass
class Data_Ingestion_config:
    artifacts_folder: str = os.path.join("artifacts")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = Data_Ingestion_config()
        self.file_path = os.path.join(self.data_ingestion_config.artifacts_folder, "wafer_fault.csv")

    def read_dataframe(self) -> pd.DataFrame:    
        try:
            df = pd.read_csv(self.file_path)

            if "_id" in df.columns.to_list():
                    df = df.drop(columns=["_id"], axis=1)
            
            df.replace({"na": np.nan}, inplace=True)

            return df
        except Exception as e:
             raise CustomException(e, sys)
    
    def store_dataframe(self, df):
        try:
            df.to_csv(self.file_path, index = False)

            return self.file_path

        except Exception as e:
            raise CustomException(e, sys)
    
        
    def initiate_data_ingestion(self):
         try:

            logging.info("Initiating Data Ingestion Step ...")
            os.makedirs(self.data_ingestion_config.artifacts_folder, exist_ok=True)

            logging.info("Reading the dataset ...")
            df = self.read_dataframe()

            logging.info("Storing the dataframe with some initial changes ...")
            path = self.store_dataframe(df)

            logging.info("Data Ingestion step completed without any errors !!")

            return str(path)

         except Exception as e:
            raise CustomException(e, sys)
    