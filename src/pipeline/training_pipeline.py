import os, sys
from src.exceptions import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import Data_Transformation
from src.components.model_training import Model_trainer

class Trainig_Pipeline:

    def __init__(self):
        pass
    
    def start_data_ingestion(self):
        try:
            data = DataIngestion()
            file_path = data.initiate_data_ingestion()
            return file_path
        except Exception as e:
              raise CustomException(e, sys)

    def start_data_transformation(self, file_path):
        try:
            data_tranform = Data_Transformation(file_path)
            train_arr, test_arr, preprocessor = data_tranform.initiate_data_transformation()
            return (train_arr, test_arr, preprocessor)
        except Exception as e:
             raise CustomException(e, sys)
    
    def start_model_training(self, train_arr, test_arr):
        try:
            model_trainer = Model_trainer()
            trained_model_path = model_trainer.initiate_model_training(train_arr = train_arr, test_arr = test_arr)
            return trained_model_path
    
        except Exception as e:
             raise CustomException(e, sys)
        

    def run_training_pipeline(self):
        try:
            file_path = self.start_data_ingestion()
            train_arr, test_arr, preprocessor = self.start_data_transformation(file_path)
            trained_model_path = self.start_model_training(train_arr,test_arr)

        except Exception as e:
            raise CustomException(e, sys)