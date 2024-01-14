import os, sys
import pandas as pd
from flask import request
from src.logger import logging
from src.exceptions import CustomException
from utils.helper import Helperfuncs
from dataclasses import dataclass

@dataclass
class Predicition_Config:
    model_path = os.path.join("artifacts", "trained_model.pkl" )
    preprocessor_path = os.path.join("artifacts", "preprocessor.pkl" )
    prediction_dir = "predictions"
    prediction_file_name = "predictions.csv"
    prediction_file_path = os.path.join(prediction_dir, prediction_file_name)


class Predict:
    def __init__(self, request : request):
        self.request = request
        self.config = Predicition_Config()
        self.utils = Helperfuncs()
        self.model = self.utils.load_object(self.config.model_path)
        self.preprocessor = self.utils.load_object(self.config.preprocessor_path)


    def save_input_files(self)-> str:
        try:
            pred_file_input_dir = "prediction_artifacts"
            os.makedirs(pred_file_input_dir, exist_ok=True)

            input_csv_file = self.request.files['file']
            pred_file_path = os.path.join(pred_file_input_dir, input_csv_file.filename)
                       
            input_csv_file.save(pred_file_path)

            return pred_file_path
        
        except Exception as e:
            raise CustomException(e,sys)


    def predict(self, input_features):
        try:
            preprocessed_input = self.preprocessor.transform(input_features)
            predictions = self.model.predict(preprocessed_input)

            return predictions

        except Exception as e:
            raise CustomException(e, sys)
        
    def return_predicted_df(self, input_dataframe):
        try:

            prediction_column_name : str = "target"
            input_dataframe: pd.DataFrame = pd.read_csv(input_dataframe)
            
            input_dataframe =  input_dataframe.drop(columns="Unnamed: 0") if "Unnamed: 0" in input_dataframe.columns else input_dataframe

            predictions = self.predict(input_dataframe)
            input_dataframe[prediction_column_name] = [pred for pred in predictions]
            target_column_mapping = {0:'bad', 1:'good'}

            input_dataframe[prediction_column_name] = input_dataframe[prediction_column_name].map(target_column_mapping)
            
            os.makedirs( self.config.prediction_dir, exist_ok= True)
            input_dataframe.to_csv(self.config.prediction_file_path, index= False)
            logging.info("predictions completed. ")


        except Exception as e:
            raise CustomException(e, sys)
        
    def run_prediction_pipeline(self):
        try:
            input_csv_path = self.save_input_files()
            self.return_predicted_df(input_csv_path)

            return self.config

        except Exception as e:
            raise CustomException(e, sys)