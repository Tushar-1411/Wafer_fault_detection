import os
import sys
from sklearn.model_selection import train_test_split
from utils.helper import save_object
from sklearn.pipeline import Pipeline
from src.exceptions import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from dataclasses import dataclass

@dataclass
class Data_Transformation_config:
    artifact_dir=os.path.join("artifacts")
    transformed_train_file_path = os.path.join(artifact_dir, 'train.npy')
    transformed_test_file_path = os.path.join(artifact_dir, 'test.npy') 
    transformed_object_file_path = os.path.join( artifact_dir, 'preprocessor.pkl' )

class Data_Transformation:

    def __init__(self, data_path):
        self.data_transform_config = Data_Transformation_config()
        self.data_path = data_path

    def get_preprocessor_object(self):
        try:
            # define the steps for the preprocessor pipeline
            imputer_step = ('imputer', SimpleImputer(strategy='median'))
            scaler_step = ('scaler', StandardScaler())

            preprocessor = Pipeline(
                steps=[
                imputer_step,
                scaler_step
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
        
    def get_data(data_path) -> pd.DataFrame:
        try:
            data = pd.read_csv(data_path)
            data.rename(columns={"Good/Bad": "target"}, inplace=True)


            return data
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self):
        
        logging.info("Entered initiate_data_transformation method of Data_Transformation class")

        try:
            dataframe = self.get_data(data_path=self.data_path)
            
            X = dataframe.drop(columns= "target")
            y = np.where(dataframe["target"]==-1,0, 1)  #replacing the -1 with 0 for model training
            
            
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

            preprocessor = self.get_data_transformer_object()

            X_train_scaled =  preprocessor.fit_transform(X_train)
            X_test_scaled  =  preprocessor.transform(X_test)


            preprocessor_path = self.data_transformation_config.transformed_object_file_path
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok= True)
            save_object( file_path= preprocessor_path,
                        obj= preprocessor)

            train_arr = np.c_[X_train_scaled, np.array(y_train) ]
            test_arr = np.c_[ X_test_scaled, np.array(y_test) ]

            return (train_arr, test_arr, preprocessor_path)
        

        except Exception as e:
            raise CustomException(e, sys) from e