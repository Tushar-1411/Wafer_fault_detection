from sklearn.base import accuracy_score
from src.exceptions import CustomException
from src.logger import logging
from utils.helper import save_object
import os, sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

@dataclass
class Model_training_config:
    artifacts = os.path.join("artifacts")
    trained_model_path = os.path.join(artifacts, "trained_model.pkl")


class Model_trainer:
    def __init__(self) -> None:
        model_config = Model_training_config()
        self.trained_model_path = model_config.trained_model_path
        self.root_folder = model_config.artifacts
        self.models = {"Logistic Regression" : LogisticRegression(),
                      "Decision tress" : DecisionTreeClassifier(),
                      "Extra Trees" : ExtraTreeClassifier(),
                      "Random Forest" : RandomForestClassifier(),
                      "Gradient Boost" : HistGradientBoostingClassifier()}

    def train_model(self, X_train, X_test, y_train, y_test):
        """Train a classifier based on the dataset and save it to disk"""
        try:
            scores = []
            
            for name, model in  self.models.items():
                # Fit the model using the training set
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                scores.append((name, score))

            return scores
        except Exception as e:
            raise CustomException(e, sys)
        
    def best_model(self, X_train, X_test, y_train, y_test):
        """Choose the most accurate model as the final one"""
        logging.info("Entering into the model training and evaluation phase ...")
        try:
            scores = self.train_model(X_train, X_test, y_train, y_test)
            max_score_model_name = scores.sort(key = lambda x: x[1], reverse= True)[0][0]
            best_model = self.models[max_score_model_name]    

            return max_score_model_name, best_model
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def save_best_model(self, best_model, trained_model_path):
        logging.info("Save the best model for prediction ...")
        try:
            

            save_object(trained_model_path, best_model)
            logging.info("Model Saved for prediction successfully ...")

            return trained_model_path

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_model_training(self, train_arr, test_arr):
        logging.info("Initiating Model Training Phase ...")
        try:
            X_train, y_train, X_test, y_test = (
                    train_arr[:, :-1],
                    train_arr[:, -1],
                    test_arr[:, :-1],
                    test_arr[:, -1],
                )
            best_model_name, best_model_obj = self.best_model(X_train, X_test, y_train, y_test)
            os.makedirs(os.path.dirname(self.trained_model_path), exist_ok=True)

            trained_model_path = self.save_best_model(best_model = best_model_obj, 
                                                    trained_model_path = self.trained_model_path)
            logging.info("Training phase completed successfully ...")
            return trained_model_path
        except Exception as e:
            raise CustomException(e, sys)





