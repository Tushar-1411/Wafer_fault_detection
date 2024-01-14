from flask import Flask, render_template, jsonify, request, send_file
from src.exceptions import CustomException
from src.logger import logging as lg
import os,sys

from src.pipeline.training_pipeline import Training_Pipeline
from src.pipeline.prediction_pipeline import Predict

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/train")
def train_route():
    try:
        train_pipeline = Training_Pipeline()
        train_pipeline.run_training_pipeline()

        return "Training Completed."

    except Exception as e:
        raise CustomException(e,sys)

@app.route('/predict', methods=['POST', 'GET'])
def upload():
    
    try:


        if request.method == 'POST':
            prediction_pipeline = Predict(request)
            prediction_file_detail = prediction_pipeline.run_prediction_pipeline()

            lg.info("prediction completed. Downloading prediction file.")
            return send_file(prediction_file_detail.prediction_file_path,
                            download_name= prediction_file_detail.prediction_file_name,
                            as_attachment= True)


        else:
            return render_template('index.html')
    except Exception as e:
        raise CustomException(e,sys)
    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug= True)
