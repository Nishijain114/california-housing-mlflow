# ---------------------------------------------------------------
# File: api_service.py
# Description:
# Flask API to serve California Housing prediction using MLflow model.
# Applies same scaler used during training.
# ---------------------------------------------------------------

from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load the scaler used during training
scaler = joblib.load("models/scaler.joblib")

# Define feature columns in training order
FEATURE_COLUMNS = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population",
    "AveOccup", "Latitude", "Longitude"
]

# Load the best model from MLflow Registry
MODEL_NAME = "CaliforniaHousingModel"
MODEL_VERSION = "1"
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

print(f"Loading model from: {MODEL_URI}")
model = mlflow.sklearn.load_model(MODEL_URI)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "California Housing Price Prediction API"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_json = request.get_json()
        input_df = pd.DataFrame([input_json])

        # Reorder columns to match training set
        input_df = input_df[FEATURE_COLUMNS]

        # Apply scaler (as done in preprocessing.py)
        input_scaled = scaler.transform(input_df)

        # Predict using model
        prediction = model.predict(input_scaled)[0]

        return jsonify({"prediction": round(float(prediction), 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696)
