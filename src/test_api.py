# ---------------------------------------------------------------
# File: test_api.py
# Description:
# Script to test the California Housing Prediction REST API.
# Sends a sample POST request with input features and prints the response.
# Part of: MLOps Pipeline - Model Inference Testing
# ---------------------------------------------------------------

import requests  # To send HTTP requests

# URL of the running Flask API
url = "http://localhost:9696/predict"

# Sample input features (must match training features)
input_data = {
    "MedInc": 8.3,
    "HouseAge": 41.0,
    "AveRooms": 6.1,
    "AveBedrms": 1.1,
    "Population": 980.0,
    "AveOccup": 2.5,
    "Latitude": 34.0,
    "Longitude": -118.0
}

# Send a POST request with JSON data
response = requests.post(url, json=input_data)

# Print the response from the API
print("Status Code:", response.status_code)
print("Response:", response.json())
