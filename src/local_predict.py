# ---------------------------------------------------------------
# File: predict.py
# Problem Statement:
# - Load the best registered model from MLflow
# - Use it to make predictions on test data (California Housing)
# - Print the predicted value(s)
# Part of: MLOps Pipeline - Local Prediction Test
# ---------------------------------------------------------------

import mlflow
import mlflow.sklearn
import pandas as pd

def load_test_data(n_rows=1):
    """
    Load test data from processed file. By default, load only one row.
    """
    X_test = pd.read_csv("data/processed/X_test.csv")
    return X_test.iloc[:n_rows]

def predict_with_registered_model():
    """
    Load the registered model from MLflow and use it to predict
    house values from the test data.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # use only if using remote server

    model_name = "CaliforniaHousingModel"
    stage = "None"  # Use "Production" or "Staging" if model is in a stage

    print(f" Loading model '{model_name}' from MLflow Registry...")
    model_uri = f"models:/{model_name}/{stage}" if stage != "None" else f"models:/{model_name}/1"

    model = mlflow.sklearn.load_model(model_uri)

    X_sample = load_test_data()
    prediction = model.predict(X_sample)

    print(f"\n Predicted House Value: {prediction[0]:.2f}")

if __name__ == "__main__":
    predict_with_registered_model()
