
# ---------------------------------------------------------------
# File: train.py
# Problem Statement:
# - Train at least two models (Linear Regression, Decision Tree for California Housing).
# - Use MLflow to track experiments: log parameters, metrics, models.
# - Select the best model based on MSE and register it in MLflow.
# Part of: MLOps Pipeline - Model Development and Experiment Tracking
# ---------------------------------------------------------------

import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from mlflow.models.signature import infer_signature

def load_data():
    """
    Load preprocessed train and test data from CSV files.
    """
    print("Loading training and test datasets from disk...")
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    print(" Data loaded successfully.\n")
    return X_train, X_test, y_train, y_test

def train_and_log_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    Trains the given model, evaluates it using MSE,
    logs the model and metrics to MLflow.
    """
    print(f" Training model: {model_name}")
    with mlflow.start_run(nested=True):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)

        # Logging to MLflow
        mlflow.log_param("model", model_name)
        mlflow.log_metric("mse", mse)
        signature = infer_signature(X_test, preds)
        input_example = X_test.iloc[:2]  # small input sample for example

        mlflow.sklearn.log_model(
        sk_model=model,
        name=f"{model_name}_artifact",
        signature=signature,
        input_example=input_example
        )


        print(f" {model_name} evaluation complete. MSE: {mse:.4f}\n")
        return model, mse

def main():
    """
    Loads data, trains models, logs results to MLflow,
    prints a summary table, and registers the best model.
    """
    # Step 1: Load data
    X_train, X_test, y_train, y_test = load_data()

    # Step 2: Configure MLflow experiment (local filesystem backend)
    print("Setting up MLflow experiment: 'CaliforniaHousing'")
    mlflow.set_experiment("CaliforniaHousing")

    best_model = None
    best_mse = float("inf")
    best_model_name = ""
    model_scores = {}  # For storing mse values

    with mlflow.start_run(run_name="CompareModels"):
        print("Starting training for all models...\n")

        models = {
            "LinearRegression": LinearRegression(),
            "DecisionTree": DecisionTreeRegressor()
        }

        for name, model in models.items():
            trained_model, mse = train_and_log_model(model, X_train, y_train, X_test, y_test, name)
            model_scores[name] = mse
            if mse < best_mse:
                best_mse = mse
                best_model = trained_model
                best_model_name = name

        # Print comparison table
        print("\nModel Comparison (Lower MSE is better):")
        print("-" * 40)
        print(f"{'Model':<20} | {'MSE':>10}")
        print("-" * 40)
        for name, mse in model_scores.items():
            print(f"{name:<20} | {mse:>10.4f}")
        print("-" * 40)

        # Register the best model
        print(f"\nBest model selected: {best_model_name} (MSE: {best_mse:.4f})")
        print("Registering the best model in MLflow...\n")
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="best_model",
            registered_model_name="CaliforniaHousingModel"
        )

        print(" Model training and registration complete.")

        # Show instructions for accessing MLflow UI
        tracking_uri = mlflow.get_tracking_uri()
        print("\n--- MLflow Tracking Info ---")
        print(f"Tracking URI: {tracking_uri}")

        if "localhost" in tracking_uri or "127.0.0.1" in tracking_uri or tracking_uri.startswith("file:") or tracking_uri.startswith("./mlruns"):
            print("To view the MLflow UI locally, run the following in your terminal:")
            print("  mlflow ui")
            print("Then open your browser and go to: http://localhost:5000")
        else:
            print("To view experiment details, open the tracking URL in your browser:")
            print(f"  {tracking_uri}")


if __name__ == "__main__":
    main()
