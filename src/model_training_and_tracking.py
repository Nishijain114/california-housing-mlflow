# src/train.py

import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from mlflow.models.signature import infer_signature
from logger import get_logger

logger = get_logger(__name__)

def load_data_with_scaler(scaler_path="models/scaler.joblib"):
    logger.info("Loading raw data and applying saved scaler...")

    scaler = joblib.load(scaler_path)

    X_train_raw = pd.read_csv("data/raw/X_train_raw.csv")
    X_test_raw = pd.read_csv("data/raw/X_test_raw.csv")
    y_train = pd.read_csv("data/raw/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/raw/y_test.csv").values.ravel()

    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, X_test, y_train, y_test

def train_and_log_model(model, model_name, X_train, y_train, X_test, y_test):
    logger.info(f"Training {model_name}")
    with mlflow.start_run(nested=True):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)

        mlflow.log_param("model", model_name)
        mlflow.log_params(model.get_params())
        mlflow.log_metric("mse", mse)

        signature = infer_signature(X_test, preds)
        input_example = X_test[:2]

        mlflow.sklearn.log_model(
            sk_model=model,
            name=f"{model_name}_artifact",
            signature=signature,
            input_example=input_example
        )

        return model, mse

def main():
    logger.info("Starting model training with raw data + saved scaler")
    mlflow.set_experiment("CaliforniaHousing")

    X_train, X_test, y_train, y_test = load_data_with_scaler()

    best_model, best_mse, best_model_name = None, float("inf"), ""

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor()
    }

    with mlflow.start_run(run_name="CompareModels"):
        for name, model in models.items():
            trained_model, mse = train_and_log_model(model, name, X_train, y_train, X_test, y_test)
            if mse < best_mse:
                best_mse = mse
                best_model = trained_model
                best_model_name = name

        logger.info(f"Best model: {best_model_name} (MSE: {best_mse:.4f})")

        # Register and save best model
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="best_model",
            registered_model_name="CaliforniaHousingModel"
        )

        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, "models/best_model.joblib")

if __name__ == "__main__":
    main()
