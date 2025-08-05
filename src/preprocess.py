# src/preprocess.py

import os
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from logger import get_logger

logger = get_logger(__name__)

def load_data():
    dataset = fetch_california_housing()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="target")
    return X, y

def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def save_data(X_train_raw, X_test_raw, y_train, y_test, X_train_scaled, X_test_scaled):
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # Save raw (unscaled) data
    X_train_raw.to_csv("data/raw/X_train_raw.csv", index=False)
    X_test_raw.to_csv("data/raw/X_test_raw.csv", index=False)
    y_train.to_csv("data/raw/y_train.csv", index=False)
    y_test.to_csv("data/raw/y_test.csv", index=False)

    # Save processed (scaled) data
    pd.DataFrame(X_train_scaled, columns=X_train_raw.columns).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=X_test_raw.columns).to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    logger.info("Raw and processed data saved.")

def save_scaler(scaler, path="models/scaler.joblib"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    logger.info("Scaler saved to disk.")

def main():
    logger.info("Starting preprocessing...")

    X, y = load_data()
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled, scaler = preprocess_data(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    save_data(X_train_raw, X_test_raw, y_train, y_test, X_train_scaled, X_test_scaled)
    save_scaler(scaler)

    logger.info("Preprocessing complete.")

if __name__ == "__main__":
    main()
