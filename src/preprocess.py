# ---------------------------------------------------------------
# File: preprocess.py
# Description: Loads California Housing dataset, applies preprocessing
#              (scaling), splits into train/test sets, and saves to disk.
# Part of: MLOps Pipeline - Data Preparation Stage
# ---------------------------------------------------------------

import os
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_data():
    # Load the California Housing dataset from sklearn
    dataset = fetch_california_housing()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="target")
    return X, y

def preprocess_data(X):
    # Apply standard scaling (zero mean, unit variance)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def split_and_save(X, y, test_size=0.2, random_state=42, output_dir="data/processed"):
    # Create output dir if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Save splits to CSV
    pd.DataFrame(X_train).to_csv(f"{output_dir}/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    return X_train, X_test, y_train, y_test

def save_scaler(scaler, path="models/scaler.joblib"):
    # Save fitted scaler for later use
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)

def main():
    """
    Main function to orchestrate the data loading, preprocessing,
    splitting, and saving steps.
    """
    print("Starting data preprocessing pipeline...")

    X, y = load_data()                        # load raw data
    X_scaled, scaler = preprocess_data(X)     # apply scaling
    X_train, X_test, y_train, y_test = split_and_save(X_scaled, y)  # split and save
    save_scaler(scaler)                       # store scaler for inference

    print("Preprocessing complete. Files saved to data/processed.")
    print("\nSample Preprocessed Training Data (first 3 rows):")
    print(pd.DataFrame(X_train).head(3))

if __name__ == "__main__":
    main()

# --- DVC Tracking Instructions (Run these commands from your terminal AFTER this script runs) ---
# To track the processed data:
# dvc add data/processed/X_train.csv data/processed/X_test.csv data/processed/y_train.csv data/processed/y_test.csv

# To track the saved scaler:
# dvc add models/scaler.joblib

# After adding with DVC, commit the changes to Git:
# git add data/processed/.gitignore data/processed/*.dvc models/.gitignore models/*.dvc preprocess.py
# git commit -m "feat: Add data preprocessing stage and DVC tracking for processed data and scaler"
