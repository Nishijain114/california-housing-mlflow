import os
import pandas as pd
import joblib
from src.data.preprocess import split_and_preprocess

def test_split_and_preprocess(tmp_path):
    # Create a small sample dataframe
    df = pd.DataFrame({
        "longitude": [-118.0, -119.0],
        "latitude": [34.0, 35.0],
        "housing_median_age": [41, 30],
        "total_rooms": [6000, 7000],
        "total_bedrooms": [1200, 1300],
        "population": [1000, 1100],
        "households": [500, 550],
        "median_income": [5.5, 6.0],
        "ocean_proximity": ["INLAND", "NEAR BAY"],
        "median_house_value": [300000, 350000]  # target
    })

    # Save dataframe to a temp CSV
    input_csv = tmp_path / "input.csv"
    df.to_csv(input_csv, index=False)

    # Define output directory inside tmp_path
    output_dir = tmp_path / "output"

    # Call your preprocessing function
    split_and_preprocess(
        input_path=str(input_csv),
        output_path=str(output_dir),
        test_size=0.5,
        random_state=42,
        target_column="median_house_value"
    )

    # Check if output directories are created
    assert (output_dir / "raw").exists()
    assert (output_dir / "processed").exists()
    assert os.path.exists("models/scaler.joblib")  # scaler saved in models/

    # Check raw files
    raw_files = ["X_train_raw.csv", "X_test_raw.csv", "y_train_raw.csv", "y_test_raw.csv"]
    for f in raw_files:
        assert (output_dir / "raw" / f).exists()

    # Check processed files
    processed_files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    for f in processed_files:
        assert (output_dir / "processed" / f).exists()

    # Load one processed file and check for one-hot columns
    X_train = pd.read_csv(output_dir / "processed" / "X_train.csv")
    assert any(col.startswith("ocean_proximity_") for col in X_train.columns)

    # Load scaler and check it's a StandardScaler object
    scaler = joblib.load("models/scaler.joblib")
    from sklearn.preprocessing import StandardScaler
    assert isinstance(scaler, StandardScaler)
