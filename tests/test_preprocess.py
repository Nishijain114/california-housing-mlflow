import os
import pandas as pd
import pytest
from data.preprocess import split_and_preprocess

@pytest.fixture
def sample_csv(tmp_path):
    # Create a small sample CSV for testing
    data = {
        "longitude": [-118.0, -117.0],
        "latitude": [34.0, 35.0],
        "housing_median_age": [41, 42],
        "total_rooms": [6000, 7000],
        "total_bedrooms": [1200, 1300],
        "population": [1000, 1100],
        "households": [500, 600],
        "median_income": [5.5, 6.0],
        "ocean_proximity": ["INLAND", "NEAR BAY"],
        "median_house_value": [200000, 250000]
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "sample_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

def test_split_and_preprocess(sample_csv, tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Run preprocessing
    split_and_preprocess(
        input_path=str(sample_csv),
        output_path=str(output_dir),
        test_size=0.5,
        random_state=42,
        target_column="median_house_value"
    )

    # Check if processed files exist
    processed_dir = output_dir / "processed"
    assert (processed_dir / "X_train.csv").exists()
    assert (processed_dir / "X_test.csv").exists()
    assert (processed_dir / "y_train.csv").exists()
    assert (processed_dir / "y_test.csv").exists()

    # Optionally check content of one file
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    assert "ocean_proximity_INLAND" in X_train.columns or "ocean_proximity_NEAR BAY" in X_train.columns
