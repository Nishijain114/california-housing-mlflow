import pandas as pd
import pytest
from src.data.preprocess import split_and_preprocess

@pytest.fixture
def sample_csv_path(tmp_path):
    """Creates a small sample CSV file for testing preprocessing."""
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

def test_split_and_preprocess(sample_csv_path, tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Run preprocessing
    split_and_preprocess(
        input_path=str(sample_csv_path),
        output_path=str(output_dir),
        test_size=0.5,
        random_state=42,
        target_column="median_house_value"
    )

    processed_dir = output_dir / "processed"

    # Assert that all processed files exist
    expected_files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    for filename in expected_files:
        file_path = processed_dir / filename
        assert file_path.exists(), f"{filename} does not exist in processed directory"

    # Load processed features to verify one-hot encoding worked correctly
    X_train_df = pd.read_csv(processed_dir / "X_train.csv")
    ocean_prox_cols = [col for col in X_train_df.columns if col.startswith("ocean_proximity_")]

    # Assert one-hot encoded columns for ocean_proximity exist
    assert len(ocean_prox_cols) > 0, "No one-hot encoded ocean_proximity columns found"

    # Assert known categories are present in columns
    known_categories = {"ocean_proximity_INLAND", "ocean_proximity_NEAR BAY"}
    assert known_categories.intersection(set(X_train_df.columns)), \
        "Expected ocean_proximity one-hot columns not found in features"
