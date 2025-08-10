import os

def test_model_and_scaler_exist():
    model_path = os.path.join("models", "best_model.joblib")
    scaler_path = os.path.join("models", "scaler.joblib")

    assert os.path.exists(model_path), f"Missing: {model_path}"
    assert os.path.exists(scaler_path), f"Missing: {scaler_path}"
