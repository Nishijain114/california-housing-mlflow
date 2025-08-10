import requests

def test_predict_api_pytest():
    url = "http://localhost:8000/predict/"
    input_data = {
        "longitude": -118.0,
        "latitude": 34.0,
        "housing_median_age": 41.0,
        "total_rooms": 6000,
        "total_bedrooms": 1200,
        "population": 1000,
        "households": 500,
        "median_income": 5.5,
        "ocean_proximity": "INLAND"
    }
    
    response = requests.post(url, json=input_data, timeout=10)
    
    # Assert HTTP response is successful
    assert response.status_code == 200
    
    json_response = response.json()
    
    # Assert response contains expected key, e.g., "prediction"
    assert "predictions" in json_response

    prediction = json_response["predictions"][0]
    assert isinstance(prediction, (int, float))
    assert prediction > 0
