## Objectives

- Load, preprocess, and version a public dataset  
- Train and track multiple models using **MLflow**  
- Serve the best model as a **REST API** using Flask  
- **Containerize** with Docker  
- Set up **CI/CD** with GitHub Actions  
- Add **logging** and optional **monitoring with Prometheus**

---

## Tech Stack (TOCHECK)

| Phase               | Tools / Technologies           |
|--------------------|--------------------------------|
| Version Control     | Git + GitHub                  |
| Data Versioning     | DVC (optional)                |
| Model Training      | Scikit-learn                  |
| Model Tracking      | MLflow                        |
| API Service         | Flask                         |
| Containerization    | Docker                        |
| CI/CD Pipeline      | GitHub Actions                |
| Logging             | Python `logging` module       |
| Monitoring (Opt.)   | Prometheus, Grafana           |

---

## Project Structure (Expected)

```
mlops-pipeline/
├── data/                  # Processed datasets
├── models/                # Saved model & scaler
├── src/                   # Core source code
│   ├── preprocess.py      # Data loading & processing
│   ├── model_training_and_tracking.py           # Model training & tracking
│   ├── local_predict.py   # Local prediction script
│   ├── app.py             # Flask API server
│   ├── utils.py           # Utility functions
├── logs/                  # Prediction logs
├── requirements.txt       # Dependencies
├── Dockerfile             # Docker container definition
├── .github/workflows/     # CI/CD pipeline (GitHub Actions)
└── README.md              # Project documentation
```

---

### 1) Set Up the Environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

### 2) Preprocess the Dataset

```bash
python src/preprocess.py
```

**Output files:**

- `data/processed/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`  
- `models/scaler.joblib`

---

### 3) Launch MLflow UI

```bash
mlflow ui
```

> Open in browser: `http://localhost:5000`

---

### 4) Train Models

```bash
python src/train.py
```

**What happens:**

- Trains **Linear Regression** and **Decision Tree** models  
- Logs parameters, metrics, and artifacts to **MLflow**  
- Registers best model as: `CaliforniaHousingModel`

---

### 5) Test Local Prediction

```bash
python src/predict.py
```

**Expected Output:**

```
Predicted House Value: <value>
```
---

6) Test the API (Flask + MLflow Model)
Option 1: Run API Locally

First, run the API service:
python src/api_service.py

It should show:
Loading model from: models:/CaliforniaHousingModel/1
 * Running on http://0.0.0.0:9696

Make a Prediction
Run the provided test script to send a sample POST request:
python src/test_api.py


Expected Output:
Status Code: 200
Response: {'prediction': 3.45}


Test using a POST request:
curl -X POST http://localhost:9696/predict \
     -H "Content-Type: application/json" \
     -d '{"MedInc": 8.3, "HouseAge": 41.0, "AveRooms": 6.1, "AveBedrms": 1.1, "Population": 980.0, "AveOccup": 2.5, "Latitude": 34.0, "Longitude": -118.0}'

7) Dockerize the API
Build Docker Image
docker build -t housing-api .

Run Docker Container
docker run -p 9696:9696 housing-api

API Endpoint
http://localhost:9696/predict

Sample Input
{
  "MedInc": 8.3,
  "HouseAge": 41.0,
  "AveRooms": 6.1,
  "AveBedrms": 1.1,
  "Population": 980.0,
  "AveOccup": 2.5,
  "Latitude": 34.0,
  "Longitude": -118.0
}

Sample Output
{"prediction": 3.45}



# Nishi's project structure 
# ------------------------
# Directory Structure:
# ------------------------
# ml_project/
# ├── data/                   # Raw and processed data
# │   ├── raw/
# │   └── processed/
# ├── models/                # Saved ML models
# ├── notebooks/             # EDA and prototyping
# ├── src/                   # Source code
# │   ├── train.py          # Model training logic
# │   ├── predict.py        # API input/output logic
# │   ├── logger.py         # Logging util
# │   └── utils.py          # Reusable functions
# ├── app/
# │   ├── main.py           # FastAPI/Flask app
# │   └── schemas.py        # Pydantic models
# ├── Dockerfile
# ├── requirements.txt
# ├── .github/workflows/
# │   └── main.yml         # GitHub Actions CI/CD
# └── README.md
