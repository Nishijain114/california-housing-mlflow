from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import os
import sys
import sqlite3
import json
import time
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "src"))

from src.logger import get_logger
from src.api.schemas import HousingInput

logger = get_logger(__name__)

DB_PATH = os.path.join(ROOT_DIR, "prediction_logs.db")

app = FastAPI(title="California Housing Price Predictor")

@app.on_event("startup")
def startup():
    # Ensure directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    logger.info("Ensuring DB directory exists and creating/checking DB table at %s", DB_PATH)

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    request_data TEXT,
                    prediction TEXT,
                    status_code INTEGER,
                    process_time REAL
                )
            """)
            conn.commit()
        logger.info("SQLite table 'prediction_logs' is ready.")

    except Exception as e:
        logger.error("Error creating SQLite table: %s", e, exc_info=True)
        raise

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_model.joblib")
SCALER_PATH = os.path.join(ROOT_DIR, "models", "scaler.joblib")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("Model and scaler loaded successfully.")
except Exception as e:
    logger.error("Failed to load model or scaler: %s", e, exc_info=True)
    raise e

OCEAN_CATEGORIES = [
    "<1H OCEAN",
    "INLAND",
    "ISLAND",
    "NEAR BAY",
    "NEAR OCEAN"
]

FEATURE_COLUMNS = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income',
    'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND',
    'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY',
    'ocean_proximity_NEAR OCEAN'
]

def log_prediction(timestamp, request_data, prediction, status_code, process_time):
    try:
        logger.info("Logging prediction at %s", timestamp)
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO prediction_logs (timestamp, request_data, prediction, status_code, process_time)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    json.dumps(request_data),
                    json.dumps(prediction),
                    status_code,
                    process_time
                )
            )
            conn.commit()
        logger.info("Prediction logged successfully.")
    except (sqlite3.Error, json.JSONDecodeError) as e:
        logger.error("Failed to insert prediction log: %s", e, exc_info=True)

@app.get("/")
def read_root():
    return {"message": "California Housing Prediction API is up!"}

@app.post("/predict/")
def predict(input_data: HousingInput):
    start_time = time.time()
    try:
        logger.info("Received prediction request")
        df = pd.DataFrame([input_data.dict()])

        # One-hot encode ocean proximity
        ocean_dummies = pd.get_dummies(df['ocean_proximity'], prefix='ocean_proximity')
        for cat in OCEAN_CATEGORIES:
            col_name = f'ocean_proximity_{cat}'
            if col_name not in ocean_dummies.columns:
                ocean_dummies[col_name] = 0

        df = df.drop(columns=['ocean_proximity'])
        df = pd.concat([df, ocean_dummies], axis=1)
        df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

        input_scaled = scaler.transform(df)
        prediction = model.predict(input_scaled)
        prediction_list = prediction.tolist()

        process_time = (time.time() - start_time) * 1000  # ms
        timestamp = datetime.now().isoformat()

        log_prediction(timestamp, input_data.dict(), prediction_list, 200, process_time)

        logger.info("Prediction made for input: %s, Output: %s, Time: %.2fms",
            input_data.dict(), prediction_list, process_time)

        return {"predictions": prediction_list}

    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        timestamp = datetime.now().isoformat()

        logger.error("Prediction failed: %s", e, exc_info=True)
        log_prediction(timestamp, input_data.dict(), [], 500, process_time)

        raise HTTPException(status_code=500, detail=str(e)) from e

# Prometheus instrumentation
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)
