from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import joblib
import numpy as np
from src.logger import get_logger
from app.schemas import HousingInput
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.middleware.cors import CORSMiddleware

logger = get_logger(__name__)

app = FastAPI(title="California Housing Price Predictor")

# Enable CORS (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
MODEL_PATH = "models/best_model.joblib"
SCALER_PATH = "models/scaler.joblib"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("Model and scaler loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model or scaler: {e}")
    raise e

N_FEATURES = 8

@app.get("/")
def read_root():
    return {"message": "California Housing Prediction API is up!"}

@app.post("/predict/")
def predict(input_data: HousingInput):
    try:
        input_array = np.array(input_data.data)

        if input_array.ndim == 1:
            input_array = input_array.reshape(1, -1)

        if input_array.shape[1] != N_FEATURES:
            raise HTTPException(status_code=400, detail=f"Each input must have {N_FEATURES} features")

        input_scaled = scaler.transform(input_array)
        predictions = model.predict(input_scaled)

        logger.info(f"Prediction made for input: {input_data.data}")
        return {"predictions": predictions.tolist()}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ✅ Register Prometheus instrumentation here (before startup)
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)
