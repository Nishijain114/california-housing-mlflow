FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies first (caching optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/ ./src
COPY models/ ./models

# Expose the API port
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "src.api.predict_api:app", "--host", "0.0.0.0", "--port", "8000"]
