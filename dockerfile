# ---------------------------------------------------------------
# File: Dockerfile
# Description:
# Container definition to serve the California Housing model API
# using Flask and MLflow. This image exposes the REST endpoint
# on port 9696 and runs api_service.py inside a slim Python base.
# ---------------------------------------------------------------

# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only necessary files
COPY api_service.py ./
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose the API port
EXPOSE 9696

# Run the Flask app
CMD ["python", "api_service.py"]
