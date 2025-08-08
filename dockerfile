# ---------------------------------------------------------------
# File: Dockerfile
# Description:
# Container definition to serve the California Housing model API
# using Flask and MLflow. This image exposes the REST endpoint
# on port 9696 and runs api_service.py inside a slim Python base.
# ---------------------------------------------------------------

FROM python:3.9

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
