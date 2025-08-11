# California Housing MLFlow Project

## Project Checklist

| Task | Done | Key Files / Location |
|------|------|----------------------|
| **Part 1 – Repository & Data Versioning** |||
| Load & preprocess dataset | ✅ | `src/data/preprocess.py`, `data/raw/` |
| Track dataset with DVC | ✅ | `dvc.yaml`, `dvc.lock`, `data/raw/housing.csv.dvc` |
| Maintain clean directory structure | ✅ | Project root, folders |
| **Part 2 – Model Dev & Experiment Tracking** |||
| Train ≥ 2 models | ✅ | `src/model/train_and_track.py`, `models/best_model.joblib` |
| Track experiments with MLflow | ✅ | `src/model/train_and_track.py` |
| Register best model in MLflow | ✅ | `models/best_model.joblib` |
| **Part 3 – API & Docker** |||
| Create prediction API (Flask/FastAPI) | ✅ | `src/api/predict_api.py`, `schemas.py` |
| Containerize service with Docker | ✅ | `Dockerfile`, `docker-compose.yml` |
| Accept JSON input & return prediction | ✅ | `src/api/predict_api.py` (hosted at `http://localhost:8000/docs`) |
| **Part 4 – CI/CD (GitHub Actions)** |||
| Lint & test on push | ✅ | `.github/workflows/ci.yml`, `tests/` folder |
| Build Docker image & push to Hub | ✅ | `.github/workflows/ci.yml` |
| Deploy locally / EC2 | ✅ | `Dockerfile`, `docker-compose.yml` |
| **Part 5 – Logging & Monitoring** |||
| Log requests & predictions | ✅ | `src/logger.py` |
| Store logs (file/SQLite) | ✅ | Hosted at `http://localhost:8000/logs` |
| Expose /metrics endpoint | ✅ | `src/api/predict_api.py`, `monitoring/prometheus/`, hosted at `http://localhost:8000/metrics` |
| **Bonus** |||
| Input validation (Pydantic/schema) | ✅ | `src/api/schemas.py` |
| Prometheus + Grafana | ✅ | `monitoring/prometheus.yml`, `grafana/dashboard.json` |
| Model re-training on new data | ✅ | `src/model/train_and_track.py`, `dvc.yaml`, `models/best_model.joblib` Triggered when new dataset is added via: ```bash dvc add data/raw/housing.csv && dvc repro ``` tracks and pushes updated model.|

---

## Project Structure
```plaintext
california-housing-mlflow/
├── .github/workflows/ci.yml
├── data/raw/
│   ├── housing.csv.dvc
│   ├── X_test_raw.csv
│   ├── X_train_raw.csv
│   ├── y_test.csv
│   └── y_train.csv
├── models/
│   ├── best_model.joblib
│   └── scaler.joblib
├── monitoring/
│   ├── grafana/california_housing_dashboard.json
│   └── prometheus/prometheus.yml
├── src/
│   ├── logger.py
│   ├── api/
│   │   ├── predict_api.py
│   │   └── schemas.py
│   ├── data/preprocess.py
│   └── model/train_and_track.py
├── tests/
│   ├── test_logs.py
│   ├── test_model_file_exists.py
│   ├── test_predict_api.py
│   ├── test_prediction_pytest.py
│   └── test_preprocess_data.py
├── docker-compose.yml
├── Dockerfile
├── dvc.lock
├── dvc.yaml
├── params.yaml
├── requirements.txt
└── README.md
```

---

## Architecture Overview

This project predicts California housing prices using a modular, reproducible machine learning pipeline. It leverages modern MLOps tools for data versioning, experiment tracking, containerization, CI/CD, logging, and monitoring.

### 1. Data Management & Versioning
- `data/raw/` contains original datasets, tracked with DVC.
- Pipeline stages and parameters are defined in `dvc.yaml`, `dvc.lock`, `params.yaml`.

### 2. Model Development & Experiment Tracking
- `src/data/preprocess.py` handles cleaning, feature engineering, and dataset splitting.
- `src/model/train_and_track.py` trains models, logs experiments with MLflow, and stores best models in `models/`.

### 3. API Service & Containerization
- FastAPI app in `src/api/predict_api.py` with Pydantic validation (`schemas.py`).
- Dockerized via `Dockerfile` and `docker-compose.yml`.

### 4. CI/CD & Automation
- `.github/workflows/ci.yml` runs linting, testing, Docker builds, and pushes to Docker Hub.

### 5. Logging & Monitoring
- `src/logger.py` logs requests and predictions.
- Prometheus (`monitoring/prometheus/`) and Grafana (`monitoring/grafana/`) monitor API metrics.

### 6. Testing & QA
- `tests/` ensures data, model, and API correctness.

-	test_logs.py: Tests logging functionality to ensure prediction requests and outputs are correctly recorded.
-	test_model_file_exists.py: Verifies that the trained model file exists after training.
-	test_predict_api.py: Tests the API endpoints for correct responses and error handling.
-	test_prediction_pytest.py: Validates model predictions using pytest.
-	test_preprocess_data.py: Tests the data preprocessing script for correct output and data splits.


### 7. Model Retraining
- Triggered by new data via:
```bash
dvc add data/raw/housing.csv && dvc repro
```
- Retrains only affected pipeline stages.

1.	Init – git init, dvc init to set up Git & DVC.
2.	Download Data – Place raw file in data/raw/housing.csv.
3.	Track Data – dvc add data/raw/housing.csv → creates .dvc pointer.
4.	Commit Pointer – git add *.dvc .gitignore && git commit.
5.	Set Remote – dvc remote add -d localstorage <path> (once).
6.	Push Data – dvc push uploads to remote.
7.	Stage 1 – Split data (make_dataset.py) → train/test sets.
8.	Stage 2 – Train model (train_model.py) → model + metrics.
9.	Re-run – dvc repro runs only changed stages.
10.	Collaboration – git push, dvc push, dvc pull to share/reproduce.


---



## Important Commands

### 1️⃣ Initialize Git & DVC (One-time)
```bash
git init
dvc init
```

---

### 2️⃣ Get Raw Dataset
```bash
mkdir -p data/raw
curl -L -o data/raw/housing.csv \
  https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv
```

---

### 3️⃣ Track Dataset with DVC
```bash
dvc add data/raw/housing.csv
git add data/raw/housing.csv.dvc .gitignore
git commit -m "Track raw housing dataset with DVC"
```

---

### 4️⃣ Setup DVC Remote Storage (One-time)
Example: Local remote folder
```bash
dvc remote add -d localstorage "C:\Users\dell\Documents\BITS_STUDY_MATERIAL\Semester - 3\Mlops\assignment\dvcstore"
git commit .dvc/config -m "Add DVC remote storage"
```

Upload dataset to remote:
```bash
dvc push
```

---

### 5️⃣ Create DVC Pipeline

**Stage 1 – Data Split**
```bash
dvc stage add -n split \
  -d src/data/make_dataset.py -d data/raw/housing.csv \
  -p preprocess.test_size,preprocess.random_state \
  -o data/processed \
  python src/data/make_dataset.py \
    --in data/raw/housing.csv \
    --out data/processed \
    --test_size ${preprocess.test_size} \
    --random_state ${preprocess.random_state}
```

**Stage 2 – Train Model**
```bash
dvc stage add -n train \
  -d src/models/train_model.py -d data/processed/train.csv \
  -p train.n_estimators,train.max_depth,train.random_state \
  -o models/model.joblib \
  -M metrics/metrics.json \
  python src/models/train_model.py \
    --train data/processed/train.csv \
    --model_out models/model.joblib \
    --metrics_out metrics/metrics.json \
    --n_estimators ${train.n_estimators} \
    --max_depth ${train.max_depth} \
    --random_state ${train.random_state}
```

---

### 6️⃣ Run Pipeline
```bash
dvc repro
```

---

### 7️⃣ Experiment Tracking (Optional)
```bash
dvc exp run
dvc exp show
```

---

### 8️⃣ Pull/Push Data for Collaboration
```bash
dvc push   # Upload data/models to remote
dvc pull   # Download data/models from remote
```

---

## 🚀 Running the API

Run FastAPI app locally:
```bash
uvicorn src.api.predict_api:app --reload --host 0.0.0.0 --port 8000
```

---

## 📊 Prometheus Monitoring

Run Prometheus (Windows example):
```bash
C:\Users\dell\Downloads\prometheus-3.5.0.windows-amd64\prometheus.exe --config.file=prometheus.yml
```

---

## 🐳 Docker Deployment

Pull image:
```bash
docker pull nishijain411/california_housing:latest
```

Run container:
```bash
docker run -p 8000:8000 nishijain411/california_housing:latest
```

---


## Conclusion

The California Housing MLFlow project demonstrates an end-to-end MLOps workflow with reproducibility, scalability, and maintainability. It is modular, API-driven, monitored, and CI/CD-enabled, making it production-ready and extensible.
