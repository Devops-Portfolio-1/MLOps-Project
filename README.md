# Diabetes Prediction Model (FastAPI + MLflow + Docker + K8s)

This project predicts diabetes risk from patient health indicators using a notebook-driven ML pipeline and a FastAPI inference service. It includes EDA, feature engineering, model comparison, MLflow tracking, and deployment artifacts.

## Dataset

- Source file: [diabetes_prediction_dataset.csv](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

## Current Project Status

- Notebook pipeline: diabetes_ml_pipeline.ipynb (EDA, preprocessing, feature engineering, CV, model selection)
- Best model bundle: best_diabetes_model.pkl (model + scaler + selected features)
- FastAPI app uses the same preprocessing steps as the notebook
- MLflow model registry: diabetes_best_model
- UI available at / (index.html).

## Quick Start

### 1. Clone the Repo

```bash
git clone https://github.com/Devops-Portfolio-1/MLOps-Project.git
cd mlops-project
```

### 2. Create Virtual Environment

```bash
python3 -m venv .mlops
source .mlops/bin/activate
```

```powershell
python -m venv .mlops
.\.mlops\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Notebook Pipeline

Open diabetes_ml_pipeline.ipynb and run all cells. This will:

- Train and evaluate models
- Track experiments in MLflow
- Register the best model as diabetes_best_model
- Export best_diabetes_model.pkl

### 5. Run the API Locally

```bash
uvicorn main:app --reload
```

The UI is available at http://127.0.0.1:8000

### Sample Input for /predict

```json
{
  "gender": "Female",
  "age": 45,
  "hypertension": 1,
  "heart_disease": 0,
  "bmi": 28.5,
  "HbA1c_level": 6.5,
  "blood_glucose_level": 140,
  "smoking_history": "former"
}
```

Response example:

```json
{
  "diabetic": false,
  "probability": 0.0639
}
```

## MLflow UI

```bash
mlflow ui --backend-store-uri mlruns
```

## Dockerize the API

```bash
docker build -t diabetes-prediction-model .
docker run -p 8000:8000 diabetes-prediction-model
```

## Deploy to Kubernetes

```bash
kubectl apply -f k8s-deploy.yml
```

## Credits

Created by Shalindra Perera. ✨

