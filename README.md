# 🩺 Diabetes Prediction Model (FastAPI + Docker + K8s)

This project helps me to learn **Building and Deploying an ML Model** using a simple and real-world use case: predicting whether a person is diabetic based on health metrics. We’ll go from:

- ✅ Model Training
- ✅ Building the Model locally
- ✅ API Deployment with FastAPI
- ✅ Applying MLOps best practices
- ✅ Dockerization
- ✅ Kubernetes Deployment

---

## 📊 Problem Statement

Diabetes is a chronic condition that affects millions worldwide. Early prediction can lead to better management and prevention. This project uses a dataset of health indicators to train a model that predicts diabetes risk, making it easier for healthcare providers to identify at-risk patients.

---

## 🚀 Quick Start

### 1. Clone the Repo

```bash
git clone https://github.com/Devops-Portfolio-1/MLOps-Project.git
cd mlops-project
```

### 2. Create Virtual Environment

```
python3 -m venv .mlops
source .mlops/bin/activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

## Train the Model

```
python train.py
```

## Run the API Locally

```
uvicorn main:app --reload
```

### Sample Input for /predict 

Option 1 : If using Postman with POST request, use this JSON body:
```
{
  "Pregnancies": 2,
  "Glucose": 130,
  "BloodPressure": 70,
  "BMI": 28.5,
  "Age": 45
}
```
option 2 : Enter the above values in the UI itself 

## Dockerize the API

### Build the Docker Image

```
docker build -t diabetes-prediction-model .
```

### Run the Container

```
docker run -p 8000:8000 diabetes-prediction-model
```

## Deploy to Kubernetes

```
kubectl apply -f diabetes-prediction-model-deployment.yaml
```

🙌 Credits

Created by `Shalindra Perera`

