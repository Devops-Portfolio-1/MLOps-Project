# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import os
import requests
from uuid import uuid4

from app_logging import log_feedback, log_inference

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_diabetes_model.pkl"
INDEX_PAGE = Path(__file__).resolve().parent / "index.html"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


bundle = load_model()

if isinstance(bundle, dict) and "model" in bundle:
    model = bundle["model"]
    scaler = bundle.get("scaler")
    selected_features = bundle.get("selected_features", [])
else:
    model = bundle
    scaler = None
    selected_features = []

RAW_FEATURE_ORDER = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    "smoking_history",
]

SMOKING_MAP = {
    "No Info": 0,
    "never": 1,
    "not current": 2,
    "ever": 3,
    "former": 4,
    "current": 5,
}

GENDER_MAP = {"Female": 0, "Male": 1}

class DiabetesInput(BaseModel):
    gender: str = "Female"
    age: float
    hypertension: int
    heart_disease: int
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float
    smoking_history: str = "No Info"


## Feature mapping helper function 
#It takes the user’s input, turns it into the same kind of numbers the model was trained on, and returns one clean row so the model can make a prediction
def build_features(data: DiabetesInput) -> pd.DataFrame:
    frame = pd.DataFrame([data.model_dump()])
    frame["gender"] = frame["gender"].map(GENDER_MAP)
    frame["smoking_history"] = frame["smoking_history"].map(SMOKING_MAP)

    if frame["gender"].isna().any() or frame["smoking_history"].isna().any():
        raise HTTPException(status_code=400, detail="Invalid gender or smoking_history value")

    frame["bmi_category"] = pd.cut(
        frame["bmi"],
        bins=[0, 18.5, 24.9, 29.9, 100],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    ).astype(int)
    frame["age_group"] = pd.cut(
        frame["age"],
        bins=[0, 30, 45, 60, 100],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    ).astype(int)
    frame["risk_score"] = (
        frame["hypertension"]
        + frame["heart_disease"]
        + (frame["bmi"] > 30).astype(int)
        + (frame["HbA1c_level"] >= 5.7).astype(int)
        + (frame["blood_glucose_level"] >= 126).astype(int)
    )
    frame["age_bmi"] = frame["age"] * frame["bmi"]

    feature_frame = frame[selected_features] if selected_features else frame[RAW_FEATURE_ORDER]
    if scaler is not None:
        feature_frame = pd.DataFrame(scaler.transform(feature_frame), columns=feature_frame.columns)
    return feature_frame

@app.get("/")
def read_root():
    return FileResponse(INDEX_PAGE)

@app.post("/predict")
def predict(data: DiabetesInput):
    request_id = str(uuid4())
    # raw features as dict for logging
    raw_features = pd.DataFrame([data.model_dump()])
    input_data = build_features(data)

    # If KSERVE endpoint configured, forward the request
    kserve_url = os.getenv("KSERVE_PREDICTOR_URL")
    if kserve_url:
        # send processed features as a list to match most predictor shapes
        payload = {"instances": [input_data.iloc[0].tolist()]}
        try:
            resp = requests.post(kserve_url, json=payload, timeout=10)
            resp.raise_for_status()
            body = resp.json()

            prediction = None
            probability = None
            if isinstance(body, dict) and "predictions" in body:
                preds = body["predictions"]
                if isinstance(preds, list) and len(preds) > 0:
                    first = preds[0]
                    if isinstance(first, (list, tuple)):
                        prediction = int(first[0])
                        if len(first) > 1:
                            probability = float(first[1])
                    elif isinstance(first, dict):
                        prediction = int(first.get("prediction", first.get("pred", 0)))
                    else:
                        prediction = int(first)
            elif isinstance(body, list) and len(body) > 0:
                prediction = int(body[0])

            log_inference({
                "request_id": request_id,
                "features": raw_features.iloc[0].to_dict(),
                "prediction": int(prediction) if prediction is not None else None,
                "prediction_proba": float(probability) if probability is not None else None,
                "kserve_response": body,
            })

            response = {"request_id": request_id, "diabetic": bool(prediction)}
            if probability is not None:
                response["probability"] = float(probability)
            return response
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"KServe request failed: {e}")

    # Fallback to local model
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

    log_inference({
        "request_id": request_id,
        "features": raw_features.iloc[0].to_dict(),
        "prediction": int(prediction),
        "prediction_proba": float(probability) if probability is not None else None,
    })

    response = {"request_id": request_id, "diabetic": bool(prediction)}
    if probability is not None:
        response["probability"] = float(probability)
    return response
