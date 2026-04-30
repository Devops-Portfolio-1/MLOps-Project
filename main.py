# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

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
    input_data = build_features(data)
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

    response = {"diabetic": bool(prediction)}
    if probability is not None:
        response["probability"] = float(probability)
    return response
