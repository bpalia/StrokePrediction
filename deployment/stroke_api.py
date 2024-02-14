from fastapi import FastAPI
from enum import IntEnum
import joblib
import pandas as pd
from pydantic import BaseModel


# Pydantic classes for input and output


class YesNoEnum(IntEnum):
    yes = 1
    no = 0


class PatientInfo(BaseModel):
    age: float
    avg_glucose_level: float
    bmi: float
    hypertension: YesNoEnum = YesNoEnum.no
    heart_disease: YesNoEnum = YesNoEnum.no


class PredictionOut(BaseModel):
    predicted_probability: float
    high_stroke_risk: int


# Load the model
model = joblib.load("model.joblib")

# Start the app
app = FastAPI()


# Home page
@app.get("/")
def home():
    return {"message": "Stroke Risk Prediction App"}


# Prediction endpoint
@app.post("/predict", response_model=PredictionOut)
def predict(data: PatientInfo):
    df = pd.DataFrame([data.model_dump()])
    proba = model.predict_proba(df)[0, 1]
    prediction = model.predict(df)
    result = {"predicted_probability": proba, "high_stroke_risk": prediction}
    return result
