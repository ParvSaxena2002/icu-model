
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("ai-icu-monitoring/models/vitals_model.joblib")

class InputData(BaseModel):
    values: list[float]

@app.get("/")
def home():
    return {"message": "ICU Model API is running"}

@app.post("/predict")
def predict(data: InputData):
    X = np.array(data.values).reshape(1, -1)
    prediction = model.predict(X)
    return {"prediction": prediction.tolist()}
