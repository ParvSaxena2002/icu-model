from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.interference_predict import predict_from_dict  # ✅ correct path
import uvicorn

app = FastAPI(title="ICU Vitals Inference")

class VitalsRequest(BaseModel):
    heart_rate: float
    bp_sys: float
    bp_dia: float
    spo2: float
    temp: float
    resp_rate: float

@app.post("/predict")
def predict(v: VitalsRequest):
    try:
        res = predict_from_dict(v.dict())
        return {
            "model_version": res.get("model_used"),
            "prediction": {"label": res["label"], "score": res["score"]},
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
