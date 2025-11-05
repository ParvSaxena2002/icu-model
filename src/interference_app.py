# inference/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from interference.predict import predict_from_dict
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
        # return a clear, simple JSON
        return {"model_version": res.get("model_used"), "prediction": {"label": res["label"], "score": res["score"]}}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

# If you run python inference/app.py directly, start uvicorn programmatically
if __name__ == "__main__":
    # run: python inference\app.py
    uvicorn.run(app, host="127.0.0.1", port=8000)
