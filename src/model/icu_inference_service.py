# file: icu_inference_service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import torch
import numpy as np
import time
import requests
import threading

# --- Configurable ---
MODEL_PATH = "models/icu_risk_model.pt"   # path to your trained model
SEQ_LEN = 60   # number of timesteps model expects
FEATURES = ["hr","sbp","dbp","spo2","rr","temp"]  # features order must match model
THRESHOLD_ALERT = 0.75  # example threshold for high risk
WEBHOOK_URL = "https://example-hospital-system/alerts"  # optional alert endpoint
DEVICE = torch.device("cpu")  # or "cuda" if available and appropriate
# ----------------------

app = FastAPI(title="ICU Inference Service")

# In-memory buffer to hold recent time series per patient
# In production replace with redis/DB eviction logic
patient_buffers: Dict[str, List[Dict[str, Any]]] = {}

# Normalization statistics: mean and std per feature used during training
# Replace with your training normalization params
NORMALIZER = {
    "hr": {"mean": 80.0, "std": 15.0},
    "sbp": {"mean": 120.0, "std": 20.0},
    "dbp": {"mean": 70.0, "std": 12.0},
    "spo2": {"mean": 97.0, "std": 2.0},
    "rr": {"mean": 16.0, "std": 4.0},
    "temp": {"mean": 36.8, "std": 0.6},
}

# Pydantic model for incoming payload
class VitalsPayload(BaseModel):
    patient_id: str = Field(..., example="patient-1234")
    timestamp: float = Field(default_factory=time.time)
    vitals: Dict[str, float]  # e.g. {"hr": 110, "sbp": 90, ...}

# Load model
def load_model(path: str):
    # Example: torch.load or load scripted model
    model = torch.load(path, map_location=DEVICE)
    model.eval()
    return model

model = load_model(MODEL_PATH)

# Preprocess a list of vitals dicts into model input (seq_len, features)
def preprocess_sequence(seq: List[Dict[str, Any]]):
    """
    seq: list of dicts with keys matching FEATURES in chronological order (oldest -> newest)
    returns: np.array shaped (1, seq_len, n_features) of floats
    """
    arr = np.zeros((SEQ_LEN, len(FEATURES)), dtype=np.float32)
    # We'll right-align available data; pad with last-known or zeros for missing
    start = max(0, SEQ_LEN - len(seq))
    for i in range(len(seq)):
        d = seq[i]
        for j, feat in enumerate(FEATURES):
            v = d.get(feat, np.nan)
            if np.isnan(v):
                # If missing, try to use previous value in seq
                v = seq[i-1].get(feat, 0) if i > 0 else 0.0
            # normalize
            m = NORMALIZER[feat]["mean"]
            s = NORMALIZER[feat]["std"]
            arr[start + i, j] = (v - m) / (s + 1e-8)
    # If the earliest rows are zeros (padding), replicate first available row to reduce step-change
    if start > 0 and len(seq) > 0:
        arr[:start] = arr[start:start+1]
    return arr.reshape(1, SEQ_LEN, len(FEATURES))

# Run inference on preprocessed input
def run_inference(model, input_array: np.ndarray) -> float:
    """
    input_array shape: (1, seq_len, features)
    returns: risk score float between 0 and 1
    """
    with torch.no_grad():
        x = torch.from_numpy(input_array).to(DEVICE)
        # adjust depending on model's forward signature
        logits = model(x)  # e.g. returns tensor shape (1,1) or (1,)
        if isinstance(logits, tuple) or isinstance(logits, list):
            logits = logits[0]
        score = torch.sigmoid(logits).cpu().numpy().squeeze()
        # ensure scalar
        return float(score)

# Alerting helper (non-blocking)
def send_alert_async(payload: Dict[str, Any]):
    def worker(p):
        try:
            requests.post(WEBHOOK_URL, json=p, timeout=2.0)
        except Exception as e:
            # log failure (print for this template)
            print("Alert send failed:", e)
    t = threading.Thread(target=worker, args=(payload,), daemon=True)
    t.start()

@app.post("/ingest", summary="Ingest a new vital sign sample")
def ingest(payload: VitalsPayload):
    pid = payload.patient_id
    if pid not in patient_buffers:
        patient_buffers[pid] = []
    buf = patient_buffers[pid]

    # append new sample
    buf.append({"timestamp": payload.timestamp, **payload.vitals})
    # keep only last SEQ_LEN samples
    if len(buf) > SEQ_LEN:
        buf[:] = buf[-SEQ_LEN:]

    # if we have enough samples to run inference
    if len(buf) >= int(SEQ_LEN * 0.5):  # allow inference even if 50% filled; change as needed
        seq = buf[-SEQ_LEN:] if len(buf) >= SEQ_LEN else buf
        model_input = preprocess_sequence(seq)
        try:
            score = run_inference(model, model_input)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

        result = {
            "patient_id": pid,
            "timestamp": time.time(),
            "risk_score": score,
            "n_timesteps": len(seq),
        }

        # if above threshold, emit alert
        if score >= THRESHOLD_ALERT:
            alert_payload = {
                "patient_id": pid,
                "risk_score": score,
                "event": "HIGH_RISK",
                "timestamp": time.time(),
                "context": {"recent_vitals": seq[-5:]},  # include last few samples
            }
            # Non-blocking send to hospital alert system
            if WEBHOOK_URL:
                send_alert_async(alert_payload)
            result["alerted"] = True
        else:
            result["alerted"] = False

        return result

    # not enough data yet
    return {
        "patient_id": pid,
        "timestamp": time.time(),
        "message": f"Buffered sample; need {SEQ_LEN} timesteps for full-length inference. Current buffer: {len(buf)}",
        "buffered": len(buf),
    }

@app.get("/status/{patient_id}", summary="Get buffer status for a patient")
def status(patient_id: str):
    buf = patient_buffers.get(patient_id, [])
    return {"patient_id": patient_id, "buffered": len(buf), "latest": buf[-1] if buf else None}

# For local testing only: run uvicorn icu_inference_service:app --reload --port 8000
