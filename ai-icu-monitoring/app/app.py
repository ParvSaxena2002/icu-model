# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # Allow frontend access (React, Streamlit, etc.)

# === Load Model ===
MODEL_PATH = os.path.join("models", "vitals_model.joblib")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# === Health Check Endpoint ===
@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "status": "API is running",
        "model": "vitals_model.joblib",
        "uptime": "OK"
    })

# === Prediction Endpoint ===
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        features = [
            "HR_mean", "RR_mean", "SBP_mean", "DBP_mean",
            "MBP_mean", "SPO2_mean", "Age", "BMI", "Gender"
        ]
        df = pd.DataFrame([data], columns=features)
        pred = model.predict(df)[0]

        if pred == 1:
            return jsonify({"prediction": 1, "status": "Abnormal vitals detected 🚨"})
        else:
            return jsonify({"prediction": 0, "status": "Normal vitals"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# === Optional Endpoints (for expansion) ===
@app.route("/api/train", methods=["POST"])
def retrain():
    return jsonify({"status": "Retrain endpoint active (not implemented yet)"})

@app.route("/api/acknowledge", methods=["POST"])
def acknowledge():
    return jsonify({"message": "Alarm acknowledged ✅"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
