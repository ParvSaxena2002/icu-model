#!/usr/bin/env python3
"""
inference_predict.py
--------------------
Load the trained model and run predictions from command line or programmatically.

Usage examples:
    python inference_predict.py
    python inference_predict.py --hr_mean 110 --sbp_mean 125 --dbp_mean 85 --spo2_mean 95
"""

import os
import sys
import joblib
import argparse
import pandas as pd
from typing import Dict, Any

# -----------------------------
# MODEL LOADING
# -----------------------------
def load_model() -> tuple[Any, list[str], str]:
    """Load model from the 'models' directory, regardless of current path."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")

    candidates = [
        os.path.join(models_dir, "vitals_model_tuned.joblib"),
        os.path.join(models_dir, "vitals_model.joblib"),
        os.path.join(models_dir, "vitals_alert_model_v1.joblib"),
    ]

    for path in candidates:
        if os.path.exists(path):
            model_artifact = joblib.load(path)
            if isinstance(model_artifact, dict) and "model" in model_artifact:
                model = model_artifact["model"]
                features = model_artifact.get("features", [])
                if features is None:
                    features = []
            else:
                model = model_artifact
                features = []

            if hasattr(model, "predict"):
                print(f"[INFO] Loaded model from: {path}")
                return model, features, path

    raise FileNotFoundError("❌ Model not found. Train it first with train_model.py.")


# -----------------------------
# INFERENCE FUNCTION
# -----------------------------
def predict_from_dict(reading: Dict[str, float], model, features, model_path: str) -> Dict:
    """Run inference on a single reading dictionary."""
    if features is None:
        feat_list = sorted(reading.keys())
    else:
        feat_list = features

    X = pd.DataFrame([[reading.get(f, None) for f in feat_list]], columns=feat_list)

    if X.isnull().any(axis=None):
        missing = X.columns[X.isnull().any()].tolist()
        raise ValueError(f"Missing or NaN features: {missing}")

    label = int(model.predict(X)[0])
    score = None
    if hasattr(model, "predict_proba"):
        try:
            score = float(model.predict_proba(X)[0, 1])
        except Exception:
            score = None

    return {"label": label, "score": score, "model_used": os.path.basename(model_path)}


# -----------------------------
# MAIN SCRIPT ENTRY
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Run inference using vitals model.")
    parser.add_argument("--hr_mean", type=float, help="Mean heart rate")
    parser.add_argument("--sbp_mean", type=float, help="Mean systolic BP")
    parser.add_argument("--dbp_mean", type=float, help="Mean diastolic BP")
    parser.add_argument("--spo2_mean", type=float, help="Mean SpO2")

    args = parser.parse_args()

    MODEL, FEATURES, MODEL_PATH = load_model()

    # Default sample if no CLI args provided
    reading = {
        "hr_mean": args.hr_mean or 80.0,
        "sbp_mean": args.sbp_mean or 120.0,
        "dbp_mean": args.dbp_mean or 80.0,
        "spo2_mean": args.spo2_mean or 98.0,
    }

    print(f"[INFO] Running inference with input: {reading}")
    result = predict_from_dict(reading, MODEL, FEATURES, MODEL_PATH)
    print(f"[RESULT] → {result}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
