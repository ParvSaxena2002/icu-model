# src/interference_predict.py
from pathlib import Path
import os
import joblib

MODEL = None
FEATURES = None
MODEL_PATH = None

def _models_dir() -> Path:
    """
    Detects the absolute path to the models directory.
    Works even if src/ and ai-icu-monitoring/ are siblings.
    """
    # ../ai-icu-monitoring/models/
    return (Path(__file__).resolve().parents[1] / "ai-icu-monitoring" / "models").resolve()

def _find_artifact(models_dir: Path) -> Path:
    # Allow override through environment variable
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        return Path(env_path).resolve()

    for pattern in ("vitals_model.joblib", "icu_model.pkl", "*.joblib", "*.pkl"):
        matches = list(models_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No model artifact found in {models_dir}")

def load_model():
    """
    Loads a model file (.joblib or .pkl) once, stores it globally.
    """
    global MODEL, FEATURES, MODEL_PATH

    models_dir = Path(os.getenv("MODEL_DIR") or _models_dir())
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    artifact = _find_artifact(models_dir)
    print(f"[INFO] Loading model from: {artifact}")
    bundle = joblib.load(artifact)

    if isinstance(bundle, dict):
        MODEL = bundle.get("model", bundle)
        FEATURES = bundle.get("features")
    else:
        MODEL = bundle
        FEATURES = None

    MODEL_PATH = str(artifact)
    return MODEL, FEATURES, MODEL_PATH


def predict_from_dict(data: dict):
    """
    Performs a simple inference using the loaded model.
    """
    global MODEL
    if MODEL is None:
        load_model()  # lazy load on first call

    import numpy as np
    x = np.array([[data["heart_rate"], data["bp_sys"], data["bp_dia"],
                   data["spo2"], data["temp"], data["resp_rate"]]])

    pred = MODEL.predict(x)
    score = MODEL.predict_proba(x)[0, 1] if hasattr(MODEL, "predict_proba") else None

    return {"label": int(pred[0]), "score": float(score) if score is not None else None, "model_used": MODEL_PATH}
