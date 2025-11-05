# inference/predict.py
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import joblib
import pandas as pd

# Globals (lazy-loaded)
MODEL: Optional[Any] = None
FEATURES: Optional[list] = None
MODEL_PATH: Optional[str] = None


def _default_models_dir() -> Path:
    """
    Resolve the models directory based on this file's location.
    Your repo layout has inference/ and ai-icu-monitoring/ as siblings.
    So we point to: ../ai-icu-monitoring/models
    """
    return (Path(__file__).resolve().parents[1] / "ai-icu-monitoring" / "models").resolve()


def _candidate_artifact_paths(models_dir: Path) -> list[Path]:
    """
    Build a prioritized list of candidate artifact paths to check.
    """
    names_in_priority = [
        "vitals_model_tuned.joblib",
        "vitals_model.joblib",
        "vitals_alert_model_v1.joblib",
    ]
    paths = [models_dir / n for n in names_in_priority]

    # Also allow any *.joblib / *.pkl as fallback (first match)
    paths += sorted(models_dir.glob("*.joblib"))
    paths += sorted(models_dir.glob("*.pkl"))
    return paths


def _resolve_artifact() -> Path:
    """
    Resolve the artifact path using (in order):
    1) MODEL_PATH env var (exact file)
    2) MODEL_DIR env var (directory to search)
    3) Default models dir inferred from this file
    Then search common filenames and fall back to *.joblib/*.pkl
    """
    # Explicit file override
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        p = Path(env_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"MODEL_PATH set but file does not exist: {p}")
        return p

    # Directory override
    models_dir = Path(os.getenv("MODEL_DIR") or _default_models_dir()).resolve()
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    # Candidate names + wildcards
    for cand in _candidate_artifact_paths(models_dir):
        if cand.exists():
            return cand

    raise FileNotFoundError(
        f"No model artifact found in {models_dir}. "
        f"Put a joblib/pkl file there or set MODEL_PATH."
    )


def _load_model_bundle(artifact: Path) -> Tuple[Any, Optional[list]]:
    """
    Load the joblib artifact. If it's a dict, unpack {model, features}; otherwise return the object as model.
    """
    bundle = joblib.load(artifact)
    if isinstance(bundle, dict):
        model = bundle.get("model", bundle)
        features = bundle.get("features")
    else:
        model = bundle
        features = None
    return model, features


def load_model() -> Tuple[Any, Optional[list], str]:
    """
    Load the model once and cache in globals.
    """
    global MODEL, FEATURES, MODEL_PATH
    artifact = _resolve_artifact()
    model, features = _load_model_bundle(artifact)

    MODEL = model
    FEATURES = features
    MODEL_PATH = str(artifact)
    print(f"[INFO] Loaded model: {MODEL_PATH}")
    return MODEL, FEATURES, MODEL_PATH


def predict_from_dict(reading: Dict[str, float]) -> Dict:
    """
    reading: dict of feature_name -> numeric value
    Returns: {"label": int, "score": float|None, "model_used": str}
    """
    global MODEL, FEATURES, MODEL_PATH
    # Lazy load to avoid crashing at import time
    if MODEL is None:
        load_model()

    # Determine feature order
    if FEATURES:
        feat_list = FEATURES
    else:
        # Fallback: sorted keys (works but not ideal if the model expects a fixed order)
        feat_list = sorted(reading.keys())

    # Build dataframe in correct order
    X = pd.DataFrame([[reading.get(f, None) for f in feat_list]], columns=feat_list)
    if X.isnull().any(axis=None):
        missing = X.columns[X.isnull().any()].tolist()
        raise ValueError(f"Missing or NaN features: {missing}")

    # Predict
    label = int(MODEL.predict(X)[0])

    score = None
    if hasattr(MODEL, "predict_proba"):
        try:
            score = float(MODEL.predict_proba(X)[0, 1])
        except Exception:
            score = None

    return {
        "label": label,
        "score": score,
        "model_used": MODEL_PATH if MODEL_PATH else "unknown",
    }
