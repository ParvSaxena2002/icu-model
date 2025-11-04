import os
import joblib
import pandas as pd

# === Dynamic Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.normpath(os.path.join(BASE_DIR, "deployment_package/sample_data/summary_features_added_data.csv"))
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "models/vitals_model_tuned.joblib"))

# === Check files exist ===
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# === Load data and model ===
df = pd.read_csv(CSV_PATH)

# normalize column names to match training script
df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

artifact = joblib.load(MODEL_PATH)
# artifact might be a dict with {"model": model, "features": [...]} or other packaging
features_used = None
model = None
if isinstance(artifact, dict):
    # common case: artifact stores model under "model"
    if "model" in artifact:
        model = artifact["model"]
        features_used = artifact.get("features")
    else:
        # try to find any value in the dict that looks like an estimator (has predict)
        for val in artifact.values():
            if hasattr(val, "predict"):
                model = val
                break
        # still capture features if present
        features_used = artifact.get("features")
else:
    model = artifact

if model is None or not hasattr(model, "predict"):
    raise TypeError(f"Loaded artifact does not contain a model with a 'predict' method. Artifact keys: {list(artifact.keys()) if isinstance(artifact, dict) else type(artifact)}")

# expected feature names (lowercased to match df)
default_features = [
    "hr_mean", "rr_mean", "sbp_mean", "dbp_mean",
    "mbp_mean", "spo2_mean", "age", "bmi", "gender"
]
features = [f.lower() for f in (features_used or default_features)]

# === Encode categorical ===
if "gender" in df.columns:
    df["gender"] = df["gender"].map({"M": 0, "F": 1, "m": 0, "f": 1}).fillna(df["gender"])

# === Check for missing columns ===
missing = [f for f in features if f not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# === Use only those features ===
X = df[features]

# === Predict ===
sample = X.iloc[[0]]  # first row (preserve 2D shape)
pred = model.predict(sample)

print("✅ Sample prediction:", int(pred[0]) if hasattr(pred[0], "item") or isinstance(pred[0], (int, float)) else pred[0])
