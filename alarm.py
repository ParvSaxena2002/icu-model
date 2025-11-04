import os
import pandas as pd
import joblib
import threading
import time

# winsound is Windows-only; provide a fallback
try:
    import winsound  # type: ignore
    _HAS_WINSOUND = True
except Exception:
    winsound = None
    _HAS_WINSOUND = False

# === Dynamic Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.normpath(os.path.join(BASE_DIR, "deployment_package/sample_data/summary_features_added_data.csv"))
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "models/vitals_model_tuned.joblib"))

# === Load Model ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")
artifact = joblib.load(MODEL_PATH)

def find_model_and_features(obj):
    """Recursively find the first object with a predict method and any 'features' entry."""
    model = None
    features = None
    if isinstance(obj, dict):
        # direct keys
        if "model" in obj and hasattr(obj["model"], "predict"):
            model = obj["model"]
            features = obj.get("features")
            return model, features
        # search values
        for val in obj.values():
            m, f = find_model_and_features(val)
            if m is not None:
                return m, f or features
    else:
        if hasattr(obj, "predict"):
            return obj, None
    return None, None

model, artifact_features = find_model_and_features(artifact)
if model is None:
    # if artifact itself is a model-like object
    if hasattr(artifact, "predict"):
        model = artifact
    else:
        raise TypeError("Loaded artifact does not contain a model with a 'predict' method.")

print("✅ Model loaded successfully.")

# === Load Dataset ===
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"❌ Dataset not found: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print("✅ Dataset loaded successfully.")

# normalize column names to match training script
df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

# === Determine features to use (must match model training) ===
default_features = [
    "hr_mean", "rr_mean", "sbp_mean", "dbp_mean",
    "mbp_mean", "spo2_mean", "age", "bmi", "gender"
]

if artifact_features:
    features = [str(f).strip().replace(" ", "_").lower() for f in artifact_features]
else:
    features = default_features

# === Encode / normalize categorical columns ===
if "gender" in df.columns:
    # normalize gender to 0/1; keep numeric if already numeric
    df["gender"] = df["gender"].astype(str).str.strip().str.lower().map({"m": 0, "f": 1})
    # if mapping produced NaN for numeric strings, try coercion
    df["gender"] = pd.to_numeric(df["gender"], errors="coerce").fillna(df["gender"])

# === Check Feature Columns ===
missing = [col for col in features if col not in df.columns]
if missing:
    raise ValueError(f"❌ Missing columns in CSV: {missing}")

# select features and ensure numeric values
X = df[features].apply(pd.to_numeric, errors="coerce")

# fill NaNs with column mean, then 0 if any remain
X = X.fillna(X.mean()).fillna(0)

# === Flags ===
alert_triggered = False
stop_program = False

# beep helper
def beep():
    if _HAS_WINSOUND and 'winsound' in globals() and winsound is not None:
        winsound.Beep(1000, 700)
    else:
        # cross-platform audible fallback
        print("\a", end="", flush=True)
        time.sleep(0.7)

# === Input Thread Function ===
def wait_for_ack():
    """Wait for ENTER key to stop alarm."""
    global alert_triggered, stop_program
    try:
        input("⚕️ Press ENTER to stop the alarm and monitoring...\n")
    except Exception:
        pass
    alert_triggered = False
    stop_program = True
    print("✅ Alarm acknowledged. Monitoring stopped.")

# Start input listener
threading.Thread(target=wait_for_ack, daemon=True).start()

print("🩺 Monitoring patient vitals...\n")

# === Main Monitoring Loop ===
try:
    for index, row in X.iterrows():
        if stop_program:
            break

        reading = row.to_numpy().reshape(1, -1)
        try:
            # Use the discovered model object
            pred_raw = model.predict(reading)
        except Exception as e:
            print(f"❌ Prediction error at row {index}: {e}")
            continue

        # extract scalar prediction safely
        if hasattr(pred_raw, "__len__"):
            pred_val = pred_raw[0]
        else:
            pred_val = pred_raw

        try:
            prediction = int(pred_val)
        except Exception:
            prediction = int(float(pred_val))

        print(f"🔍 Row {index}: Prediction = {prediction}")

        if prediction == 1:
            print(f"🚨 ALERT: Abnormal vitals detected at row {index}")
            alert_triggered = True
            while alert_triggered and not stop_program:
                beep()
                time.sleep(0.3)
        else:
            print(f"🩶 Row {index}: Normal vitals.")
            time.sleep(1)

except KeyboardInterrupt:
    print("\n🛑 Monitoring interrupted by user.")
finally:
    print("🛑 Monitoring stopped.")
