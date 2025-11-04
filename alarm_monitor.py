import pandas as pd
import joblib
import time
import winsound
import threading

# -----------------------------
# ⚙️ Load model and dataset
# -----------------------------
model_path = r"C:\Users\Dell\Downloads\icu-model-main\models\vitals_model.joblib"
data_path = r"C:\Users\Dell\Downloads\icu-model-main\deployment_package\sample_data\summary_features_added_data.csv"

print("📦 Loading model and dataset...")
model = joblib.load(model_path)
df = pd.read_csv(data_path)

# -----------------------------
# 🧩 Create features the model expects
# -----------------------------
# Convert your dataset columns to model-friendly names
df['heart rate'] = df['HR_mean']
df['respiratory rate'] = df['RR_mean']
df['oxygen saturation'] = df['SPO2_mean']
df['systolic blood pressure'] = df['SBP_mean']
df['diastolic blood pressure'] = df['DBP_mean']
df['pulse_pressure'] = df['SBP_mean'] - df['DBP_mean']

# If your model uses temperature but dataset doesn't include it, add a default value
df['body temperature'] = 37.0  # Normal temp in Celsius

# Add patient info (already present in dataset)
df['age'] = df['Age']
df['bmi'] = df['BMI']

# Select the features model expects (exact same order)
expected_features = [
    'heart rate', 'respiratory rate', 'body temperature',
    'oxygen saturation', 'systolic blood pressure',
    'diastolic blood pressure', 'pulse_pressure',
    'age', 'bmi'
]

# -----------------------------
# 🔍 Validate feature alignment
# -----------------------------
if len(model.feature_names_in_) != len(expected_features):
    raise ValueError(
        f"❌ Feature mismatch: model expects {len(model.feature_names_in_)} features "
        f"but script provides {len(expected_features)}. "
        f"\nModel expects: {list(model.feature_names_in_)}"
    )

X = df[expected_features]

print("✅ Features aligned successfully. Starting monitoring...\n")

# -----------------------------
# 🚨 Real-time Monitoring Logic
# -----------------------------
alert_triggered = False
alarm_acknowledged = False
stop_program = False

def acknowledge_input():
    """Wait for user to acknowledge alarm"""
    global alarm_acknowledged, stop_program
    while not stop_program:
        input("⚕️ Press ENTER to acknowledge and stop the alarm... ")
        if alert_triggered:
            alarm_acknowledged = True
            print("✅ Alarm acknowledged. Beep stopped.\n")
            break

# Run input listener on a separate thread
threading.Thread(target=acknowledge_input, daemon=True).start()

# Iterate through data rows
for index, row in X.iterrows():
    reading = row.values.reshape(1, -1)
    prediction = model.predict(reading)[0]

    if prediction == 1:  # abnormal vitals
        if not alert_triggered:
            alert_triggered = True
            print(f"🚨 ALERT: Abnormal vitals detected at row {index}")
            winsound.Beep(1000, 1000)
        elif not alarm_acknowledged:
            print(f"⚠️ Still abnormal at row {index} (waiting for acknowledgment)")
        else:
            print(f"🩺 Monitoring silently after acknowledgment (row {index})")
    else:
        if alert_triggered:
            print(f"✅ Vitals returned to normal at row {index}")
            alert_triggered = False
            alarm_acknowledged = False
        else:
            print(f"✅ Normal vitals at row {index}")

    if alarm_acknowledged:
        break  # Stop after acknowledgment

    time.sleep(1)

stop_program = True
print("🛑 Monitoring stopped after acknowledgment.")

