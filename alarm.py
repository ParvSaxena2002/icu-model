import pandas as pd
import joblib
import threading
import time
import winsound
import os

# === File Paths ===
model_path = r"C:\Users\Dell\Downloads\icu-model-main\models\icu_model.pkl"
csv_path = r"C:\Users\Dell\Downloads\icu-model-main\deployment_package\sample_data\summary_features_added_data.csv"

# === Load Model ===
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model not found: {model_path}")
model = joblib.load(model_path)
print("✅ Model loaded successfully.")

# === Load Dataset ===
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ Dataset not found: {csv_path}")
df = pd.read_csv(csv_path)
print("✅ Dataset loaded successfully.")

# === Encode Columns ===
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].map({'M': 0, 'F': 1})
    print("Converted 'Gender' column to numeric values.")
if 'Risk Category' in df.columns:
    df['Risk Category'] = pd.factorize(df['Risk Category'])[0]
    print("Converted 'Risk Category' column to numeric values.")

# === Define Features (must match model training) ===
features = [
    'HR_mean', 'RR_mean', 'SBP_mean', 'DBP_mean',
    'MBP_mean', 'SPO2_mean', 'Age', 'BMI', 'Gender'
]

# === Check Feature Columns ===
missing = [col for col in features if col not in df.columns]
if missing:
    raise ValueError(f"❌ Missing columns in CSV: {missing}")

X = df[features]

# === Flags ===
alert_triggered = False
stop_program = False

# === Input Thread Function ===
def wait_for_ack():
    """Wait for ENTER key to stop alarm."""
    global alert_triggered, stop_program
    input("⚕️ Press ENTER to stop the alarm and monitoring...\n")
    alert_triggered = False
    stop_program = True
    print("✅ Alarm acknowledged. Monitoring stopped.")

# Start input listener
threading.Thread(target=wait_for_ack, daemon=True).start()

print("🩺 Monitoring patient vitals...\n")

# === Main Monitoring Loop ===
for index, row in X.iterrows():
    if stop_program:
        break

    reading = row.values.reshape(1, -1)
    prediction = model.predict(reading)[0]
    print(f"🔍 Row {index}: Prediction = {prediction}")

    # 🚨 For testing — Force alarm on first row
    if index == 0:
        prediction = 1

    if prediction == 1:
        print(f"🚨 ALERT: Abnormal vitals detected at row {index}")
        alert_triggered = True
        while alert_triggered and not stop_program:
            winsound.Beep(1000, 700)  # frequency, duration(ms)
            time.sleep(0.3)
    else:
        print(f"🩶 Row {index}: Normal vitals.")
        time.sleep(1)


print("🛑 Monitoring stopped.")
