
import pandas as pd
import joblib
import time
import winsound  # for beep sound on Windows

# Load the trained model
model = joblib.load("models/vitals_model.joblib")
winsound.Beep(1000, 1000)
# Load the dataset
df = pd.read_csv("processed/vitals_combined_labeled.csv")

# Select numeric columns (same as training)
X = df.select_dtypes(include="number").drop(columns=["label"], errors="ignore")

# Simulate real-time monitoring
for index, row in X.iterrows():
    reading = row.values.reshape(1, -1)
    prediction = model.predict(reading)[0]

    if prediction == 1:
        print(f"🚨 ALERT: Abnormal vitals detected at row {index}")
        # Beep sound — 1000Hz for 1 second
        winsound.Beep(1000, 1000)
    else:
        print(f"✅ Normal vitals at row {index}")

    # Delay between readings (1 second)
    time.sleep(1)


print(f"Heart Rate: {row['heart_rate']}, BP: {row['bp_sys']}/{row['bp_dia']}, SpO2: {row['spo2']}, Status: {'ALERT 🚨' if prediction == 1 else 'Normal ✅'}")


