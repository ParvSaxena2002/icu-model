import pandas as pd
import joblib
import time
import winsound  # for beep sound on Windows

# Load the trained model
model = joblib.load("models/vitals_model.joblib")

# Load the dataset
df = pd.read_csv("deployment_package/sample_data/vitals_combined_labeled.csv")

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
