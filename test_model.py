import joblib, pandas as pd
df = pd.read_csv("deployment_package/sample_data/vitals_combined_labeled.csv")
model = joblib.load("models/vitals_model.joblib")

sample = df.select_dtypes(include="number").iloc[0:1].drop(columns=["label"], errors="ignore")
pred = model.predict(sample)
print("Sample prediction:", pred[0])
