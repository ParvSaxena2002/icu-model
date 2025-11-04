import joblib
import pandas as pd

# === Paths ===
csv_path = r"C:\Users\Dell\Downloads\icu-model-main\deployment_package\sample_data\summary_features_added_data.csv"
model_path = r"C:\Users\Dell\Downloads\icu-model-main\models\vitals_model.joblib"

# === Load data and model ===
df = pd.read_csv(csv_path)
model = joblib.load(model_path)

# === Encode categorical ===
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].map({"M": 0, "F": 1})
if "Risk Category" in df.columns:
    df["Risk Category"] = pd.factorize(df["Risk Category"])[0]

# === Define the same features used during training ===
features = [
    "HR_mean", "RR_mean", "SBP_mean", "DBP_mean",
    "MBP_mean", "SPO2_mean", "Age", "BMI", "Gender"
]

# === Check for missing columns ===
missing = [f for f in features if f not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# === Use only those features ===
X = df[features]

# === Predict ===
sample = X.iloc[[0]]  # first row
pred = model.predict(sample)

print("✅ Sample prediction:", pred[0])
