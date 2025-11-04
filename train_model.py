import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# === File paths ===
csv_path = r"C:\Users\Dell\Downloads\icu-model-main\deployment_package\sample_data\summary_features_added_data.csv"
model_path = r"C:\Users\Dell\Downloads\icu-model-main\models\vitals_model.joblib"

# === Load dataset ===
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ Dataset not found: {csv_path}")

df = pd.read_csv(csv_path)
print(f"✅ Loaded dataset with shape: {df.shape}")

# === Encode categorical columns ===
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].map({"M": 0, "F": 1})
    print("Converted Gender to numeric values.")
if "Risk Category" in df.columns:
    df["Risk Category"] = pd.factorize(df["Risk Category"])[0]
    print("Converted Risk Category to numeric values.")

# === Define features and target ===
features = [
    "HR_mean", "RR_mean", "SBP_mean", "DBP_mean",
    "MBP_mean", "SPO2_mean", "Age", "BMI", "Gender"
]

# Detect target column
possible_targets = ["Risk Category", "label", "Outcome", "Alert"]
target = next((col for col in possible_targets if col in df.columns), None)

if not target:
    raise ValueError("❌ No valid target column found in CSV.")

df = df.dropna(subset=features + [target])
X = df[features]
y = df[target]

# === Split & train ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
preds = model.predict(X_test)
print(f"✅ Training complete. Accuracy: {accuracy_score(y_test, preds):.2f}")
print(classification_report(y_test, preds))

# === Save model ===
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)
print(f"💾 Model saved at: {model_path}")
