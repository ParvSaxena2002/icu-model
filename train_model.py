import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib, os

df = pd.read_csv("data/processed/vitals_combined_labeled.csv")

# select only numeric columns for training
X = df.select_dtypes(include="number").drop(columns=["label"], errors="ignore")
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(classification_report(y_test, preds))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/vitals_model.joblib")
print("✅ Model saved to models/vitals_model.joblib")
