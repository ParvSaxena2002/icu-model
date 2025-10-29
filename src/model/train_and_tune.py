import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

# --- Config ---
DATA_PATH = "data/processed/vitals_combined_labeled.csv"
OUT_MODEL = "models/vitals_model_tuned.joblib"
REPORT_PATH = "reports/training_report.txt"
FEATURE_IMPORTANCE_PNG = "reports/feature_importance.png"
RANDOM_STATE = 42

# Ensure folders exist
Path("models").mkdir(parents=True, exist_ok=True)
Path("reports").mkdir(parents=True, exist_ok=True)

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data not found at {path}. Run preprocessing first.")
    df = pd.read_csv(path)
    return df

def select_features(df):
    # Preferred feature names; modify if your CSV uses different names
    preferred = ['heart_rate','bp_sys','bp_dia','spo2','temp','resp_rate']
    features = [c for c in preferred if c in df.columns]
    if not features:
        # fallback: use numeric columns except label
        features = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'label']
    return features

def main():
    print("Loading data...")
    df = load_data(DATA_PATH)
    print("Data loaded. Shape:", df.shape)

    if 'label' not in df.columns:
        raise ValueError("No 'label' column found in processed data. Ensure dataset is labeled 0/1.")

    features = select_features(df)
    print("Using features:", features)

    X = df[features].fillna(method='ffill').fillna(0)
    y = df['label'].astype(int)

    # quick class balance info
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print("Class distribution (original):", class_counts)

    # train-test split (stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # SMOTE: oversample minority class in training set
    print("Applying SMOTE to training set ...")
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print("After SMOTE train distribution:", np.bincount(y_train_res))

    # Define base model
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

    # Parameter grid (keep small to reduce runtime); expand later if needed
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', None]
    }

    # GridSearch (optimize recall to reduce missed alerts)
    print("Starting GridSearchCV (this may take time)...")
    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=4,
        scoring='recall',
        n_jobs=-1,
        verbose=2
    )
    grid.fit(X_train_res, y_train_res)

    best = grid.best_estimator_
    print("Grid search complete. Best params:", grid.best_params_)

    # Evaluate on test set
    preds = best.predict(X_test)
    probs = best.predict_proba(X_test)[:,1] if hasattr(best, "predict_proba") else None

    report = classification_report(y_test, preds, digits=4)
    cm = confusion_matrix(y_test, preds)
    auc = None
    if probs is not None:
        try:
            auc = roc_auc_score(y_test, probs)
        except Exception:
            auc = None

    # Save model artifact with feature list
    joblib.dump({"model": best, "features": features}, OUT_MODEL)
    print(f"Saved tuned model to: {OUT_MODEL}")

    # Save a simple text report
    with open(REPORT_PATH, "w", encoding="utf8") as f:
        f.write("TRAINING REPORT\n")
        f.write("================\n\n")
        f.write(f"Data path: {DATA_PATH}\n")
        f.write(f"Model path: {OUT_MODEL}\n\n")
        f.write("Features used:\n")
        for feat in features:
            f.write(" - " + feat + "\n")
        f.write("\nClass distribution (original): " + str(class_counts) + "\n")
        f.write("After SMOTE (train): " + str(np.bincount(y_train_res).tolist()) + "\n\n")
        f.write("Best params:\n")
        f.write(str(grid.best_params_) + "\n\n")
        f.write("Classification report (test set):\n")
        f.write(report + "\n\n")
        f.write("Confusion matrix:\n")
        f.write(str(cm) + "\n\n")
        if auc is not None:
            f.write(f"ROC AUC: {auc:.4f}\n")
    print(f"Saved training report to: {REPORT_PATH}")

    # Feature importance plot
    try:
        importances = best.feature_importances_
        plt.figure(figsize=(8,4))
        ys = np.argsort(importances)[::-1]
        feat_sorted = [features[i] for i in ys]
        vals_sorted = importances[ys]
        plt.bar(feat_sorted, vals_sorted)
        plt.title("Feature importances")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.savefig(FEATURE_IMPORTANCE_PNG, bbox_inches='tight')
        print(f"Saved feature importance plot to: {FEATURE_IMPORTANCE_PNG}")
    except Exception as e:
        print("Could not create feature importance plot:", e)

    # Print summary to console
    print("\n=== SUMMARY ===")
    print("Best params:", grid.best_params_)
    print("\nClassification report:\n", report)
    print("\nConfusion matrix:\n", cm)
    if auc is not None:
        print("\nROC AUC:", auc)
    print("\nDone.")
    
if __name__ == "_main_":
    main()

import joblib
art = joblib.load("models/vitals_model_tuned.joblib")
m = art["model"]
feats = art["features"]
print("Model:", type(m))
print("Features:", feats)
print("Importances:", dict(zip(feats, m.feature_importances_)))