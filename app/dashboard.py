# app/dashboard.py
import streamlit as st
import pandas as pd
import time
import joblib
import os
import numpy as np

# Try to import winsound for Windows beep (optional)
try:
    import winsound
    def play_beep():
        winsound.Beep(1000, 700)  # freq=1000Hz, duration=700ms
except Exception:
    def play_beep():
        pass  # no beep if not available

st.set_page_config(page_title="AI ICU Monitor", layout="wide")
st.title("🏥 AI ICU Monitoring Dashboard (Simulation)")

MODEL_PATHS = [
    "models/vitals_alert_model_v1.joblib",
    "models/vitals_model.joblib",
    "models/vitals_alert_model.joblib"
]

# Load model (try several expected filenames)
model = None
for p in MODEL_PATHS:
    if os.path.exists(p):
        try:
            art = joblib.load(p)
            # art can be a raw model or dict {"model":..., "features":[...]}
            if isinstance(art, dict) and "model" in art:
                model = art["model"]
                features = art.get("features", None)
            else:
                model = art
                features = None
            st.sidebar.write(f"Loaded model: {p}")
            break
        except Exception as e:
            st.sidebar.write(f"Failed to load {p}: {e}")
if model is None:
    st.sidebar.error("Model not found in models/*.joblib. Train model first.")
    st.stop()

# Load processed data
DATA_PATH = "data/processed/vitals_combined_labeled.csv"
if not os.path.exists(DATA_PATH):
    st.sidebar.error("Processed data not found: " + DATA_PATH)
    st.stop()

df = pd.read_csv(DATA_PATH)

# Determine numeric feature columns automatically if features not provided
if features is None:
    # choose typical vitals in preference order
    preferred = ['heart_rate','bp_sys','bp_dia','spo2','temp','resp_rate']
    features = [c for c in preferred if c in df.columns]
    if not features:
        # fallback to numeric columns except label
        features = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'label']

st.sidebar.markdown("**Model features used:**")
st.sidebar.write(features)

# Sidebar controls
st.sidebar.header("Simulation Controls")
delay = st.sidebar.slider("Update delay (seconds)", 0.5, 5.0, 1.0, step=0.5)
start_idx = st.sidebar.number_input("Start row index", min_value=0, max_value=max(0, len(df)-1), value=0, step=1)
auto_run = st.sidebar.checkbox("Auto-run simulation", value=False)
loop = st.sidebar.checkbox("Loop when finished", value=False)
sound_on = st.sidebar.checkbox("Play beep on alert (Windows)", value=True)

# If model predict_proba available, show probability
use_proba = hasattr(model, "predict_proba")

# Layout: two columns - left metrics, right history/log
col1, col2 = st.columns([1, 1])

# Initialize session state
if "idx" not in st.session_state:
    st.session_state.idx = start_idx
if "log" not in st.session_state:
    st.session_state.log = []

# Metrics area
with col1:
    st.subheader("Current Reading")
    metric_placeholders = {f: st.empty() for f in features}
    status_placeholder = st.empty()
    prob_placeholder = st.empty()

# History / log area
with col2:
    st.subheader("Alerts Log (most recent on top)")
    log_box = st.empty()

# Helper: format reading row into dict for model
def make_reading(row):
    return {f: float(row[f]) for f in features}

# Run one step function
def run_step(i):
    row = df.iloc[i]
    reading = make_reading(row)
    X = pd.DataFrame([ [reading[f] for f in features] ], columns=features)
    try:
        pred = model.predict(X)[0]
    except Exception as e:
        st.error("Model prediction error: " + str(e))
        pred = 0
    proba = None
    if use_proba:
        try:
            proba = model.predict_proba(X)[0][1]
        except Exception:
            proba = None

    # Update metric displays
    for f in features:
        metric_placeholders[f].metric(label=f.replace("_"," ").title(), value=reading[f])
    status_text = "🚨 ALERT — HIGH RISK" if int(pred) == 1 else "✅ Normal"
    if int(pred) == 1:
        status_placeholder.error(status_text)
        if sound_on:
            play_beep()
    else:
        status_placeholder.success(status_text)

    if proba is not None:
        prob_placeholder.write(f"Alert score (probability): {proba:.2f}")

    # update log
    entry = {
        "index": int(i),
        "status": "ALERT" if int(pred) == 1 else "Normal",
        "reading": reading,
        "prob": float(proba) if proba is not None else None
    }
    st.session_state.log.insert(0, entry)
    # keep last 200
    st.session_state.log = st.session_state.log[:200]
    # display log
    log_df = pd.DataFrame([{
        "Index": e["index"],
        "Status": e["status"],
        **{k: v for k,v in e["reading"].items()},
        "Score": e["prob"]
    } for e in st.session_state.log])
    log_box.dataframe(log_df)

# Buttons
col_buttons = st.container()
with col_buttons:
    run_once = st.button("Run one step")
    if st.button("Reset log"):
        st.session_state.log = []
    if st.button("Jump to start"):
        st.session_state.idx = int(start_idx)

# Manual single step
if run_once:
    run_step(st.session_state.idx)
    st.session_state.idx += 1

# Auto-run loop (non-blocking-ish) - use while but allow stop via ctrl+C or page reload
if auto_run:
    # note: Streamlit will run this and block; it's fine for simple local demos.
    i = st.session_state.idx
    while True:
        run_step(i)
        i += 1
        st.session_state.idx = i
        if i >= len(df):
            if loop:
                i = int(start_idx)
            else:
                break
        time.sleep(delay)
    st.experimental_rerun()

st.write("---")
st.markdown("**How to use:** Start index chooses the row where simulation begins. Toggle `Auto-run` to stream rows continuously. Enable sound for beep on alerts (Windows only).")