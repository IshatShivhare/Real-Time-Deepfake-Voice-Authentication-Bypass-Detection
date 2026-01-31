import streamlit as st

status_box = st.empty()
bar = st.empty()

def update_ui(result):
    label, confidence = result

    if label == "FAKE":
        status_box.error("🚨 AI VOICE DETECTED")
    else:
        status_box.success("✅ REAL HUMAN VOICE")

    bar.progress(int(confidence * 100))
