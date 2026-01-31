import streamlit as st
from audio_capture import start_audio
from ui import update_ui

st.set_page_config(page_title="WhatsApp Voice Detector")

st.title("📞 WhatsApp Call Deepfake Detection")

if st.button("▶ Start Listening"):
    start_audio(update_ui)
    st.success("Listening to WhatsApp call audio...")
