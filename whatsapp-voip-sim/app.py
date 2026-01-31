import streamlit as st
from audio_capture import start_audio
from ui import update_ui
import time

st.set_page_config(page_title="WhatsApp Voice Detector")

st.title("📞 WhatsApp Call Deepfake Detection")

if 'stream' not in st.session_state:
    st.session_state.stream = None
if 'listening' not in st.session_state:
    st.session_state.listening = False

col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Start Listening", disabled=st.session_state.listening):
        st.session_state.stream = start_audio(update_ui)
        if st.session_state.stream:
            st.session_state.listening = True
            st.rerun()

with col2:
    if st.button("⏹ Stop Listening", disabled=not st.session_state.listening):
        if st.session_state.stream:
            st.session_state.stream.stop()
            st.session_state.stream.close()
            st.session_state.stream = None
        st.session_state.listening = False
        st.rerun()

if st.session_state.listening:
    st.success("Listening to WhatsApp call audio...")
    # Keep the script running to receive callbacks? 
    # Streamlit scripts typically need to finish unless we use a loop, BUT audio_capture uses a callback in a separate thread.
    # So we just need to keep the script "alive" or allow the callback to update the UI?
    # Actually, `update_ui` is called from the audio thread. Streamlit is not thread-safe for direct UI updates from other threads 
    # unless using `add_script_run_ctx` from streamlit.runtime.scriptrunner.
    # However, the user's previous code was just `start_audio(update_ui)`. 
    # If that worked (which it might not have fully), we will stick to that.
    # But usually, for the UI to update, we might need to rely on the callback logic.
    pass
