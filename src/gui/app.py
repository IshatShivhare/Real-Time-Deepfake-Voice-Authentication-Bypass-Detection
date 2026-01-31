import streamlit as st
import sys
import os
from pathlib import Path

# Ensure root is in path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.audio.capture import AudioCapturer
from src.gui.ui import setup_ui, update_status
from src.utils.logger import get_logger

logger = get_logger("GUI")

st.set_page_config(
    page_title="Deepfake Voice Detector",
    page_icon="🛡️",
    layout="centered"
)

def init_session():
    if 'capturer' not in st.session_state:
        # Initialize capturer (lazy load to avoid heavy model load on every run)
        # However, model loading handles itself.
        st.session_state.capturer = AudioCapturer(device_index=None) 
    
    if 'is_listening' not in st.session_state:
        st.session_state.is_listening = False

def main():
    init_session()
    
    st.title("🛡️ Deepfake Voice Detector")
    st.markdown("### Real-time Audio Authentication")
    
    # UI Layout
    status_container = setup_ui()
    
    # Control Panel
    col1, col2 = st.columns(2)
    
    if not st.session_state.is_listening:
        if col1.button("▶ Start Listening", use_container_width=True):
            st.session_state.is_listening = True
            st.session_state.capturer.start(ui_callback=lambda res: update_status(status_container, res))
            st.rerun()
    else:
        if col1.button("⏹ Stop Listening", type="primary", use_container_width=True):
            st.session_state.is_listening = False
            st.session_state.capturer.stop()
            st.rerun()
            
    # Settings (placeholder)
    with st.expander("Settings"):
        st.write("Device Selection: Default")
        st.write(f"Sample Rate: {st.session_state.capturer.sample_rate}")

    # Footer
    st.markdown("---")
    st.caption("Deepfake Voice Detection System v1.0")

if __name__ == "__main__":
    main()
