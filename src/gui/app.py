import streamlit as st
import sys
import os
import tempfile
import shutil
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
        st.session_state.capturer = AudioCapturer(device_index=None)
    
    if 'is_listening' not in st.session_state:
        st.session_state.is_listening = False

def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            shutil.copyfileobj(uploaded_file, tmp_file)
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        return None

def main():
    init_session()
    
    st.title("🛡️ Deepfake Voice Detector")
    st.markdown("### Real-time & File-based Authentication")
    
    # UI Layout
    status_container = setup_ui()
    
    # Tabs for different modes
    tab_live, tab_file = st.tabs(["🎙️ Live Analysis", "📂 File Analysis"])
    
    # --- Live Analysis Tab ---
    with tab_live:
        st.subheader("WhatsApp VoIP / Microphone Analysis")
        
        # Hardcoded Device Index as requested
        # Matches Real-Time-Deepfake-Voice-Authentication-Bypass-Detection/whatsapp-voip-sim/audio_capture.py
        DEVICE_INDEX = 25
        
        # Update capturer device if changed (and not currently running)
        if not st.session_state.is_listening:
             st.session_state.capturer.device_index = DEVICE_INDEX

        st.info(f"Using Hardcoded Input Device Index: {DEVICE_INDEX}")

        col1, col2 = st.columns(2)
        
        col1, col2 = st.columns(2)
        
        if not st.session_state.is_listening:
            if col1.button("▶ Start Listening", use_container_width=True):
                st.session_state.is_listening = True
                st.session_state.capturer.start(ui_callback=lambda res: update_status(status_container, res))
                st.rerun()
        else:
            # IMPORTANT: Update the callback to the current session's container
            # This ensures that when the script reruns, the background thread targets the NEW container
            st.session_state.capturer.set_callback(lambda res: update_status(status_container, res))
            
            if col1.button("⏹ Stop Listening", type="primary", use_container_width=True):
                st.session_state.is_listening = False
                st.session_state.capturer.stop()
                st.rerun()
                
    # --- File Analysis Tab ---
    with tab_file:
        st.subheader("Upload Audio File")
        uploaded_file = st.file_uploader("Choose a WAV, MP3, or FLAC file", type=["wav", "mp3", "flac", "ogg"])
        
        if uploaded_file is not None:
            if st.button("Analyze File", use_container_width=True):
                with st.spinner("Analyzing audio file..."):
                    # Save to temp
                    temp_path = save_uploaded_file(uploaded_file)
                    if temp_path:
                        try:
                            # Use the detector from the capturer instance
                            detector = st.session_state.capturer.detector
                            prediction, confidence, details = detector.predict_single(temp_path)
                            
                            # Display result
                            result_text = "FAKE" if prediction == 1 else "REAL"
                            update_status(status_container, (result_text, confidence))
                            
                            st.divider()
                            st.write(f"**Result:** {result_text}")
                            st.write(f"**Confidence:** {confidence:.2%}")
                            with st.expander("Detailed Scores"):
                                st.json(details)
                                
                        except Exception as e:
                            st.error(f"Error analyzing file: {e}")
                        finally:
                            # Cleanup
                            if os.path.exists(temp_path):
                                os.remove(temp_path)

    # Footer
    st.markdown("---")
    st.caption("Deepfake Voice Detection System v1.1")

if __name__ == "__main__":
    main()
