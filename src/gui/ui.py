import streamlit as st

def setup_ui():
    """
    Setup the main UI containers.
    Returns the container to update.
    """
    container = st.container()
    with container:
        st.info("Ready to analyze...")
    return container

def update_status(container, result):
    """
    Update the UI with prediction results.
    Args:
        container: Streamlit container
        result: Tuple (label, confidence)
    """
    label, confidence = result
    
    with container:
        container.empty()
        if label == "FAKE":
            st.error(f"🚨 **FAKE VOICE DETECTED**\n\nConfidence: {confidence:.2%}")
        else:
            st.success(f"✅ **REAL HUMAN VOICE**\n\nConfidence: {confidence:.2%}")
        
        st.progress(int(confidence * 100))
