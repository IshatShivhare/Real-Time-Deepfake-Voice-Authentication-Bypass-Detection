import sounddevice as sd
import numpy as np
import time
import sys
import os
import librosa
import soundfile as sf
import torch
from pathlib import Path

# Add parent directory to path to import ensemble_detector
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from ensemble_detector import DeepfakeDetector

# Configuration
DEVICE_INDEX = 25
SAMPLE_RATE = 48000
BUFFER_SIZE = 48000  # 1 second at 48kHz

# Initialize buffer
buffer = []

# Initialize Detector
print("Initializing Deepfake Detector...")

# Define paths
weight_path = parent_dir / 'weights' / 'librifake_pretrained_lambda0.5_epoch_25.pth'
config_path = parent_dir / 'models' / 'model_config_RawNet.yaml'

try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = DeepfakeDetector(device=device)
    detector.load_model(weight_path=str(weight_path), config_path=str(config_path))
    print("Detector initialized successfully.")
except Exception as e:
    print(f"Failed to initialize detector: {e}")
    sys.exit(1)

def process_window(audio_48k, ui_callback=None):
    try:
        # Convert list to numpy array if needed
        audio_np = np.array(audio_48k)
        
        # Resample to 16kHz
        audio_16k = librosa.resample(y=audio_np, orig_sr=SAMPLE_RATE, target_sr=16000)
        
        # Save temp file
        temp_file = "temp.wav"
        sf.write(temp_file, audio_16k, 16000)
        
        # Run prediction
        prediction, confidence, details = detector.predict_single(temp_file)
        
        result_text = "FAKE" if prediction == 1 else "REAL"
        color = "\033[91m" if prediction == 1 else "\033[92m" 
        reset = "\033[0m"
        
        print(f"\n{color}>> [Audio 1s] Prediction: {result_text} | Confidence: {confidence:.2%}{reset}")
        
        # Call UI callback if provided
        if ui_callback:
            try:
                ui_callback((result_text, confidence))
            except Exception as e:
                print(f"UI Callback Error: {e}")

    except Exception as e:
        print(f"Error processing window: {e}")

# Start Stream
def start_audio(ui_callback=None):
    global stream
    
    def wrapped_callback(indata, frames, time_info, status):
        # Pass the ui_callback to the global scope or process_window
        # To avoid complex scoping, we can modify process_window to use a global variable
        # OR better: make process_window accept the callback, but callback signature is fixed by sd.
        # So we'll use a global variable or closure.
        # Let's use the global `buffer` and `process_window` which we will modify to use `ui_callback`
        
        nonlocal ui_callback 
        # Actually simplest is to set a module-level variable for the callback or re-define process_window logic here.
        # But process_window is defined above.
        
        global buffer
        if status:
            print(status)
        
        buffer.extend(indata.flatten().tolist())

        if len(buffer) >= BUFFER_SIZE:
            audio_1s = buffer[:BUFFER_SIZE]
            buffer = buffer[BUFFER_SIZE:]
            process_window(audio_1s, ui_callback)

    try:
        stream = sd.InputStream(
            device=DEVICE_INDEX,
            channels=1,
            samplerate=SAMPLE_RATE,
            callback=wrapped_callback
        )
        stream.start()
        print(f"Listening to WhatsApp call audio on device {DEVICE_INDEX}...")
        return stream
    except Exception as e:
        print(f"Error starting audio stream: {e}")
        print("Please verify DEVICE_INDEX using 'python -m sounddevice'")
        return None

if __name__ == "__main__":
    s = start_audio()
    if s:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping...")
            s.stop()
