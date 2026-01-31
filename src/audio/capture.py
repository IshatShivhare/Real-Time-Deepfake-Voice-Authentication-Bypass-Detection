import sounddevice as sd
import numpy as np
import soundfile as sf
import librosa
import torch
import tempfile
import os
from src.utils.logger import get_logger
from src.models.ensemble import EnsembleDetector

logger = get_logger("AudioCapture")

class AudioCapturer:
    def __init__(self, device_index=None, sample_rate=48000, buffer_duration=1.0):
        self.device_index = device_index
        self.sample_rate = sample_rate # Capture rate
        self.buffer_size = int(sample_rate * buffer_duration)
        self.buffer = []
        self.stream = None
        self.running = False
        
        self.detector = None
        self._init_detector()
        
    def _init_detector(self):
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.detector = EnsembleDetector(device=device)
            # Assuming load methods are unified or we call them explicitly
            # self.detector.load_models() # We need to ensure EnsembleDetector has this
            logger.info("Detector initialized in capture module.")
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            
    def process_window(self, audio_48k, ui_callback=None):
        try:
            audio_np = np.array(audio_48k)
            
            # Temporary: Save to file because predict_single expects file
            # Ideally: predict_single should accept tensor/array
            
            # Resample to 16k for model
            audio_16k = librosa.resample(y=audio_np, orig_sr=self.sample_rate, target_sr=16000)
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                sf.write(tf.name, audio_16k, 16000)
                temp_path = tf.name
            
            try:
                prediction, confidence, details = self.detector.predict_single(temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            result_text = "FAKE" if prediction == 1 else "REAL"
            
            logger.info(f"Prediction: {result_text} | Confidence: {confidence:.2%}")
            
            if ui_callback:
                ui_callback((result_text, confidence))
                
        except Exception as e:
            logger.error(f"Error processing window: {e}")

    def audio_callback(self, indata, frames, time_info, status, ui_callback=None):
        if status:
            logger.warning(f"Audio status: {status}")
            
        self.buffer.extend(indata.flatten().tolist())
        
        if len(self.buffer) >= self.buffer_size:
            audio_chunk = self.buffer[:self.buffer_size]
            self.buffer = self.buffer[self.buffer_size:]
            self.process_window(audio_chunk, ui_callback)

    def start(self, ui_callback=None):
        if self.running:
            return

        def callback(indata, frames, time_info, status):
            self.audio_callback(indata, frames, time_info, status, ui_callback)

        try:
            self.stream = sd.InputStream(
                device=self.device_index,
                channels=1,
                samplerate=self.sample_rate,
                callback=callback
            )
            self.stream.start()
            self.running = True
            logger.info(f"Started listening on device {self.device_index if self.device_index else 'default'}")
        except Exception as e:
            logger.error(f"Error starting audio stream: {e}")
            raise e

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.running = False
        logger.info("Stopped listening.")
