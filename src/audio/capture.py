import sounddevice as sd
import numpy as np
import soundfile as sf
import librosa
import torch
import tempfile
import os
import queue
import threading
from src.utils.logger import get_logger
from src.models.ensemble import EnsembleDetector

logger = get_logger("AudioCapture")

class AudioCapturer:
    @staticmethod
    def get_input_devices():
        try:
            # Query devices with MME host API to avoid WDM-KS issues
            devices = sd.query_devices()
            input_devices = []
            
            # Find MME host API index
            mme_index = None
            for i, host in enumerate(sd.query_hostapis()):
                if 'MME' in host['name']:
                    mme_index = i
                    break
            
            for i, dev in enumerate(devices):
                # Filter for MME devices if found, otherwise include all input devices
                if dev['max_input_channels'] > 0:
                    if mme_index is not None and dev['hostapi'] != mme_index:
                        continue
                    input_devices.append((i, dev['name']))
            return input_devices
        except Exception as e:
            logger.error(f"Error listing devices: {e}")
            return []

    def __init__(self, device_index=None, sample_rate=48000, buffer_duration=1.0):
        self.device_index = device_index
        self.sample_rate = sample_rate # Capture rate
        self.buffer_size = int(sample_rate * buffer_duration)
        self.stream = None
        self.running = False
        
        # Queue for audio chunks to avoid blocking callback
        self.audio_queue = queue.Queue()
        self.worker_thread = None
        
        self.detector = None
        self._init_detector()
        
    def _init_detector(self):
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.detector = EnsembleDetector(device=device)
            logger.info("Detector initialized in capture module.")
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            
    def process_window(self, audio_48k, ui_callback=None):
        try:
            audio_np = np.array(audio_48k)
            
            # Check amplitude to verify we are actually capturing sound
            max_amp = np.max(np.abs(audio_np))
            if max_amp < 0.001:
                # logger.debug("Silence detected")
                pass

            # Resample to 16k for model
            audio_16k = librosa.resample(y=audio_np, orig_sr=self.sample_rate, target_sr=16000)
            
            # Use static filename like reference to avoid Windows tempfile permission issues
            temp_path = "temp_live.wav"
            sf.write(temp_path, audio_16k, 16000)
            
            try:
                prediction, confidence, details = self.detector.predict_single(temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            result_text = "FAKE" if prediction == 1 else "REAL"
            
            # Print to terminal like reference script
            color = "\033[91m" if prediction == 1 else "\033[92m" 
            reset = "\033[0m"
            print(f"\n{color}>> [Audio 1s | Amp: {max_amp:.4f}] Prediction: {result_text} | Confidence: {confidence:.2%}{reset}")
            
            logger.info(f"Prediction: {result_text} | Confidence: {confidence:.2%}")
            
            if ui_callback:
                ui_callback((result_text, confidence))
                
        except Exception as e:
            logger.error(f"Error processing window: {e}")

    def set_callback(self, ui_callback):
        """Update the UI callback"""
        self.ui_callback = ui_callback

    def _processing_worker(self, ctx=None):
        """Worker thread to process audio chunks"""
        logger.info("Processing worker started")
        
        # Attach streamlit context to thread
        if ctx:
            try:
                from streamlit.runtime.scriptrunner import add_script_run_ctx
                add_script_run_ctx(ctx=ctx)
            except ImportError:
                try:
                    # Fallback for older steamlit versions
                    from streamlit.scriptrunner import add_script_run_ctx
                    add_script_run_ctx(ctx=ctx)
                except Exception as e:
                    logger.warning(f"Could not add script run context: {e}")

        local_buffer = []
        
        while self.running:
            try:
                # Get data with timeout to allow checking self.running
                data_in = self.audio_queue.get(timeout=0.5)
                # data_in is numpy array (frames, channels)
                # Flatten and convert to list for buffer extension
                data_flat = data_in.flatten().tolist()
                local_buffer.extend(data_flat)
                
                if len(local_buffer) >= self.buffer_size:
                    chunk = local_buffer[:self.buffer_size]
                    local_buffer = local_buffer[self.buffer_size:] # overlap logic could be added here
                    
                    # Use self.ui_callback which can be updated dynamically
                    if hasattr(self, 'ui_callback') and self.ui_callback:
                        self.process_window(chunk, self.ui_callback)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")
                import traceback
                logger.error(traceback.format_exc())

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio status: {status}")
        
        # Debug: Check raw amplitude directly in callback
        # amp = np.max(np.abs(indata))
        # if amp > 0.01:
        #    print(f"RAW CALLBACK AMP: {amp}")

        # Non-blocking enqueue of numpy array copy (faster than tolist)
        try:
            self.audio_queue.put(indata.copy(), block=False)
        except queue.Full:
            logger.warning("Audio queue full, dropping frame")

    def start(self, ui_callback=None):
        if self.running:
            # If already running, just update the callback
            if ui_callback:
                self.ui_callback = ui_callback
            return

        self.running = True
        self.ui_callback = ui_callback
        
        # Debug: Print device info
        try:
            device_info = sd.query_devices(self.device_index)
            print(f"\n[DEBUG] Opening Device Index: {self.device_index}")
            print(f"[DEBUG] Device Name: {device_info['name']}")
            print(f"[DEBUG] Device Channels: {device_info['max_input_channels']}")
            print(f"[DEBUG] Host API: {device_info['hostapi']}\n")
        except Exception as e:
            print(f"[DEBUG] Could not query device info: {e}")

        # Capture current context to pass to worker
        ctx = None
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            ctx = get_script_run_ctx()
        except ImportError:
            try:
                from streamlit.scriptrunner import get_script_run_ctx
                ctx = get_script_run_ctx()
            except:
                pass
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._processing_worker, args=(ctx,), daemon=True)
        self.worker_thread.start()

        def callback(indata, frames, time_info, status):
            self.audio_callback(indata, frames, time_info, status)

        try:
            self.stream = sd.InputStream(
                device=self.device_index,
                channels=1,
                samplerate=self.sample_rate,
                callback=callback
                # Removed explicit blocksize to let backend decide optimal size
            )
            self.stream.start()
            logger.info(f"Started listening on device {self.device_index if self.device_index else 'default'}")
        except Exception as e:
            self.running = False
            logger.error(f"Error starting audio stream: {e}")
            raise e

    def stop(self):
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
            self.worker_thread = None
            
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # clear queue
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
            
        logger.info("Stopped listening.")
