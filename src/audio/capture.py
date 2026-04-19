import sounddevice as sd
import numpy as np
import queue
import threading
from src.utils.logger import get_logger

logger = get_logger("AudioCapture")

class AudioCapturer:
    @staticmethod
    def get_input_devices():
        try:
            devices = sd.query_devices()
            input_devices = []
            
            mme_index = None
            for i, host in enumerate(sd.query_hostapis()):
                if 'MME' in host['name']:
                    mme_index = i
                    break
            
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    if mme_index is not None and dev['hostapi'] != mme_index:
                        continue
                    input_devices.append((i, dev['name']))
            return input_devices
        except Exception as e:
            logger.error(f"Error listing devices: {e}")
            return []

    def __init__(self, device_index=None, sample_rate=16000, buffer_duration=4.0):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.buffer_size = int(sample_rate * buffer_duration)
        self.stream = None
        self.running = False
        
        self.audio_queue = queue.Queue()
        self.worker_thread = None
        
        self.detector = None # Will be set by GUI
        
    def process_window(self, audio_chunk, ui_callback=None):
        try:
            audio_np = np.array(audio_chunk)
            
            max_amp = np.max(np.abs(audio_np))
            if max_amp < 0.001:
                pass
            
            if self.detector:
                result_dict = self.detector.predict(audio_np, self.sample_rate)
                
                verdict = result_dict['verdict']
                confidence = result_dict['confidence']
                
                color = "\033[91m" if verdict == "SPOOF" else "\033[92m" 
                reset = "\033[0m"
                print(f"\n{color}>> [Audio 4.0s | Amp: {max_amp:.4f}] Prediction: {verdict} | Confidence: {confidence:.2%}{reset}")
                logger.info(f"Prediction: {verdict} | Confidence: {confidence:.2%}")
                
                if ui_callback:
                    ui_callback(result_dict)
                    
        except Exception as e:
            logger.error(f"Error processing window: {e}")

    def set_callback(self, ui_callback):
        self.ui_callback = ui_callback

    def _processing_worker(self):
        logger.info("Processing worker started")
        local_buffer = []
        
        while self.running:
            try:
                data_in = self.audio_queue.get(timeout=0.5)
                data_flat = data_in.flatten().tolist()
                local_buffer.extend(data_flat)
                
                if len(local_buffer) >= self.buffer_size:
                    chunk = local_buffer[:self.buffer_size]
                    local_buffer = local_buffer[self.buffer_size:]
                    
                    if hasattr(self, 'ui_callback') and self.ui_callback:
                        self.process_window(chunk, self.ui_callback)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio status: {status}")
        try:
            self.audio_queue.put(indata.copy(), block=False)
        except queue.Full:
            logger.warning("Audio queue full, dropping frame")

    def start(self, ui_callback=None):
        if self.running:
            if ui_callback:
                self.ui_callback = ui_callback
            return

        self.running = True
        self.ui_callback = ui_callback
        
        self.worker_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.worker_thread.start()

        def callback(indata, frames, time_info, status):
            self.audio_callback(indata, frames, time_info, status)

        try:
            self.stream = sd.InputStream(
                device=self.device_index,
                channels=1,
                samplerate=self.sample_rate,
                callback=callback
            )
            self.stream.start()
            logger.info(f"Started listening on device {self.device_index}")
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
        
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
            
        logger.info("Stopped listening.")
