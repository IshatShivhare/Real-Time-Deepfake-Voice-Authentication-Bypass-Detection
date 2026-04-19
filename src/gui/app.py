import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sys
import threading
import queue
import os
import librosa
from pathlib import Path

# Ensure root is in path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.audio.capture import AudioCapturer
from src.utils.logger import get_logger
from src.utils.config_loader import get_config
from src.models.model_loader import load_all_models
from src.models.ensemble import Ensemble

logger = get_logger("GUI_TK")

class DeepfakeDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Deepfake Voice Detector")
        self.root.geometry("600x550")
        self.root.resizable(False, False)
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure("TLabel", font=("Helvetica", 12))
        self.style.configure("Header.TLabel", font=("Helvetica", 18, "bold"))
        self.style.configure("Result.TLabel", font=("Helvetica", 24, "bold"))
        self.style.configure("Confidence.TLabel", font=("Helvetica", 14))
        
        # Variables
        self.device_var = tk.StringVar()
        self.file_path_var = tk.StringVar()
        self.is_listening = False
        self.capturer = AudioCapturer(device_index=None)
        self.update_queue = queue.Queue()
        
        # Helper map for device selection
        self.device_map = {} # "Name": index
        
        self.config = get_config()
        self.ensemble = None

        self._build_ui()
        self._load_devices()
        
        # Start periodic UI update check
        self.root.after(100, self.check_queue)
        
        # Handle close correctly
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self._show_loading()
        threading.Thread(target=self._load_models_thread, daemon=True).start()

    def _show_loading(self):
        self.loading_win = tk.Toplevel(self.root)
        self.loading_win.title("Loading Models")
        self.loading_win.geometry("300x100")
        self.loading_win.resizable(False, False)
        # Center window
        self.loading_win.transient(self.root)
        self.loading_win.grab_set()
        
        ttk.Label(self.loading_win, text="Loading models... Please wait.", font=("Helvetica", 12)).pack(pady=20)
        self.loading_progress = ttk.Progressbar(self.loading_win, mode="indeterminate")
        self.loading_progress.pack(fill=tk.X, padx=20)
        self.loading_progress.start(10)
        self.root.update()

    def _load_models_thread(self):
        try:
            models_dict = load_all_models(self.config)
            ens_cfg = self.config.get('ensemble', {})
            w1 = ens_cfg.get('wav2vec2_weight', 0.5)
            w2 = ens_cfg.get('rawnet2_weight', 0.5)
            threshold = ens_cfg.get('threshold', 0.5)
            self.ensemble = Ensemble(models_dict, w1, w2, threshold)
            self.capturer.detector = self.ensemble
            self.update_queue.put(('loading_done', None))
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.update_queue.put(('error', f"Failed to load models: {e}"))

    def _build_ui(self):
        # Create Tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Live Analysis Tab
        self.live_tab = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(self.live_tab, text="🎙️ Live Analysis")
        self._build_live_tab(self.live_tab)
        
        # File Analysis Tab
        self.file_tab = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(self.file_tab, text="📂 File Analysis")
        self._build_file_tab(self.file_tab)
        
        # Status footer (shared)
        self.status_label = ttk.Label(self.root, text="Ready", font=("Helvetica", 9), foreground="gray")
        self.status_label.pack(side=tk.BOTTOM, anchor=tk.W, padx=10, pady=5)

    def _build_live_tab(self, parent):
        header = ttk.Label(parent, text="Real-time Detection", style="Header.TLabel", foreground="#333")
        header.pack(pady=(0, 20))
        
        dev_frame = ttk.LabelFrame(parent, text="Input Device", padding="10")
        dev_frame.pack(fill=tk.X, pady=10)
        
        self.device_combo = ttk.Combobox(dev_frame, textvariable=self.device_var, state="readonly")
        self.device_combo.pack(fill=tk.X)
        self.device_combo.bind("<<ComboboxSelected>>", self.on_device_change)
        
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(pady=20)
        
        self.start_btn = ttk.Button(btn_frame, text="▶ Start Listening", command=self.start_listening, state="disabled")
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹ Stop Listening", command=self.stop_listening, state="disabled")
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        res_frame = ttk.LabelFrame(parent, text="Analysis Result", padding="20")
        res_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.live_result_label = ttk.Label(res_frame, text="WAITING...", style="Result.TLabel", foreground="gray")
        self.live_result_label.pack(pady=10)
        
        self.live_confidence_label = ttk.Label(res_frame, text="Confidence: --%", style="Confidence.TLabel")
        self.live_confidence_label.pack(pady=5)
        
        self.live_wav2vec2_label = ttk.Label(res_frame, text="Wav2Vec2: --", font=("Helvetica", 10))
        self.live_wav2vec2_label.pack(pady=2)
        
        self.live_rawnet2_label = ttk.Label(res_frame, text="RawNet2: --", font=("Helvetica", 10))
        self.live_rawnet2_label.pack(pady=2)

    def _build_file_tab(self, parent):
        header = ttk.Label(parent, text="File Analysis", style="Header.TLabel", foreground="#333")
        header.pack(pady=(0, 20))
        
        file_frame = ttk.LabelFrame(parent, text="Select Audio File", padding="10")
        file_frame.pack(fill=tk.X, pady=10)
        
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, state="readonly")
        file_entry.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=(0, 5))
        
        browse_btn = ttk.Button(file_frame, text="Browse...", command=self.browse_file)
        browse_btn.pack(side=tk.LEFT)
        
        self.analyze_btn = ttk.Button(parent, text="🔍 Analyze File", command=self.analyze_file, state="disabled")
        self.analyze_btn.pack(pady=10)
        
        res_frame = ttk.LabelFrame(parent, text="Analysis Result", padding="20")
        res_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.file_result_label = ttk.Label(res_frame, text="--", style="Result.TLabel", foreground="gray")
        self.file_result_label.pack(pady=10)
        
        self.file_confidence_label = ttk.Label(res_frame, text="Confidence: --%", style="Confidence.TLabel")
        self.file_confidence_label.pack(pady=5)
        
        self.file_wav2vec2_label = ttk.Label(res_frame, text="Wav2Vec2: --", font=("Helvetica", 10))
        self.file_wav2vec2_label.pack(pady=2)
        
        self.file_rawnet2_label = ttk.Label(res_frame, text="RawNet2: --", font=("Helvetica", 10))
        self.file_rawnet2_label.pack(pady=2)

    def _load_devices(self):
        try:
            devices = AudioCapturer.get_input_devices()
            self.device_map = {name: idx for idx, name in devices}
            device_names = list(self.device_map.keys())
            
            self.device_combo['values'] = device_names
            if device_names:
                self.device_combo.current(0)
                
            self.status_label.config(text=f"Loaded {len(devices)} devices.")
        except Exception as e:
            logger.error(f"Error loading devices: {e}")
            messagebox.showerror("Error", f"Failed to load audio devices: {e}")

    def on_device_change(self, event=None):
        name = self.device_var.get()
        idx = self.device_map.get(name)
        if idx is not None:
            self.capturer.device_index = idx
            logger.info(f"Selected device: {name} (Index {idx})")

    def start_listening(self):
        if self.is_listening or not self.ensemble:
            return
            
        try:
            name = self.device_var.get()
            if not name:
                messagebox.showwarning("Warning", "Please select an input device.")
                return
                
            idx = self.device_map.get(name)
            self.capturer.device_index = idx 
            
            self.is_listening = True
            logger.info(f"Starting capture on {idx}...")
            
            self.capturer.start(ui_callback=self.audio_callback)
            
            self.start_btn.config(state="disabled")
            self.device_combo.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.live_result_label.config(text="LISTENING...", foreground="#FFA500") 
            self.status_label.config(text="Listening active...")
            
        except Exception as e:
            self.is_listening = False
            logger.error(f"Failed to start: {e}")
            messagebox.showerror("Error", f"Failed to start capture: {e}")

    def stop_listening(self):
        if not self.is_listening:
            return
            
        self.is_listening = False
        self.capturer.stop()
        
        self.start_btn.config(state="normal")
        self.device_combo.config(state="readonly")
        self.stop_btn.config(state="disabled")
        self.live_result_label.config(text="STOPPED", foreground="gray")
        self.status_label.config(text="Stopped.")

    def audio_callback(self, result_dict):
        self.update_queue.put(('live', result_dict))

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg"), ("All Files", "*.*")]
        )
        if filename:
            self.file_path_var.set(filename)
            if self.ensemble:
                self.analyze_btn.config(state="normal")
            self.file_result_label.config(text="READY", foreground="gray")
            self.file_confidence_label.config(text="Confidence: --%")

    def analyze_file(self):
        file_path = self.file_path_var.get()
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Error", "Invalid file path.")
            return

        self.file_result_label.config(text="ANALYZING...", foreground="#FFA500")
        self.status_label.config(text=f"Analyzing {os.path.basename(file_path)}...")
        self.root.update()

        threading.Thread(target=self._run_file_analysis, args=(file_path,), daemon=True).start()

    def _run_file_analysis(self, file_path):
        try:
            audio, sr = librosa.load(file_path, sr=None)
            result = self.ensemble.predict(audio, sr)
            self.update_queue.put(('file', result))
        except Exception as e:
            logger.error(f"Error analyzing file: {e}")
            self.update_queue.put(('error', str(e)))

    def check_queue(self):
        try:
            while True:
                msg_type, data = self.update_queue.get_nowait()
                
                if msg_type == 'loading_done':
                    self.loading_win.destroy()
                    self.start_btn.config(state="normal")
                    if self.file_path_var.get():
                        self.analyze_btn.config(state="normal")
                    self.status_label.config(text="Models loaded successfully.")
                elif msg_type == 'live':
                    self.update_live_display(data)
                elif msg_type == 'file':
                    self.update_file_display(data)
                elif msg_type == 'error':
                    messagebox.showerror("Error", data)
                    self.file_result_label.config(text="ERROR", foreground="red")
                    
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.check_queue)

    def update_live_display(self, result):
        verdict = result['verdict']
        confidence = result['confidence']
        w2v_score = result['wav2vec2_score']
        rn_score = result['rawnet2_score']
        
        color = "#FF0000" if verdict == "SPOOF" else "#008000"
        self.live_result_label.config(text=verdict, foreground=color)
        self.live_confidence_label.config(text=f"Confidence: {confidence:.2%}")
        self.live_wav2vec2_label.config(text=f"Wav2Vec2 P(spoof): {w2v_score:.2f}")
        self.live_rawnet2_label.config(text=f"RawNet2 P(spoof): {rn_score:.2f}")

    def update_file_display(self, result):
        verdict = result['verdict']
        confidence = result['confidence']
        w2v_score = result['wav2vec2_score']
        rn_score = result['rawnet2_score']
        
        color = "#FF0000" if verdict == "SPOOF" else "#008000"
        self.file_result_label.config(text=verdict, foreground=color)
        self.file_confidence_label.config(text=f"Confidence: {confidence:.2%}")
        self.file_wav2vec2_label.config(text=f"Wav2Vec2 P(spoof): {w2v_score:.2f}")
        self.file_rawnet2_label.config(text=f"RawNet2 P(spoof): {rn_score:.2f}")
        self.status_label.config(text="Analysis complete.")

    def on_close(self):
        if self.is_listening:
            self.stop_listening()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = DeepfakeDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
