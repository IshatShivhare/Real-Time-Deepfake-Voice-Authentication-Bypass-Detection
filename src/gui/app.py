import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sys
import threading
import queue
import os
from pathlib import Path

# Ensure root is in path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.audio.capture import AudioCapturer
from src.utils.logger import get_logger

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

        self._build_ui()
        self._load_devices()
        
        # Start periodic UI update check
        self.root.after(100, self.check_queue)
        
        # Handle close correctly
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

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
        # Header
        header = ttk.Label(parent, text="Real-time Detection", style="Header.TLabel", foreground="#333")
        header.pack(pady=(0, 20))
        
        # Device Selection
        dev_frame = ttk.LabelFrame(parent, text="Input Device", padding="10")
        dev_frame.pack(fill=tk.X, pady=10)
        
        self.device_combo = ttk.Combobox(dev_frame, textvariable=self.device_var, state="readonly")
        self.device_combo.pack(fill=tk.X)
        self.device_combo.bind("<<ComboboxSelected>>", self.on_device_change)
        
        # Controls
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(pady=20)
        
        self.start_btn = ttk.Button(btn_frame, text="▶ Start Listening", command=self.start_listening)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹ Stop Listening", command=self.stop_listening, state="disabled")
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Results Display
        res_frame = ttk.LabelFrame(parent, text="Analysis Result", padding="20")
        res_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.live_result_label = ttk.Label(res_frame, text="WAITING...", style="Result.TLabel", foreground="gray")
        self.live_result_label.pack(pady=10)
        
        self.live_confidence_label = ttk.Label(res_frame, text="Confidence: --%", style="Confidence.TLabel")
        self.live_confidence_label.pack(pady=5)

    def _build_file_tab(self, parent):
        # Header
        header = ttk.Label(parent, text="File Analysis", style="Header.TLabel", foreground="#333")
        header.pack(pady=(0, 20))
        
        # File Selection
        file_frame = ttk.LabelFrame(parent, text="Select Audio File", padding="10")
        file_frame.pack(fill=tk.X, pady=10)
        
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, state="readonly")
        file_entry.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=(0, 5))
        
        browse_btn = ttk.Button(file_frame, text="Browse...", command=self.browse_file)
        browse_btn.pack(side=tk.LEFT)
        
        # Analyze Button
        self.analyze_btn = ttk.Button(parent, text="🔍 Analyze File", command=self.analyze_file, state="disabled")
        self.analyze_btn.pack(pady=10)
        
        # Results Display
        res_frame = ttk.LabelFrame(parent, text="Analysis Result", padding="20")
        res_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.file_result_label = ttk.Label(res_frame, text="--", style="Result.TLabel", foreground="gray")
        self.file_result_label.pack(pady=10)
        
        self.file_confidence_label = ttk.Label(res_frame, text="Confidence: --%", style="Confidence.TLabel")
        self.file_confidence_label.pack(pady=5)
        
        self.file_details_label = ttk.Label(res_frame, text="", font=("Courier", 8))
        self.file_details_label.pack(pady=5)

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

    # --- Live Analysis Methods ---
    def start_listening(self):
        if self.is_listening:
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

    def audio_callback(self, result):
        self.update_queue.put(('live', result))

    # --- File Analysis Methods ---
    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg"), ("All Files", "*.*")]
        )
        if filename:
            self.file_path_var.set(filename)
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

        # Run in thread to avoid freezing UI
        threading.Thread(target=self._run_file_analysis, args=(file_path,), daemon=True).start()

    def _run_file_analysis(self, file_path):
        try:
            # Ensure detector is loaded
            if not self.capturer.detector:
                 self.capturer._init_detector()

            prediction, confidence, details = self.capturer.detector.predict_single(file_path)
            result_text = "FAKE" if prediction == 1 else "REAL"
            
            self.update_queue.put(('file', (result_text, confidence, details)))
            
        except Exception as e:
            logger.error(f"Error analyzing file: {e}")
            self.update_queue.put(('error', str(e)))

    # --- Shared Core Methods ---
    def check_queue(self):
        try:
            while True:
                msg_type, data = self.update_queue.get_nowait()
                
                if msg_type == 'live':
                    text, conf = data
                    self.update_live_display(text, conf)
                elif msg_type == 'file':
                    text, conf, details = data
                    self.update_file_display(text, conf, details)
                elif msg_type == 'error':
                    messagebox.showerror("Analysis Error", data)
                    self.file_result_label.config(text="ERROR", foreground="red")
                    
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.check_queue)

    def update_live_display(self, text, confidence):
        color = "#FF0000" if text == "FAKE" else "#008000"
        self.live_result_label.config(text=text, foreground=color)
        self.live_confidence_label.config(text=f"Confidence: {confidence:.2%}")

    def update_file_display(self, text, confidence, details):
        color = "#FF0000" if text == "FAKE" else "#008000"
        self.file_result_label.config(text=text, foreground=color)
        self.file_confidence_label.config(text=f"Confidence: {confidence:.2%}")
        self.status_label.config(text="Analysis complete.")
        # Optional: show details in tooltip or simpler text
        # self.file_details_label.config(text=str(details)) 

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
