# Real-Time Deepfake Voice Detection System

A production-ready system for detecting AI-generated (deepfake) voices in real-time, designed to protect voice authentication systems from sophisticated audio synthesis attacks.

---

## 🎯 The Problem: Deepfake Voice Threat

### What are Deepfake Voices?

Deepfake voices are AI-generated synthetic speech that mimics real human voices with alarming accuracy. Modern neural vocoders (WaveNet, HiFi-GAN, Tacotron) can clone anyone's voice from just seconds of audio.

### Why This Matters

**Voice authentication systems are under attack:**

- 🏦 **Banking & Finance**: Voice biometric authentication can be bypassed.
- 📱 **VoIP Services**: WhatsApp, Telegram, Zoom calls can be spoofed.
- 🎭 **Identity Fraud**: Attackers impersonate executives or family members.

### The Challenge

**Real-time systems must:**
- ✅ Operate on short audio samples (3-10 seconds).
- ✅ Handle network noise and codec artifacts.
- ✅ Distinguish synthetic voices from natural variations.
- ✅ Detect modern neural vocoder artifacts with low latency.

---

## 💡 Our Solution

### Three-Layer Detection Strategy

#### 1. **Vocoder Artifact Detection (RawNet2)**
- Targets artifacts left by neural vocoders during audio synthesis.
- Utilizes the RawNet2 architecture trained on the ASVspoof 2019 LA dataset to analyze raw waveform anomalies.

#### 2. **Acoustic Feature Analysis (Wav2Vec2)**
- Uses state-of-the-art transformer-based acoustic models (Wav2Vec2).
- Extracts complex acoustic representations directly from audio to learn discriminative patterns of synthetic voices.
- Trained on the ASVspoof 2019 LA dataset on Kaggle.

#### 3. **Ensemble Intelligence**
- **Inference**: Combines outputs using a weighted average strategy.
- **Configuration**: Adjustable weights (default: Custom Model 0.6, Vocoder Model 0.4) to balance detection sensitivity.

### Key Technical Innovations

**🎯 Real-Time Pipeline**
```
Audio Stream → Buffer (3s) → Feature Extraction → 
Dual Model Inference → Ensemble Decision → Alert/Allow
```

**🛡️ Robustness Features**
- **Data Augmentation**: Includes noise injection, time stretching, and codec simulation (Opus) to mimic real VoIP conditions.
- **Class Imbalance Handling**: Weighted training and sampling strategies for imbalanced datasets.

---

## 🚀 Key Features

- **✨ Real-Time Detection**: Designed for low-latency inference suitable for live scenarios.
- **🔊 Audio Capture**: Interfaces with system audio (supports virtual audio cables).
- **🖥️ User-Friendly GUI**: Python/Tkinter-based desktop interface for live monitoring and file analysis.
- **📊 Comprehensive Pipeline**: End-to-end processing: Download -> Organize -> Extract -> Train.
- **⚙️ Fully Configurable**: `config.yaml` controls datasets, model architectures, and thresholds.
- **🧩 Modular Design**: Clean `src/` directory structure.

---

## 📂 Project Structure
```
Deepfake Voice Detection/
├── config.yaml              # Central configuration
├── main.py                  # CLI entry point
├── requirements.txt         # Dependencies
├── README.md
│
├── data/                    # Dataset storage
├── weights/                 # Model weights
│   ├── custom_model/        # Wav2Vec2 weights
│   └── rawnet2_...pth       # Pretrained vocoder weights
│
└── src/
    ├── audio/               # Audio capture & processing
    ├── data_processing/     # ETL Pipeline
    ├── models/              # Model architectures
    │   ├── custom/          # Wav2Vec2 Notebooks & Implementation
    │   ├── vocoder/         # RawNet2 Notebooks & Implementation
    │   └── ensemble.py      # Ensemble logic
    ├── gui/                 # Desktop Application (app.py)
    └── utils/               # Shared utilities
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (Recommended for training)
- Virtual Audio Cable (e.g., VB-Cable) for VoIP monitoring.

### Setup

1. **Clone the repository**
2. **Install dependencies**
```bash
pip install -r requirements.txt
```
3. **Download Pretrained Weights**: Pull the latest model weights from Hugging Face or ensure they are present in the `weights/` directory.

---

## 🚦 Usage

### 1. 🎨 Desktop GUI (Live Detection)

Launch the Tkinter-based real-time detection interface:
```bash
python src/gui/app.py
```
*Tip: To monitor VoIP calls (Zoom/WhatsApp), route audio through a Virtual Audio Cable and select it as the input device in the GUI.*

### 2. 🔍 Single File Detection (CLI)

Analyze a specific audio file:
```bash
python main.py detect --audio "path/to/suspicious_call.wav"
```

### 3. 📊 Data Pipeline (ETL)

Prepare your dataset (configured in `config.yaml`). The pipeline supports organizing, extracting features, and preparing final data.

```bash
# Run complete pipeline
python main.py pipeline

# Run specific steps
python main.py pipeline --steps organize,extract,prepare
```
*Note: Feature analysis can be run separately via `src/data_processing/feature_analysis.py` if needed.*

### 4. 🧠 Model Training (Kaggle & Hugging Face)

Model training has been migrated to Kaggle to leverage cloud GPU resources. The trained models are pushed to Hugging Face for seamless integration.

To train or fine-tune the models, run the provided Jupyter Notebooks on Kaggle:
- **Custom Acoustic Model (Wav2Vec2)**: `src/models/custom/asvspoof-la-2019-on-wav2vec2.ipynb`
- **Vocoder Detector (RawNet2)**: `src/models/vocoder/rawnet2-on-asvspoof-2019-la.ipynb`

Once trained, models can be pulled from Hugging Face or placed manually in the `weights/` directory for inference.

---

## ⚙️ Configuration Guide

The `config.yaml` file controls system behavior:

### Dataset Configuration
```yaml
dataset:
  name: "SceneFake"   # or "ASVspoof"
  time_limit: null    # Seconds to load
  # ...
```

### Audio Processing
```yaml
audio:
  sample_rate: 16000
  chunk_duration: 3.0
```

### Feature Extraction
```yaml
features:
  n_mfcc: 40
  extract_spectral: true
  extract_temporal: true
  extract_chroma: true
```

### Model Architecture (`model_custom`)
```yaml
model_custom:
  architecture: "simple_cnn" # Options: simple_cnn, cnn_gru, cnn_lstm
  conv_filters: [64, 128]
  rnn_units: 128
  # ...
```

### Ensemble Configuration
```yaml
ensemble:
  weights:
    custom: 0.6    # Weight for Custom Model
    vocoder: 0.4   # Weight for RawNet Model
```

---

## 🧠 Technical Details

### Supported Models

#### Custom Acoustic Model (Wav2Vec2 / Transformers)
- Fine-tuned **Wav2Vec2** model for robust feature extraction and deepfake detection.
- Trained on the ASVspoof 2019 LA dataset using Kaggle.

#### Vocoder Detector (RawNet2 / PyTorch)
- Uses raw waveform analysis to detect synthesis artifacts.
- Trained on the ASVspoof 2019 LA dataset using Kaggle.

### Pipeline
1. **Organize**: Structured into train/dev/eval splits.
2. **Feature Extraction**: Extracts 48+ dimensions of acoustic features.
3. **Preparation**: Normalization, class balancing (if configured), and formatting for training.

---

## 📄 License

MIT License.

---

## 🔒 Security Notice

This system is a **detection tool**, not a prevention mechanism. It should be used as part of a layered security approach. No deepfake detector is 100% accurate.
