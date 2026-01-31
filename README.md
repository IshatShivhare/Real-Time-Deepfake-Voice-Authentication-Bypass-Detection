# Deepfake Voice Detection System

A complete, end-to-end system for detecting deepfake audio, featuring a real-time GUI, comprehensive data processing pipeline, and an ensemble model architecture (RawNet Vocoder + Custom CNN-GRU).

## 🚀 Key Features

- **Ensemble Detection**: Combines a pretrained Vocoder-Artifacts model (RawNet) with a custom trainable CNN-GRU model for robust detection.
- **Data Processing Pipeline**: Full ETL pipeline to download, organize, feature-extract, and prepare the ASVspoof 2019 dataset.
- **Real-Time GUI**: Streamlit-based web interface for live audio analysis.
- **Modular Architecture**: Clean, scalable `src/` directory structure with centralized configuration.
- **Configurable**: All system parameters controlled via `config.yaml`.

## 📂 Project Structure

```
Deepfake Voice Detection/
├── config.yaml          # Central configuration for data, models, and app
├── main.py              # Main entry point (CLI)
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
├── weights/             # Model weights
│   ├── custom_model/
│   └── vocoder_model/
└── src/                 # Source Code
    ├── audio/           # Audio capture and preprocessing
    ├── data_processing/ # ETL Pipeline (Download, Extract, Prepare)
    ├── gui/             # Streamlit Web Application
    ├── models/          # Model architectures
    │   ├── custom/      # CNN+GRU Model
    │   ├── vocoder/     # RawNet Model
    │   └── ensemble.py  # Ensemble Logic
    └── utils/           # Shared utilities (Logger, Config Loader)
```

## 🛠️ Installation

1.  **Clone the repository** (or download source).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For GPU support, ensure you have the correct CUDA-enabled version of PyTorch and TensorFlow.*

## 🚦 Usage

All interactions are handled through the `main.py` entry point.

### 1. Web GUI (Real-Time Detection)
Launch the Streamlit interface to record or upload audio for analysis.
```bash
python main.py gui
```

### 2. Single File Detection
Analyze a specific audio file from the command line.
```bash
python main.py detect --audio "path/to/audio.wav"
```

### 3. Data Pipeline (ETL)
Run the full data processing pipeline to prepare the ASVspoof 2019 dataset.
Configuration for dataset paths is in `config.yaml`.

```bash
# Run all steps (Organize -> Extract -> Prepare)
python main.py pipeline

# Run specific steps
python main.py pipeline --steps organize,extract
```

### 4. Model Training
Train the custom CNN-GRU model using the prepared dataset.
The training uses a **Progressive Strategy**: starts with a simple model and increases complexity only if needed.

```bash
python main.py train
```

## ⚙️ Configuration

The `config.yaml` file controls all aspects of the system:

*   **`dataset`**: Paths and sample limits for ASVspoof.
*   **`audio`**: Sample rate, chunk duration, and FFT settings.
*   **`features`**: Toggles for MFCC, Spectral, Temporal, and Chroma features.
*   **`model_custom`**: Architecture (CNN-GRU/LSTM), hyperparameters, and training settings.
*   **`model_vocoder`**: Settings for the RawNet model.
*   **`ensemble`**: Weights for combining model predictions.

## 🧠 Model Details

### Ensemble Approach
The system uses a weighted average of two models:
1.  **Vocoder Model (RawNet)**: Detects artifacts left by neural vocoders (e.g., WaveNet, HiFi-GAN). Operates on raw waveforms.
2.  **Custom Model (CNN-GRU)**: Learns temporal and spectral patterns from extracted features (MFCCs, etc.).

### Progressive Training
To avoid overfitting and waste resources, the training script checks validation accuracy:
1.  **Stage 1**: Trains a Simple CNN. If baseline threshold is met, it stops.
2.  **Stage 2**: If needed, trains a CNN-GRU.
3.  **Stage 3**: If needed, trains a CNN-BiLSTM (most complex).

## 📄 License

MIT License.

## 🙏 Acknowledgments

-   **ASVspoof 2019**: Dataset and protocols.
-   **RawNet**: Base architecture for vocoder artifact detection.
