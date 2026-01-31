# Deepfake Audio Detector

A deepfake audio detection system using the **Vocoder-Artifacts (RawNet)** model for accurate TTS and voice synthesis detection.

## 🎯 Features

- **Vocoder-Artifacts Detection**: Specialized in detecting neural vocoder artifacts from TTS systems
- **Pretrained Weights**: Ready to use with included model weights
- **Silence Removal**: Automatic preprocessing to focus on speech content
- **Easy to Use**: Simple command-line interface
- **High Accuracy**: Excellent performance on synthetic voice detection

## 📁 Project Structure

```
ensemble-deepfake-detector/
├── models/
│   ├── vocoder_model.py         # Vocoder-Artifacts (RawNet) architecture
│   └── model_config_RawNet.yaml # Model configuration
├── weights/
│   └── librifake_pretrained_lambda0.5_epoch_25.pth  # Pretrained weights
├── utils/
│   └── audio_utils.py          # Audio preprocessing utilities
├── examples/
│   └── (place test audio files here)
├── ensemble_detector.py        # Main inference script
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Detection

```bash
python ensemble_detector.py --audio path/to/your/audio.wav
```

Or with full path:

```bash
python ensemble_detector.py --audio "C:\path\to\audio.mp3"
```

## 💡 Usage Examples

### Basic Detection

```bash
python ensemble_detector.py --audio examples/test_audio.wav
```

### Specify Device

```bash
# Use GPU
python ensemble_detector.py --audio test.wav --device cuda

# Use CPU
python ensemble_detector.py --audio test.wav --device cpu
```

## 🧠 How It Works

### Vocoder-Artifacts Model
- **Purpose**: Detects neural vocoder artifacts in synthetic speech
- **Strength**: Highly effective against TTS and voice cloning systems
- **Architecture**: Modified RawNet2 with SincConv layers
- **Training**: Pretrained on LibriFake dataset
- **Performance**: Excellent detection of modern deepfake audio

### Preprocessing Pipeline

1. **Audio Loading**: Supports WAV, MP3, FLAC, and other formats
2. **Resampling**: Converts to 16kHz sample rate
3. **Silence Removal**: Removes silent segments using Voice Activity Detection
4. **Normalization**: Amplitude normalization to [-1, 1]
5. **Padding/Trimming**: Fixed length input (64,600 samples ≈ 4 seconds)

## 📊 Example Output

```
Using device: cpu
Loading Vocoder-Artifacts (RawNet) model...
[OK] Model weights loaded from weights/librifake_pretrained_lambda0.5_epoch_25.pth

Analyzing: C:\Users\...\audio.mp3
--------------------------------------------------

Final Prediction: FAKE (Deepfake)
Confidence: 99.99%

Probabilities:
  Real: 0.01%
  Fake: 99.99%
```

## 🔧 Using in Your Code

```python
from ensemble_detector import DeepfakeDetector

# Initialize
detector = DeepfakeDetector(device='cuda')
detector.load_model()

# Predict single file
prediction, confidence, details = detector.predict_single('audio.wav')

if prediction == 1:
    print(f"FAKE audio detected with {confidence:.2%} confidence")
    print(f"Fake probability: {details['fake_probability']:.2%}")
else:
    print(f"REAL audio with {confidence:.2%} confidence")
    print(f"Real probability: {details['real_probability']:.2%}")

# Predict multiple files
audio_files = ['audio1.wav', 'audio2.wav', 'audio3.wav']
results = detector.predict_batch(audio_files)

for i, (pred, conf, details) in enumerate(results):
    result_text = "FAKE" if pred == 1 else "REAL"
    print(f"{audio_files[i]}: {result_text} ({conf:.2%})")
```

## 📝 Technical Details

### Model Architecture

The Vocoder-Artifacts model uses:
- **SincConv**: Learnable mel-scale filterbank convolution
- **Residual Blocks**: Deep feature extraction
- **Attention Mechanism**: Focus on discriminative features
- **GRU Layers**: Temporal modeling
- **Binary Classification**: Real vs Fake output

### Supported Audio Formats

- WAV (recommended)
- MP3
- FLAC
- MPEG
- Any format supported by librosa/soundfile

### Performance Characteristics

- **Input**: 16kHz mono audio, ~4 seconds
- **Processing Time**: ~0.5-2 seconds per file (CPU)
- **Memory**: ~500MB (model loaded)
- **Accuracy**: High precision on TTS-generated audio

## 🎓 Model Reference

### Vocoder-Artifacts Detection
Based on research in neural vocoder artifact detection for synthetic speech identification.

**Key Insight**: Modern TTS systems use neural vocoders (WaveNet, HiFi-GAN, etc.) that leave subtle artifacts in the generated audio. This model is trained to detect these artifacts.

## 📄 License

MIT License - Free to use for research and applications

## 🆘 Troubleshooting

### "Model not loaded with weights!"
- Make sure `weights/librifake_pretrained_lambda0.5_epoch_25.pth` exists
- Check file permissions

### CUDA out of memory
- Use `--device cpu` flag
- Process shorter audio clips

### Audio format not supported
- Convert to WAV format using: `ffmpeg -i input.mp3 output.wav`
- Ensure audio is not corrupted

### UnicodeEncodeError on Windows
- This has been fixed in the latest version
- If you still see it, ensure you're using Python 3.7+

### Always predicting "Real"
- Ensure the audio contains speech (silence is often classified as real)
- Check that the audio is not corrupted
- Try with a known synthetic audio sample

## 🚀 Future Improvements

- [ ] Add batch processing with progress bar
- [ ] Support for streaming audio
- [ ] Web interface for easy testing
- [ ] Additional preprocessing options
- [ ] Model quantization for faster inference

## 🙏 Acknowledgments

- RawNet architecture from anti-spoofing research
- LibriFake dataset for pretraining
- ASVspoof challenge for evaluation protocols
