"""Simple end-to-end test of deepfake detector"""
import sys

# Run the detector
print("Testing Deepfake Detector with sample audio file...")
print("=" * 70)

from ensemble_detector import DeepfakeDetector

# Initialize detector
detector = DeepfakeDetector(device='cpu')

# Load model
detector.load_model()

# Test with sample audio
audio_path = r"C:\Users\ASUS\.cache\kagglehub\datasets\mohammedabdeldayem\scenefake\versions\1\eval\fake\C_06335_0_A.wav"

print(f"\nAnalyzing: {audio_path}")
print("-" * 70)

try:
    prediction, confidence, details = detector.predict_single(audio_path)
    
    # Display results
    result_text = "FAKE (Deepfake)" if prediction == 1 else "REAL (Bonafide)"
    print(f"\nFinal Prediction: {result_text}")
    print(f"Confidence: {confidence:.2%}")
    print(f"\nProbabilities:")
    print(f"  Real: {details['real_probability']:.2%}")
    print(f"  Fake: {details['fake_probability']:.2%}")
    
    print("\n" + "=" * 70)
    print("[OK] Test completed successfully!")
    
except Exception as e:
    print(f"\n[ERR] Error during prediction: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
