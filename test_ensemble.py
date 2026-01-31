"""Simple end-to-end test of ensemble detector"""
import sys

# Run the ensemble detector
print("Testing Ensemble Detector with sample audio file...")
print("=" * 70)

from ensemble_detector import EnsembleDetector

# Initialize detector
detector = EnsembleDetector(device='cpu')

# Load models
detector.load_aasist()
detector.load_vocoder()

# Test with sample audio
audio_path = r"C:\Users\ASUS\.cache\kagglehub\datasets\mohammedabdeldayem\scenefake\versions\1\eval\fake\C_06335_0_A.wav"

print(f"\nAnalyzing: {audio_path}")
print("-" * 70)

try:
    prediction, confidence, details = detector.predict_single(audio_path, method='weighted_average')
    
    # Display results
    result_text = "FAKE (Deepfake)" if prediction == 1 else "REAL (Bonafide)"
    print(f"\n🎯 Final Prediction: {result_text}")
    print(f"📊 Confidence: {confidence:.2%}")
    print(f"\n📋 Individual Model Results:")
    
    for model_name, pred in details['individual_predictions'].items():
        conf = details['individual_confidences'][model_name]
        pred_text = "FAKE" if pred == 1 else "REAL"
        print(f"  • {model_name.upper()}: {pred_text} ({conf:.2%} confidence)")
    
    print(f"\n⚙️  Ensemble Method: {details['ensemble_method']}")
    print("\n" + "=" * 70)
    print("✓ Test completed successfully!")
    
except Exception as e:
    print(f"\n✗ Error during prediction: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
