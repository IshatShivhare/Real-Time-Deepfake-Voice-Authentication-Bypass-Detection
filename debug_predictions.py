import torch
import numpy as np
import soundfile as sf
import os
from ensemble_detector import EnsembleDetector

def generate_synthetic_audio(filename, duration=4, sr=16000, type='noise'):
    """Generate synthetic audio for testing"""
    print(f"Generating {type} audio: {filename}")
    samples = int(duration * sr)
    
    if type == 'noise':
        audio = np.random.uniform(-0.1, 0.1, samples)
    elif type == 'silence':
        audio = np.zeros(samples)
    elif type == 'sine':
        t = np.linspace(0, duration, samples)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t) # 440Hz sine wave
        
    sf.write(filename, audio, sr)
    return filename

def test_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    detector = EnsembleDetector(device=device)
    detector.load_aasist()
    detector.load_vocoder()
    
    # Create test files
    os.makedirs('temp_test', exist_ok=True)
    files = {
        'noise': 'temp_test/test_noise.wav',
        'silence': 'temp_test/test_silence.wav',
        'sine': 'temp_test/test_sine.wav'
    }
    
    for type_name, path in files.items():
        if not os.path.exists(path):
            generate_synthetic_audio(path, type=type_name)
    
    print("\n" + "="*50)
    print("DEBUGGING RAW PREDICTIONS")
    print("="*50)
    
    for name, path in files.items():
        print(f"Processing {name}...", flush=True)
        print(f"\nAnalyzing: {name} ({path})")
        try:
            # We want to peek inside, so we'll do part of what predict_single does manually
            # or just interpret the details carefully
            pred, conf, details = detector.predict_single(path)
            
            print(f"Final Decision: {'FAKE' if pred==1 else 'REAL'}")
            print(f"Final Confidence: {conf:.4f}")
            
            print("Individual Models:")
            for model, p in details['individual_predictions'].items():
                c = details['individual_confidences'][model]
                prob = details.get('fake_probabilities', {}).get(model, 'N/A')
                print(f"  {model}: {'FAKE' if p==1 else 'REAL'} (Conf: {c:.4f}, FakeProb: {prob:.4f})")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_model()
