"""
Quick test script to verify the ensemble detector setup
"""
import os
import sys

def check_file(path, required=True):
    """Check if file exists"""
    exists = os.path.exists(path)
    status = "✓" if exists else ("✗" if required else "⚠")
    req_text = "(required)" if required else "(optional)"
    print(f"{status} {path} {req_text}")
    return exists

def main():
    print("=" * 60)
    print("Ensemble Deepfake Detector - Setup Verification")
    print("=" * 60)
    
    print("\n📁 Checking Project Structure...")
    print("-" * 60)
    
    # Check directories
    dirs = ['models', 'weights', 'utils', 'examples']
    for d in dirs:
        check_file(d, required=True)
    
    print("\n📄 Checking Core Files...")
    print("-" * 60)
    
    files = [
        ('ensemble_detector.py', True),
        ('requirements.txt', True),
        ('README.md', True),
        ('models/__init__.py', True),
        ('models/aasist_model.py', True),
        ('models/vocoder_model.py', True),
        ('models/model_config_RawNet.yaml', True),
        ('utils/__init__.py', True),
        ('utils/audio_utils.py', True),
    ]
    
    for file, required in files:
        check_file(file, required)
    
    print("\n🎯 Checking Model Weights...")
    print("-" * 60)
    
    aasist_exists = check_file('weights/AASIST-L.pth', required=True)
    vocoder_exists = check_file('weights/vocoder_model.pth', required=False)
    
    print("\n📊 Summary...")
    print("-" * 60)
    
    if aasist_exists:
        print("✓ AASIST-L model ready to use")
    else:
        print("✗ AASIST-L weights missing!")
    
    if vocoder_exists:
        print("✓ Vocoder model ready to use")
        print("✓ Full ensemble mode available")
    else:
        print("⚠ Vocoder weights not found")
        print("  Download from: https://drive.google.com/file/d/15qOi26czvZddIbKP_SOR8SLQFZK8cf8E/view")
        print("  Save to: weights/vocoder_model.pth")
        print("  (System will work with AASIST-L only)")
    
    print("\n🚀 Next Steps...")
    print("-" * 60)
    print("1. Install dependencies: pip install -r requirements.txt")
    
    if not vocoder_exists:
        print("2. (Optional) Download Vocoder weights for best results")
        print("3. Place test audio in examples/ folder")
        print("4. Run: python ensemble_detector.py --audio examples/your_audio.wav")
    else:
        print("2. Place test audio in examples/ folder")
        print("3. Run: python ensemble_detector.py --audio examples/your_audio.wav")
    
    print("\n" + "=" * 60)
    
    # Try importing dependencies
    print("\n🔍 Checking Python Dependencies...")
    print("-" * 60)
    
    deps = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'soundfile': 'SoundFile',
        'librosa': 'Librosa',
        'yaml': 'PyYAML'
    }
    
    missing = []
    for module, name in deps.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} (not installed)")
            missing.append(name)
    
    if missing:
        print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
        print("  Run: pip install -r requirements.txt")
    else:
        print("\n✓ All dependencies installed!")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()
