#!/usr/bin/env python
"""
Deepfake Voice Detection System
Main Entry Point
"""
import argparse
import sys
import os
import librosa
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.utils.logger import setup_logger
from src.utils.config_loader import load_config, get_config
from src.models.model_loader import load_all_models
from src.models.ensemble import Ensemble

logger = setup_logger("Main")

def run_gui():
    """Run the Tkinter GUI"""
    print("Launching GUI...")
    app_path = Path("src/gui/app.py").resolve()
    os.system(f"python \"{app_path}\"")

def run_detection(args):
    """Run inference on a single file"""
    if not args.audio:
        print("Error: --audio argument required for detection")
        return
        
    try:
        load_config("config.yaml")
    except Exception as e:
        logger.warning(f"Could not load config.yaml: {e}")
        
    config = get_config()
    
    print("Loading models... This may take a moment if weights need to be downloaded.")
    try:
        models_dict = load_all_models(config)
    except Exception as e:
        print(f"Error loading models: {e}")
        return
        
    ens_cfg = config.get('ensemble', {})
    ensemble = Ensemble(
        models=models_dict,
        wav2vec2_weight=ens_cfg.get('wav2vec2_weight', 0.5),
        rawnet2_weight=ens_cfg.get('rawnet2_weight', 0.5),
        threshold=ens_cfg.get('threshold', 0.5)
    )
    
    print(f"Analyzing {args.audio}...")
    try:
        audio, sr = librosa.load(args.audio, sr=None)
        result = ensemble.predict(audio, sr)
        
        print("\n--- Detection Results ---")
        print(f"Result: {result['verdict']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Final Score: {result['final_score']:.4f}")
        print(f"Wav2Vec2 Score: {result['wav2vec2_score']:.4f}")
        print(f"RawNet2 Score: {result['rawnet2_score']:.4f}")
        
    except Exception as e:
        print(f"Error analyzing audio: {e}")

def main():
    parser = argparse.ArgumentParser(description="Deepfake Voice Detection System")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # GUI
    subparsers.add_parser('gui', help='Launch the Web GUI')
    
    # Detect
    detect_parser = subparsers.add_parser('detect', help='Detect deepfake in audio file')
    detect_parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    
    args = parser.parse_args()
    
    if args.command == 'gui':
        run_gui()
    elif args.command == 'detect':
        run_detection(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
