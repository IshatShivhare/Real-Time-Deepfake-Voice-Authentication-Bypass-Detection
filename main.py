#!/usr/bin/env python
"""
Deepfake Voice Detection System
Main Entry Point
"""
import argparse
import sys
import os
from pathlib import Path

# Suppress TensorFlow info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.utils.logger import setup_logger
from src.utils.config_loader import load_config

logger = setup_logger("Main")

def run_gui():
    """Run the Streamlit GUI"""
    print("Launching GUI...")
    # Streamlit needs to be run as a module or script
    # We use os.system or subprocess to launch streamlit
    app_path = Path("src/gui/app.py").resolve()
    os.system(f"streamlit run \"{app_path}\"")

def run_pipeline(args):
    """Run data processing pipeline"""
    from src.data_processing.pipeline import run_pipeline as execute_pipeline
    
    steps = None
    if args.steps:
        steps = args.steps.split(',')
        
    execute_pipeline(steps)

def run_training(args):
    """Run model training"""
    from src.models.custom.train_wrapper import run_training as execute_training
    execute_training()

def run_detection(args):
    """Run inference on a single file"""
    if not args.audio:
        print("Error: --audio argument required for detection")
        return
        
    from src.models.ensemble import EnsembleDetector
    
    detector = EnsembleDetector()
    if not detector.weights_loaded:
        print("Warning: Model weights not fully loaded.")
        
    prediction, confidence, details = detector.predict_single(args.audio)
    
    result = "FAKE" if prediction == 1 else "REAL"
    print(f"\nResult: {result}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Details: {details}")

def main():
    parser = argparse.ArgumentParser(description="Deepfake Voice Detection System")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # GUI
    subparsers.add_parser('gui', help='Launch the Web GUI')
    
    # Pipeline
    pipeline_parser = subparsers.add_parser('pipeline', help='Run Data Processing Pipeline')
    pipeline_parser.add_argument('--steps', type=str, help='Comma-separated steps (organize,extract,prepare)')
    
    # Train
    subparsers.add_parser('train', help='Train the model')
    
    # Detect
    detect_parser = subparsers.add_parser('detect', help='Detect deepfake in audio file')
    detect_parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    
    args = parser.parse_args()
    
    # Load config
    try:
        load_config("config.yaml")
    except Exception as e:
        logger.warning(f"Could not load config.yaml: {e}")
    
    if args.command == 'gui':
        run_gui()
    elif args.command == 'pipeline':
        run_pipeline(args)
    elif args.command == 'train':
        run_training(args)
    elif args.command == 'detect':
        run_detection(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
