"""
Deepfake Audio Detector
Uses Vocoder-Artifacts (RawNet) model for deepfake detection
"""
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import yaml
from pathlib import Path

# Add models directory to path
sys.path.append(str(Path(__file__).parent / 'models'))
sys.path.append(str(Path(__file__).parent / 'utils'))

from models.vocoder_model import RawNet
from utils.audio_utils import preprocess_audio


class DeepfakeDetector:
    """
    Deepfake detector using Vocoder-Artifacts (RawNet) model
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.weights_loaded = False
        
        print(f"Using device: {self.device}")
        
    def load_model(self, weight_path='weights/librifake_pretrained_lambda0.5_epoch_25.pth', 
                   config_path='models/model_config_RawNet.yaml'):
        """Load Vocoder-Artifacts (RawNet) model"""
        print("Loading Vocoder-Artifacts (RawNet) model...")
        
        try:
            # Load config
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                model_config = config['model']
            else:
                # Default config
                model_config = {
                    'in_channels': 1,
                    'first_conv': 251,
                    'filts': [[20, 20], [20, 64], [64, 64]],
                    'gru_node': 1024,
                    'nb_gru_layer': 3,
                    'nb_fc_node': 1024
                }
            
            self.model = RawNet(model_config, self.device).to(self.device)
            
            if os.path.exists(weight_path):
                state_dict = torch.load(weight_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.weights_loaded = True
                print(f"[OK] Model weights loaded from {weight_path}")
            else:
                self.weights_loaded = False
                print(f"[WARN] Model weights not found at {weight_path}")
                print("  Model initialized with random weights")
            
            self.model.eval()
            
        except Exception as e:
            print(f"[ERR] Error loading model: {e}")
            self.model = None
    
    def predict_single(self, audio_path):
        """
        Predict if audio is real or fake
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            prediction: 0 (bonafide/real) or 1 (spoof/fake)
            confidence: Confidence score (0-1)
            details: Dictionary with prediction details
        """
        if not self.weights_loaded or self.model is None:
            raise ValueError("Model not loaded with weights!")
        
        # Preprocess audio
        audio = preprocess_audio(audio_path).unsqueeze(0).to(self.device)
        
        # Vocoder prediction
        with torch.no_grad():
            output_binary, _ = self.model(audio)
            probs = torch.exp(output_binary)  # LogSoftmax output
            prob_fake = probs[0, 1].item()
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0, pred].item()
        
        details = {
            'prediction': pred,
            'confidence': conf,
            'fake_probability': prob_fake,
            'real_probability': probs[0, 0].item()
        }
        
        return pred, conf, details
    
    def predict_batch(self, audio_paths):
        """
        Predict multiple audio files
        
        Args:
            audio_paths: List of paths to audio files
        
        Returns:
            results: List of (prediction, confidence, details) tuples
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict_single(audio_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                results.append((None, None, {'error': str(e)}))
        
        return results


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deepfake Audio Detector')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Initialize detector
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    detector = DeepfakeDetector(device=device)
    
    # Load model
    detector.load_model()
    
    # Make prediction
    print(f"\nAnalyzing: {args.audio}")
    print("-" * 50)
    
    prediction, confidence, details = detector.predict_single(args.audio)
    
    # Display results
    result_text = "FAKE (Deepfake)" if prediction == 1 else "REAL (Bonafide)"
    print(f"\nFinal Prediction: {result_text}")
    print(f"Confidence: {confidence:.2%}")
    print(f"\nProbabilities:")
    print(f"  Real: {details['real_probability']:.2%}")
    print(f"  Fake: {details['fake_probability']:.2%}")


if __name__ == '__main__':
    main()
