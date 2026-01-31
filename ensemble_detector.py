"""
Ensemble Deepfake Audio Detector
Combines AASIST-L and Vocoder-Artifacts models for robust detection
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

from models.aasist_model import Model as AASIST
from models.vocoder_model import RawNet
from utils.audio_utils import preprocess_audio


class EnsembleDetector:
    """
    Ensemble detector combining AASIST-L and Vocoder-Artifacts models
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = {}
        self.weights_loaded = {}
        
        print(f"Using device: {self.device}")
        
    def load_aasist(self, weight_path='weights/AASIST-L.pth', 
                    config_path='models/model_config_AASIST.yaml'):
        """Load AASIST-L model"""
        print("Loading AASIST-L model...")
        
        try:
            # Load config
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                model_config = config['model']
            else:
                # Default AASIST-L config
                model_config = {
                    'first_conv': 128,
                    'filts': [70, [1, 32], [32, 32], [32, 64], [64, 64]],
                    'gat_dims': [80, 160],
                    'pool_ratios': [0.5, 0.5, 0.5],
                    'temperatures': [1.0, 1.0, 1.0],
                    'nb_samp': 64600,
                    'in_channels': 1
                }
            
            self.models['aasist'] = AASIST(model_config).to(self.device)
            
            if os.path.exists(weight_path):
                state_dict = torch.load(weight_path, map_location=self.device)
                self.models['aasist'].load_state_dict(state_dict)
                self.weights_loaded['aasist'] = True
                print(f"✓ AASIST-L weights loaded from {weight_path}")
            else:
                self.weights_loaded['aasist'] = False
                print(f"⚠ Warning: AASIST-L weights not found at {weight_path}")
                print("  Model initialized with random weights")
            
            self.models['aasist'].eval()
            
        except Exception as e:
            print(f"✗ Error loading AASIST-L: {e}")
            self.models['aasist'] = None
    
    def load_vocoder(self, weight_path='weights/librifake_pretrained_lambda0.5_epoch_25.pth', 
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
            
            self.models['vocoder'] = RawNet(model_config, self.device).to(self.device)
            
            if os.path.exists(weight_path):
                state_dict = torch.load(weight_path, map_location=self.device)
                self.models['vocoder'].load_state_dict(state_dict)
                self.weights_loaded['vocoder'] = True
                print(f"✓ Vocoder model weights loaded from {weight_path}")
            else:
                self.weights_loaded['vocoder'] = False
                print(f"⚠ Warning: Vocoder weights not found at {weight_path}")
                print("  Model initialized with random weights")
            
            self.models['vocoder'].eval()
            
        except Exception as e:
            print(f"✗ Error loading Vocoder model: {e}")
            self.models['vocoder'] = None
    
    def predict_single(self, audio_path, method='weighted_average'):
        """
        Predict if audio is real or fake
        
        Args:
            audio_path: Path to audio file
            method: Ensemble method ('weighted_average', 'voting', 'max_confidence')
        
        Returns:
            prediction: 0 (bonafide/real) or 1 (spoof/fake)
            confidence: Confidence score (0-1)
            details: Dictionary with individual model predictions
        """
        # Preprocess audio
        audio = preprocess_audio(audio_path).unsqueeze(0).to(self.device)
        
        predictions = {}
        confidences = {}
        
        # AASIST-L prediction
        if self.models.get('aasist') and self.weights_loaded.get('aasist'):
            with torch.no_grad():
                _, output = self.models['aasist'](audio)
                probs = F.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                conf = probs[0, pred].item()
                
                predictions['aasist'] = pred
                confidences['aasist'] = conf
        
        # Vocoder prediction
        if self.models.get('vocoder') and self.weights_loaded.get('vocoder'):
            with torch.no_grad():
                output_binary, _ = self.models['vocoder'](audio)
                probs = torch.exp(output_binary)  # LogSoftmax output
                pred = torch.argmax(probs, dim=1).item()
                conf = probs[0, pred].item()
                
                predictions['vocoder'] = pred
                confidences['vocoder'] = conf
        
        # Ensemble decision
        if len(predictions) == 0:
            raise ValueError("No models loaded with weights!")
        
        if method == 'weighted_average':
            # Weight AASIST more heavily (it's generally more accurate)
            weights = {'aasist': 0.6, 'vocoder': 0.4}
            weighted_score = 0
            total_weight = 0
            
            for model_name, pred in predictions.items():
                weight = weights.get(model_name, 0.5)
                weighted_score += pred * weight * confidences[model_name]
                total_weight += weight
            
            final_pred = 1 if weighted_score / total_weight > 0.5 else 0
            final_conf = abs(weighted_score / total_weight - 0.5) * 2
            
        elif method == 'voting':
            # Simple majority voting
            votes = list(predictions.values())
            final_pred = 1 if sum(votes) > len(votes) / 2 else 0
            final_conf = sum(confidences.values()) / len(confidences)
            
        elif method == 'max_confidence':
            # Use prediction from most confident model
            max_conf_model = max(confidences, key=confidences.get)
            final_pred = predictions[max_conf_model]
            final_conf = confidences[max_conf_model]
        
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        details = {
            'individual_predictions': predictions,
            'individual_confidences': confidences,
            'ensemble_method': method
        }
        
        return final_pred, final_conf, details
    
    def predict_batch(self, audio_paths, method='weighted_average'):
        """
        Predict multiple audio files
        
        Args:
            audio_paths: List of paths to audio files
            method: Ensemble method
        
        Returns:
            results: List of (prediction, confidence, details) tuples
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict_single(audio_path, method)
                results.append(result)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                results.append((None, None, {'error': str(e)}))
        
        return results


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ensemble Deepfake Audio Detector')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--method', type=str, default='weighted_average',
                       choices=['weighted_average', 'voting', 'max_confidence'],
                       help='Ensemble method')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Initialize detector
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    detector = EnsembleDetector(device=device)
    
    # Load models
    detector.load_aasist()
    detector.load_vocoder()
    
    # Make prediction
    print(f"\nAnalyzing: {args.audio}")
    print("-" * 50)
    
    prediction, confidence, details = detector.predict_single(args.audio, args.method)
    
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


if __name__ == '__main__':
    main()
