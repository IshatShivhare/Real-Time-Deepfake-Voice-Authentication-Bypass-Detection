"""
Deepfake Audio Detector Ensemble
Combines Vocoder-Artifacts (RawNet) and Custom (CNN+GRU) models.
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
from pathlib import Path

from src.utils.config_loader import get_config
from src.utils.logger import get_logger
from src.models.vocoder.model import RawNet
from src.audio.utils import preprocess_audio

logger = get_logger("EnsembleDetector")

class EnsembleDetector:
    """
    Ensemble Deepfake detector.
    """
    def __init__(self, device=None):
        self.config = get_config()
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.vocoder_model = None
        self.custom_model = None
        self.weights_loaded = False
        
        logger.info(f"Initialized EnsembleDetector on {self.device}")
        
        # Load models immediately
        self.load_models()

    def load_models(self):
        """Load both models."""
        self._load_vocoder()
        self._load_custom()
        self.weights_loaded = True

    def _load_vocoder(self):
        """Load Vocoder-Artifacts (RawNet) model (PyTorch)"""
        logger.info("Loading Vocoder-Artifacts model...")
        cfg = self.config['model_vocoder']
        weights_path = cfg['weights_path']
        
        try:
            # Model definition expects a dict
            model_config = {
                'in_channels': cfg['in_channels'],
                'first_conv': cfg['first_conv'],
                'filts': cfg['filts'],
                'gru_node': cfg['gru_node'],
                'nb_gru_layer': cfg['nb_gru_layer'],
                'nb_fc_node': cfg['nb_fc_node']
            }
            
            self.vocoder_model = RawNet(model_config, self.device).to(self.device)
            
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location=self.device)
                self.vocoder_model.load_state_dict(state_dict)
                self.vocoder_model.eval()
                logger.info(f"Vocoder model loaded from {weights_path}")
            else:
                logger.warning(f"Vocoder weights not found at {weights_path}")
                
        except Exception as e:
            logger.error(f"Error loading Vocoder model: {e}")
            self.vocoder_model = None

    def _load_custom(self):
        """Load Custom CNN+GRU model (Keras)"""
        logger.info("Loading Custom model...")
        cfg = self.config['model_custom']
        weights_path = cfg['weights_path']
        
        try:
            if os.path.exists(weights_path):
                self.custom_model = tf.keras.models.load_model(weights_path)
                logger.info(f"Custom model loaded from {weights_path}")
            else:
                logger.warning(f"Custom model weights not found at {weights_path}")
                
        except Exception as e:
            logger.error(f"Error loading Custom model: {e}")
            self.custom_model = None

    def predict_vocoder(self, audio_tensor):
        """Get prediction from Vocoder model"""
        if self.vocoder_model is None:
            return 0.5 # Uncertain
            
        with torch.no_grad():
            audio_tensor = audio_tensor.to(self.device)
            if len(audio_tensor.shape) == 1:
                 audio_tensor = audio_tensor.unsqueeze(0)
                 
            output_binary, _ = self.vocoder_model(audio_tensor)
            probs = torch.exp(output_binary) # LogSoftmax -> Prob
            # Index 1 is usually fake in this dataset (Bonafide=0, Spoof=1)
            # Re-verifying assumption: in ASVspoof, 0=Bonafide, 1=Spoof usually.
            prob_fake = probs[0, 1].item()
            return prob_fake

    def predict_custom(self, audio_path):
        """Get prediction from Custom model"""
        if self.custom_model is None:
            return 0.5
            
        try:
            import librosa
            import pickle
            from src.data_processing.utils.features import extract_all_features
            
            # Load audio
            sr = self.config['audio']['sample_rate']
            audio, _ = librosa.load(audio_path, sr=sr)
            
            # Ensure minimum length (same as training)
            min_length = 8192
            if len(audio) < min_length:
                audio = np.pad(audio, (0, min_length - len(audio)), mode='constant')
            
            # Extract features (same as training)
            features = extract_all_features(audio, sr, self.config)
            
            # Load scaler and normalize
            scaler_path = Path(self.config['dataset']['output_path']) / 'scaler.pkl'
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                features = scaler.transform(features.reshape(1, -1))
            else:
                features = features.reshape(1, -1)
            
            # Predict
            prediction = self.custom_model.predict(features, verbose=0)
            prob_fake = float(prediction[0][0])
            
            return prob_fake
            
        except Exception as e:
            logger.error(f"Error in custom prediction: {e}")
            return 0.5

    def predict_single(self, audio_path):
        """
        Predict if audio is real or fake using ensemble.
        """
        if not self.weights_loaded:
            raise ValueError("Models not loaded")

        # 1. Vocoder Prediction (Raw Audio)
        vocoder_input = preprocess_audio(audio_path)
        vocoder_score = self.predict_vocoder(vocoder_input)
        
        # 2. Custom Prediction (Feature-based)
        custom_score = self.predict_custom(audio_path)
        
        # 3. Ensemble
        weights = self.config['ensemble']['weights']
        final_score = (vocoder_score * weights['vocoder']) + (custom_score * weights['custom'])
        
        pred = 1 if final_score > self.config['ensemble']['threshold'] else 0
        confidence = final_score if pred == 1 else (1 - final_score)
        
        details = {
            'prediction': pred,
            'confidence': confidence,
            'fake_probability': final_score,
            'vocoder_score': vocoder_score,
            'custom_score': custom_score
        }
        
        return pred, confidence, details
