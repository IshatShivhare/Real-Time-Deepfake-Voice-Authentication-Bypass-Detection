"""
Deepfake Audio Detector Ensemble
Combines Vocoder-Artifacts (RawNet) and Custom (CNN+GRU) models.
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
import pickle
import librosa
from pathlib import Path

from src.utils.config_loader import get_config
from src.utils.logger import get_logger
from src.models.vocoder.model import RawNet
from src.audio.utils import preprocess_audio
from src.data_processing.utils.features import extract_all_features

logger = get_logger("EnsembleDetector")

class EnsembleDetector:
    """
    Ensemble Deepfake detector.
    """
    def __init__(self, device=None):
        self.config = get_config()
        if device is None:
            # Check config for GPU preference
            use_gpu = self.config.get('app', {}).get('use_gpu', True)
            if use_gpu and torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
                if use_gpu and not torch.cuda.is_available():
                    logger.warning("GPU enabled in config but CUDA not available. Using CPU.")
        else:
            self.device = device
            
        self.vocoder_model = None
        self.custom_model = None
        self.scaler = None
        self.weights_loaded = False
        
        logger.info(f"Initialized EnsembleDetector on {self.device}")
        
        # Load models immediately
        self.load_models()

    def load_models(self):
        """Load both models and scaler."""
        self._load_vocoder()
        self._load_custom()
        self._load_scaler()
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
                # Try relative to project root if needed
                alt_path = Path(weights_path)
                if not alt_path.is_absolute():
                     # Assume it's relative to root
                     pass
                logger.warning(f"Custom model weights not found at {weights_path}")
                
        except Exception as e:
            logger.error(f"Error loading Custom model: {e}")
            self.custom_model = None

    def _load_scaler(self):
        """Load feature scaler."""
        logger.info("Loading Feature Scaler...")
        # Path from config
        data_dir = self.config.get('dataset', {}).get('output_path', './data')
        scaler_path = Path(data_dir) / 'scaler.pkl'
        
        try:
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Scaler loaded from {scaler_path}")
            else:
                logger.warning(f"Scaler not found at {scaler_path}")
        except Exception as e:
            logger.error(f"Error loading scaler: {e}")
            self.scaler = None

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
        """Get prediction from Custom model using extracted features"""
        if self.custom_model is None or self.scaler is None:
            logger.warning("Custom model or scaler not loaded, returning default score 0.5")
            return 0.5
            
        try:
            # 1. Load audio
            sr = self.config['audio']['sample_rate']
            audio, _ = librosa.load(audio_path, sr=sr)
            
            # 2. Extract features
            features = extract_all_features(audio, sr, self.config)
            
            # 3. Scale features
            # features is 1D, transform expects 2D
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # 4. Reshape for model (batch, features, channels)
            # Match trainer.py: np.expand_dims(X, axis=-1)
            X = np.expand_dims(features_scaled, axis=-1)
            
            # 5. Predict
            prediction = self.custom_model.predict(X, verbose=0)
            
            # Return probability of fake (1)
            return float(prediction[0][0])
            
        except Exception as e:
            logger.error(f"Error in Custom model prediction: {e}")
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
        
        # 2. Custom Prediction
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
