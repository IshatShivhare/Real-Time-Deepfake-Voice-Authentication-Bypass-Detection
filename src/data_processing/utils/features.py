"""
Feature extraction functions
"""
import numpy as np
import librosa

def extract_all_features(audio_data, sr, config):
    """
    Extract all features as specified in config
    Returns: 1D feature vector
    """
    # FINAL SAFETY CHECK: Ensure minimum length for FFT
    # Use 8192 to cover all librosa operations including tonnetz (which uses internal n_fft=2048)
    n_fft = config['audio'].get('n_fft', 2048)
    min_length = max(8192, n_fft * 4)  # Extra safety margin
    
    if len(audio_data) < min_length:
        # If we got here with short audio, pad it now
        padding = min_length - len(audio_data)
        audio_data = np.pad(audio_data, (0, padding), mode='constant')

    features_list = []
    
    n_fft = config['audio'].get('n_fft', 2048)
    hop_length = config['audio'].get('hop_length', 512)

    # MFCC features (40 coefficients)
    n_mfcc = config['features']['n_mfcc']
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfccs_mean = np.mean(mfccs, axis=1)
    features_list.append(mfccs_mean)
    
    # Spectral features
    if config['features']['extract_spectral']:
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length))
        
        features_list.extend([
            spectral_centroid,
            spectral_bandwidth,
            spectral_contrast,
            spectral_rolloff
        ])
    
    # Chroma and Tonnetz
    if config['features']['extract_chroma']:
        chroma = np.mean(librosa.feature.chroma_stft(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length))
        tonnetz = np.mean(librosa.feature.tonnetz(y=audio_data, sr=sr))  # tonnetz manages its own fft usually but respects y length
        
        features_list.extend([chroma, tonnetz])
    
    # Temporal features
    if config['features']['extract_temporal']:
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_data, hop_length=hop_length, frame_length=n_fft))
        rmse = np.mean(librosa.feature.rms(y=audio_data, frame_length=n_fft, hop_length=hop_length))
        
        features_list.extend([zero_crossing_rate, rmse])
    
    # Combine all features
    features = np.hstack(features_list)
    
    return features

def get_feature_names(config):
    """
    Get feature names matching the extraction order
    """
    names = []
    
    # MFCC names
    n_mfcc = config['features']['n_mfcc']
    names.extend([f'MFCC_{i+1}' for i in range(n_mfcc)])
    
    # Spectral features
    if config['features']['extract_spectral']:
        names.extend([
            'Spectral_Centroid',
            'Spectral_Bandwidth',
            'Spectral_Contrast',
            'Spectral_Rolloff'
        ])
    
    # Chroma and Tonnetz
    if config['features']['extract_chroma']:
        names.extend(['Chroma', 'Tonnetz'])
    
    # Temporal features
    if config['features']['extract_temporal']:
        names.extend(['Zero_Crossing_Rate', 'RMSE'])
    
    return names