"""
Audio preprocessing utilities.
"""
import numpy as np
import soundfile as sf
import librosa
import torch
from src.utils.config_loader import get_config

def get_audio_config():
    config = get_config()
    return {
        'sr': config.get('audio', {}).get('sample_rate', 16000),
        # Default to ~4s if not specified, or calc from chunk_duration
        'target_length': config.get('model_vocoder', {}).get('nb_samp', 64600)
    }

def load_audio(file_path, target_sr=None):
    """
    Load audio file and resample if necessary.
    """
    if target_sr is None:
        target_sr = get_audio_config()['sr']
        
    try:
        audio, sr = sf.read(file_path)
    except Exception:
        audio, sr = librosa.load(file_path, sr=None)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
        
    # Resample if necessary
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    return audio, sr

def pad_or_trim(audio, target_length=None):
    """
    Pad or trim audio to target length.
    """
    if target_length is None:
        target_length = get_audio_config()['target_length']
        
    current_length = len(audio)
    
    if current_length >= target_length:
        # Trim: take center portion
        start = (current_length - target_length) // 2
        return audio[start:start + target_length]
    else:
        # Pad: repeat audio to fill length
        num_repeats = int(np.ceil(target_length / current_length))
        padded = np.tile(audio, num_repeats)
        return padded[:target_length]

def normalize_audio(audio):
    """
    Normalize audio amplitude to [-1, 1].
    """
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio

def remove_silence(audio, top_db=30):
    """
    Remove silence from audio using librosa.
    """
    intervals = librosa.effects.split(audio, top_db=top_db)
    
    if len(intervals) == 0:
        return audio
        
    non_silent_audio = np.concatenate([audio[start:end] for start, end in intervals])
    
    if len(non_silent_audio) < 1000:
        return audio
        
    return non_silent_audio

def preprocess_audio(file_path, target_sr=None, target_length=None):
    """
    Complete preprocessing pipeline for audio.
    """
    cfg = get_audio_config()
    if target_sr is None: target_sr = cfg['sr']
    if target_length is None: target_length = cfg['target_length']
    
    # Load audio
    audio, _ = load_audio(file_path, target_sr)
    
    # Remove silence
    audio = remove_silence(audio)
    
    # Normalize
    audio = normalize_audio(audio)
    
    # Pad or trim
    audio = pad_or_trim(audio, target_length)
    
    # Convert to tensor
    audio_tensor = torch.FloatTensor(audio)
    
    return audio_tensor
