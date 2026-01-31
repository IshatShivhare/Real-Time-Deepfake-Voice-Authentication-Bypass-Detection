"""
Audio preprocessing utilities for ensemble deepfake detector
"""
import numpy as np
import soundfile as sf
import librosa
import torch
from torch import Tensor


def load_audio(file_path, target_sr=16000):
    """
    Load audio file and resample if necessary
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate (default: 16000)
    
    Returns:
        audio: Audio waveform as numpy array
        sr: Sample rate
    """
    try:
        audio, sr = sf.read(file_path)
    except:
        audio, sr = librosa.load(file_path, sr=None)
    
    # Resample if necessary
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    return audio, sr


def pad_or_trim(audio, target_length=64600):
    """
    Pad or trim audio to target length
    
    Args:
        audio: Audio waveform
        target_length: Target length in samples (default: 64600 = ~4 seconds at 16kHz)
    
    Returns:
        Padded/trimmed audio
    """
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
    Normalize audio amplitude to [-1, 1]
    
    Args:
        audio: Audio waveform
    
    Returns:
        Normalized audio
    """
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


def preprocess_audio(file_path, target_sr=16000, target_length=64600):
    """
    Complete preprocessing pipeline for audio
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        target_length: Target length in samples
    
    Returns:
        Preprocessed audio as torch tensor
    """
    # Load audio
    audio, sr = load_audio(file_path, target_sr)
    
    # Normalize
    audio = normalize_audio(audio)
    
    # Pad or trim
    audio = pad_or_trim(audio, target_length)
    
    # Convert to tensor
    audio_tensor = torch.FloatTensor(audio)
    
    return audio_tensor
