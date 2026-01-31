"""
Audio augmentation functions
"""
import numpy as np
import librosa
import subprocess
import tempfile
from pathlib import Path

def add_noise(data, noise_factor=0.005):
    """Add random Gaussian noise"""
    noise = np.random.randn(len(data))
    return data + noise_factor * noise

def time_stretch(data, rate=1.1):
    """Time-stretch audio"""
    # Ensure minimum length before time stretching
    min_length = 4096
    if len(data) < min_length:
        data = np.pad(data, (0, min_length - len(data)), mode='constant')
    return librosa.effects.time_stretch(data, rate=rate)

def pitch_shift(data, sr, n_steps=2):
    """Pitch-shift audio"""
    # Ensure minimum length before pitch shifting
    min_length = 4096
    if len(data) < min_length:
        data = np.pad(data, (0, min_length - len(data)), mode='constant')
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)

def codec_simulation(data, sr, codec='opus', bitrate='16k'):
    """
    Simulate VoIP codec compression/decompression
    This is a DIFFERENTIATOR - simulates real WhatsApp/Telegram quality
    """
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_in:
            with tempfile.NamedTemporaryFile(suffix=f'.{codec}', delete=False) as tmp_codec:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_out:
                    # Write original audio
                    import soundfile as sf
                    sf.write(tmp_in.name, data, sr)
                    
                    # Encode to codec
                    subprocess.run([
                        'ffmpeg', '-y', '-i', tmp_in.name,
                        '-c:a', f'lib{codec}', '-b:a', bitrate,
                        tmp_codec.name
                    ], capture_output=True, check=True)
                    
                    # Decode back to wav
                    subprocess.run([
                        'ffmpeg', '-y', '-i', tmp_codec.name,
                        tmp_out.name
                    ], capture_output=True, check=True)
                    
                    # Read compressed audio
                    compressed_data, _ = librosa.load(tmp_out.name, sr=sr)
                    
                    # Cleanup
                    Path(tmp_in.name).unlink()
                    Path(tmp_codec.name).unlink()
                    Path(tmp_out.name).unlink()
                    
                    return compressed_data
    except Exception as e:
        # If codec simulation fails, return original
        return data

def apply_augmentation(data, sr, config):
    """
    Apply random augmentation based on config
    """
    if not config['augmentation']['enabled']:
        return data
    
    # Ensure minimum length before any augmentation
    min_length = 4096
    if len(data) < min_length:
        data = np.pad(data, (0, min_length - len(data)), mode='constant')
    
    aug_config = config['augmentation']['techniques']
    
    # Choose augmentation randomly
    choice = np.random.random()
    cumulative_prob = 0
    
    # Noise
    if aug_config['noise']['enabled']:
        prob = aug_config['noise']['probability']
        if choice < cumulative_prob + prob:
            return add_noise(data, aug_config['noise']['noise_factor'])
        cumulative_prob += prob
    
    # Time stretch
    if aug_config['time_stretch']['enabled']:
        prob = aug_config['time_stretch']['probability']
        if choice < cumulative_prob + prob:
            rate_range = aug_config['time_stretch']['rate_range']
            rate = np.random.uniform(*rate_range)
            return time_stretch(data, rate)
        cumulative_prob += prob
    
    # Pitch shift
    if aug_config['pitch_shift']['enabled']:
        prob = aug_config['pitch_shift']['probability']
        if choice < cumulative_prob + prob:
            semitone_range = aug_config['pitch_shift']['semitone_range']
            n_steps = np.random.randint(*semitone_range)
            return pitch_shift(data, sr, n_steps)
        cumulative_prob += prob
    
    # Codec simulation
    if aug_config['codec_simulation']['enabled']:
        prob = aug_config['codec_simulation']['probability']
        if choice < cumulative_prob + prob:
            return codec_simulation(
                data, sr,
                aug_config['codec_simulation']['codec'],
                aug_config['codec_simulation']['bitrate']
            )
        cumulative_prob += prob
    
    # No augmentation
    return data