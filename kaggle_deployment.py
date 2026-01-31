import os
# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import sys
import time
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from joblib import Parallel, delayed
import soundfile as sf
import warnings
import tempfile
import subprocess
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

CONFIG = {
    'audio': {
        'sample_rate': 16000,
        'chunk_duration': 3.0,
        'n_fft': 1024,
        'hop_length': 256
    },
    'features': {
        'n_mfcc': 40,
        'extract_spectral': True,
        'extract_temporal': True,
        'extract_chroma': True
    },
    # Dataset paths
    'dataset': {
        'base_path': "/kaggle/input", 
        'output_path': "./processed_data"
    },
    'augmentation': {
        'enabled': True,
        'train_only': True,
        'techniques': {
            'noise': {'enabled': True, 'probability': 0.3, 'noise_factor': 0.005},
            'time_stretch': {'enabled': True, 'probability': 0.2, 'rate_range': [0.9, 1.1]},
            'pitch_shift': {'enabled': False, 'probability': 0.0, 'semitone_range': [-2, 2]},
            'codec_simulation': {'enabled': True, 'probability': 0.3, 'codec': "opus", 'bitrate': "16k"}
        }
    },
    'model': {
        'architecture': "cnn_gru",
        'conv_filters': [64, 128],
        'kernel_size': 3,
        'pool_size': 2,
        'rnn_units': 128,
        'rnn_type': "gru",
        'dense_units': [64],
        'dropout_rate': 0.5,
        'dropout_rnn': 0.3,
        'batch_size': 16, # Reduced batch size for safety
        'epochs': 30,
        'learning_rate': 0.001,
        'early_stopping_patience': 5,
        'reduce_lr_patience': 3,
        'use_class_weights': True,
        'save_best_only': True,
        'model_save_path': "./output_models"
    },
    'processing': {
        'n_jobs': 4
    }
}

# =============================================================================
# 2. AUDIO AUGMENTATION UTILS
# =============================================================================

def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    return data + noise_factor * noise

def time_stretch(data, rate=1.1):
    return librosa.effects.time_stretch(data, rate=rate)

def pitch_shift(data, sr, n_steps=2):
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)

def codec_simulation(data, sr, codec='opus', bitrate='16k'):
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_in:
            with tempfile.NamedTemporaryFile(suffix=f'.{codec}', delete=False) as tmp_codec:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_out:
                    sf.write(tmp_in.name, data, sr)
                    subprocess.run([
                        'ffmpeg', '-y', '-i', tmp_in.name,
                        '-c:a', f'lib{codec}', '-b:a', bitrate,
                        tmp_codec.name
                    ], capture_output=True, check=True)
                    subprocess.run([
                        'ffmpeg', '-y', '-i', tmp_codec.name,
                        tmp_out.name
                    ], capture_output=True, check=True)
                    compressed_data, _ = librosa.load(tmp_out.name, sr=sr)
                    
                    Path(tmp_in.name).unlink()
                    Path(tmp_codec.name).unlink()
                    Path(tmp_out.name).unlink()
                    return compressed_data
    except Exception:
        return data

def apply_augmentation(data, sr, config):
    if not config['augmentation']['enabled']:
        return data
    
    aug_config = config['augmentation']['techniques']
    choice = np.random.random()
    cumulative_prob = 0
    
    if aug_config['noise']['enabled']:
        prob = aug_config['noise']['probability']
        if choice < cumulative_prob + prob:
            return add_noise(data, aug_config['noise']['noise_factor'])
        cumulative_prob += prob
        
    if aug_config['time_stretch']['enabled']:
        prob = aug_config['time_stretch']['probability']
        if choice < cumulative_prob + prob:
            rate = np.random.uniform(*aug_config['time_stretch']['rate_range'])
            return time_stretch(data, rate)
        cumulative_prob += prob
        
    if aug_config['codec_simulation']['enabled']:
        prob = aug_config['codec_simulation']['probability']
        if choice < cumulative_prob + prob:
            return codec_simulation(data, sr, aug_config['codec_simulation']['codec'], aug_config['codec_simulation']['bitrate'])
        cumulative_prob += prob
        
    return data

# =============================================================================
# 3. FEATURE EXTRACTION UTILS
# =============================================================================

def extract_all_features(audio_data, sr, config):
    features_list = []
    
    # MFCC
    n_mfcc = config['features']['n_mfcc']
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs, axis=1)
    features_list.append(mfccs_mean)
    
    # Spectral
    if config['features']['extract_spectral']:
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sr))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio_data, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sr))
        features_list.extend([spectral_centroid, spectral_bandwidth, spectral_contrast, spectral_rolloff])
    
    # Chroma
    if config['features']['extract_chroma']:
        chroma = np.mean(librosa.feature.chroma_stft(y=audio_data, sr=sr))
        tonnetz = np.mean(librosa.feature.tonnetz(y=audio_data, sr=sr))
        features_list.extend([chroma, tonnetz])
    
    # Temporal
    if config['features']['extract_temporal']:
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_data))
        rmse = np.mean(librosa.feature.rms(y=audio_data))
        features_list.extend([zero_crossing_rate, rmse])
        
    return np.hstack(features_list)

def process_single_file(file_path, label, config, apply_aug=False):
    try:
        sr = config['audio']['sample_rate']
        try:
            # Main loader for wav
            audio, _ = librosa.load(file_path, sr=sr)
        except Exception:
            # Fallback
            audio, samplerate = sf.read(file_path)
            if samplerate != sr:
                audio = librosa.resample(audio, orig_sr=samplerate, target_sr=sr)
        
        if apply_aug:
            audio = apply_augmentation(audio, sr, config)
            
        features = extract_all_features(audio, sr, config)
        return features, label
    except Exception as e:
        return None

# =============================================================================
# 4. DATA LOADER (SCENEFAKE SPECIFIC)
# =============================================================================

def find_files(base_path, pattern):
    """Recursive find files matching pattern"""
    return list(Path(base_path).rglob(pattern))

def load_and_process_data(split_name, config):
    print(f"\nProcessing {split_name.upper()} split...")
    
    kaggle_input = Path("/kaggle/input")
    print(f"   Searching for data in {kaggle_input}...")

    # SceneFake Structure:
    # root/train/real/*.wav
    # root/train/fake/*.wav
    
    # 1. Find the split directory (e.g., 'train', 'dev', 'eval')
    # We look for a directory that matches the split name AND contains 'real'/'fake' or '0'/'1' subdirs
    
    target_dir = None
    
    # Heuristic 1: Look for exact folder name match first
    # Walk the directory tree to find 'train' / 'dev' / 'eval' folders
    for root, dirs, files in os.walk(kaggle_input):
        current_path = Path(root)
        
        # Check if the folder name matches the split (case insensitive)
        if current_path.name.lower() == split_name.lower():
            # Verify it has suitable subdirectories or content
            # We look for 'real'/'fake' subdirs
            subdirs = [d.lower() for d in dirs]
            if 'real' in subdirs or 'fake' in subdirs:
                target_dir = current_path
                break
                
    # Heuristic 2: If finding by name failed, just look for ANY folder with 'real' and 'fake' 
    if target_dir is None:
        for root, dirs, files in os.walk(kaggle_input):
            subdirs = [d.lower() for d in dirs]
            if 'real' in subdirs and 'fake' in subdirs:
                # We found a folder containing classes. Is it the right split?
                if split_name.lower() in str(root).lower():
                    target_dir = Path(root)
                    break
    
    # Special Handling for SceneFake if folders are named differently (e.g. A, B, C)
    if target_dir is None:
        alt_names = {'train': 'A', 'dev': 'B', 'eval': 'C'}
        alt_name = alt_names.get(split_name)
        if alt_name:
             for root, dirs, files in os.walk(kaggle_input):
                 if Path(root).name == alt_name and ('real' in dirs or 'fake' in dirs):
                     target_dir = Path(root)
                     break

    if target_dir is None:
        print(f"   ❌ CRITICAL: Could not find directory for split '{split_name}'.")
        print("   Dumping available directories for debug:")
        for root, dirs, _ in os.walk(kaggle_input):
            subdirs = [d.lower() for d in dirs]
            if 'real' in subdirs or 'fake' in subdirs:
                print(f"      Found candidate (but rejected): {root}")
        return np.array([]), np.array([])
        
    print(f"   ✅ Found data directory: {target_dir}")
    
    # 2. Collect Files
    real_dir = None
    fake_dir = None
    
    for d in target_dir.iterdir():
        if d.is_dir():
            if d.name.lower() == 'real': real_dir = d
            if d.name.lower() == 'fake': fake_dir = d
            
    file_label_pairs = []
    
    if real_dir and real_dir.exists():
        real_files = list(real_dir.rglob('*.wav'))
        print(f"   Found {len(real_files)} Real files")
        for f in real_files: file_label_pairs.append((str(f), 0)) # 0 for Bonafide
        
    if fake_dir and fake_dir.exists():
        fake_files = list(fake_dir.rglob('*.wav'))
        print(f"   Found {len(fake_files)} Fake files")
        for f in fake_files: file_label_pairs.append((str(f), 1)) # 1 for Spoof/Fake
        
    if not file_label_pairs:
        print("   ⚠️ No .wav files found in real/fake directories.")
        return np.array([]), np.array([])

    print(f"   Prepared {len(file_label_pairs)} files for processing.")
    
    # 3. Parallel Extraction
    # Augmentation only for training split
    apply_aug = config['augmentation']['enabled'] and config['augmentation']['train_only'] and split_name == 'train'
    n_jobs = config['processing']['n_jobs']
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_file)(fp, lbl, config, apply_aug)
        for fp, lbl in tqdm(file_label_pairs, desc=f"   Extracting features")
    )
    
    features, labels = [], []
    for res in results:
        if res is not None:
            f, l = res
            features.append(f)
            labels.append(l)
            
    return np.array(features), np.array(labels)

# =============================================================================
# 5. MODEL ARCHITECTURE
# =============================================================================

def build_cnn_gru(input_shape, config):
    inputs = layers.Input(shape=input_shape, name='input')
    x = inputs
    
    # CNN layers
    for i, filters in enumerate(config['conv_filters']):
        x = layers.Conv1D(filters=filters, kernel_size=config['kernel_size'], padding='same', activation='relu')(x)
        x = layers.MaxPooling1D(pool_size=config['pool_size'])(x)
        x = layers.Dropout(config['dropout_rnn'])(x)
    
    # GRU layer
    x = layers.GRU(units=config['rnn_units'], return_sequences=False)(x)
    x = layers.Dropout(config['dropout_rate'])(x)
    
    # Dense layers
    for units in config['dense_units']:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(config['dropout_rate'])(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=outputs, name='CNN_GRU')

# =============================================================================
# 6. TRAINING WORKFLOW
# =============================================================================

def run_pipeline():
    # 1. Prepare Data
    X_train, y_train = load_and_process_data('train', CONFIG)
    X_val, y_val = load_and_process_data('dev', CONFIG) # Use 'dev' as validation
    
    if len(X_train) == 0:
        print("No training data found. Exiting.")
        return

    # Expand dims for Conv1D (Samples, Features) -> (Samples, Features, 1)
    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    
    print(f"Train Shape: {X_train.shape}, Val Shape: {X_val.shape}")
    
    # 2. Build Model
    model = build_cnn_gru(X_train.shape[1:], CONFIG['model'])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['model']['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    model.summary()
    
    # 3. Train
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=CONFIG['model']['early_stopping_patience'], restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=CONFIG['model']['reduce_lr_patience'])
    ]
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(zip(np.unique(y_train), class_weights))
    
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG['model']['epochs'],
        batch_size=CONFIG['model']['batch_size'],
        class_weight=class_weights_dict,
        callbacks=callbacks
    )
    
    # 4. Save Weights
    os.makedirs(CONFIG['model']['model_save_path'], exist_ok=True)
    save_path = os.path.join(CONFIG['model']['model_save_path'], "deepfake_detector_weights.h5")
    model.save_weights(save_path)
    print(f"Weights saved to {save_path}")
    
    # Save full model too just in case
    model.save(os.path.join(CONFIG['model']['model_save_path'], "deepfake_detector_model.h5"))

if __name__ == "__main__":
    run_pipeline()
