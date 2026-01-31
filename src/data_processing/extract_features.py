"""
Step 2: Extract features from audio files with parallel processing
"""
import numpy as np
import librosa
import time
import pickle
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

from src.utils.config_loader import get_config
from src.data_processing.utils.augmentation import apply_augmentation
from src.data_processing.utils.features import extract_all_features, get_feature_names
from src.data_processing.utils.validation import validate_dataset, print_dataset_stats

def process_single_file(file_path, label, config, apply_aug=False):
    """
    Process a single audio file
    """
    try:
        sr = config['audio']['sample_rate']
        
        # Load audio
        try:
            audio, _ = librosa.load(file_path, sr=sr)
        except Exception:
            # Fallback if librosa fails (sometimes happens with flac on windows)
            import soundfile as sf
            audio, samplerate = sf.read(file_path)
            if samplerate != sr:
                audio = librosa.resample(audio, orig_sr=samplerate, target_sr=sr)
        
        # Skip only truly invalid files (< 100 samples)
        if len(audio) < 100:
            return None
        
        # Apply augmentation FIRST (if enabled)
        if apply_aug:
            audio = apply_augmentation(audio, sr, config)
        
        # THEN pad to safe length AFTER augmentation
        # This ensures time_stretch doesn't leave us with short audio
        n_fft = config['audio'].get('n_fft', 2048)
        min_safe_length = max(8192, n_fft * 4)  # Extra safety margin
        
        if len(audio) < min_safe_length:
            audio = np.pad(audio, (0, min_safe_length - len(audio)), mode='constant')
        
        # Extract features - now audio is guaranteed to be long enough
        features = extract_all_features(audio, sr, config)
        
        return features, label
    
    except Exception as e:
        # Silently skip problematic files
        return None

def extract_features_for_split(split_name, config):
    print(f"\n{'='*60}")
    print(f"Processing {split_name.upper()} split")
    print(f"{'='*60}")
    
    data_path = Path(config['dataset']['output_path']) / split_name
    
    file_ext = config['dataset'].get('file_extension', 'flac')
    bonafide_files = list((data_path / 'bonafide').glob(f'*.{file_ext}'))
    spoof_files = list((data_path / 'spoof').glob(f'*.{file_ext}'))
    
    file_label_pairs = []
    for f in bonafide_files:
        file_label_pairs.append((f, 0))
    for f in spoof_files:
        file_label_pairs.append((f, 1))
    
    np.random.shuffle(file_label_pairs)
    
    print(f"   Total: {len(file_label_pairs)} files")
    
    # Simple processing for now, omitting complex checkpoint resume for brevity unless critical
    # But retaining parallel execution
    
    apply_aug = config['augmentation']['enabled'] and config['augmentation']['train_only'] and split_name == 'train'
    
    # Use only 2 jobs for maximum stability on Windows
    n_jobs = 2
    
    # Use threading backend for better stability on Windows with librosa
    results = Parallel(n_jobs=n_jobs, backend='threading', verbose=0)(
        delayed(process_single_file)(file_path, label, config, apply_aug)
        for file_path, label in tqdm(file_label_pairs, desc=f"   Extracting {split_name}")
    )
    
    features_list = []
    labels_list = []
    
    for result in results:
        if result is not None:
            features, label = result
            features_list.append(features)
            labels_list.append(label)
            
    skipped_count = len(file_label_pairs) - len(features_list)
    if skipped_count > 0:
        print(f"   ⚠️  Skipped {skipped_count} short/invalid files")
    print(f"   ✓ Extracted: {len(features_list)} files")

    return np.array(features_list), np.array(labels_list)

def run_extraction(config=None):
    if config is None:
        config = get_config()
        
    print("\n" + "="*60)
    print("STEP 2: Feature Extraction Pipeline")
    print("="*60)
    
    feature_names = get_feature_names(config)
    datasets = {}
    
    for split in ['train', 'dev', 'eval']:
        features, labels = extract_features_for_split(split, config)
        if len(features) == 0:
            print(f"Warning: No features extracted for {split}")
            features = np.zeros((0, len(feature_names)))
            labels = np.zeros((0,))
            
        datasets[split] = (features, labels)
        print_dataset_stats(features, labels, feature_names, f"{split.upper()} Set")
        
    output_path = Path(config['dataset']['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path / 'features_raw.npz',
        X_train=datasets['train'][0],
        y_train=datasets['train'][1],
        X_dev=datasets['dev'][0],
        y_dev=datasets['dev'][1],
        X_eval=datasets['eval'][0],
        y_eval=datasets['eval'][1],
        feature_names=feature_names
    )
    
    # Save partial metadata
    metadata = {
        'dataset_name': config['dataset']['name'],
        'n_features': len(feature_names),
        'feature_names': feature_names
    }
    with open(output_path / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

if __name__ == "__main__":
    run_extraction()