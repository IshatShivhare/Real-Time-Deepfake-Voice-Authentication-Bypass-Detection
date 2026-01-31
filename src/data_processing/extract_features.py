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
        # Load audio
        sr = config['audio']['sample_rate']
        try:
           audio, _ = librosa.load(file_path, sr=sr)
        except Exception:
           # Fallback if librosa fails (sometimes happens with flac on windows)
           import soundfile as sf
           audio, samplerate = sf.read(file_path)
           if samplerate != sr:
               audio = librosa.resample(audio, orig_sr=samplerate, target_sr=sr)
        
        # Apply augmentation if enabled
        if apply_aug:
            audio = apply_augmentation(audio, sr, config)
        
        # Extract features
        features = extract_all_features(audio, sr, config)
        
        return features, label
    
    except Exception as e:
        # print(f"Error processing {file_path}: {e}") # Reduce noise
        return None

def extract_features_for_split(split_name, config):
    print(f"\n{'='*60}")
    print(f"Processing {split_name.upper()} split")
    print(f"{'='*60}")
    
    data_path = Path(config['dataset']['output_path']) / split_name
    
    bonafide_files = list((data_path / 'bonafide').glob('*.flac'))
    spoof_files = list((data_path / 'spoof').glob('*.flac'))
    
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
    n_jobs = config['processing']['n_jobs']
    
    # Process in chunks or all at once? All at once is fine for <10k files usually
    # But better to use batching for memory safety if dataset is huge.
    
    results = Parallel(n_jobs=n_jobs)(
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