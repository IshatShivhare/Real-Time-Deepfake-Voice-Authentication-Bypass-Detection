"""
Step 2: Extract features from audio files with parallel processing
"""
import numpy as np
import librosa
import yaml
from pathlib import Path
from tqdm import tqdm
import pickle
from joblib import Parallel, delayed
import time

from utils import apply_augmentation, extract_all_features, get_feature_names, validate_dataset, print_dataset_stats

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def process_single_file(file_path, label, config, apply_aug=False):
    """
    Process a single audio file
    Returns: (features, label) or None if error
    """
    try:
        # Load audio
        sr = config['audio']['sample_rate']
        audio, _ = librosa.load(file_path, sr=sr)
        
        # Apply augmentation if enabled
        if apply_aug:
            audio = apply_augmentation(audio, sr, config)
        
        # Extract features
        features = extract_all_features(audio, sr, config)
        
        return features, label
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def extract_features_for_split(split_name, config):
    """
    Extract features for a single split (train/dev/eval)
    """
    print(f"\n{'='*60}")
    print(f"Processing {split_name.upper()} split")
    print(f"{'='*60}")
    
    data_path = Path(config['dataset']['output_path']) / split_name
    
    # Collect all files
    bonafide_files = list((data_path / 'bonafide').glob('*.flac'))
    spoof_files = list((data_path / 'spoof').glob('*.flac'))
    
    print(f"\n📁 Found files:")
    print(f"   Bonafide: {len(bonafide_files)}")
    print(f"   Spoof: {len(spoof_files)}")
    
    # Prepare file list with labels
    file_label_pairs = []
    for f in bonafide_files:
        file_label_pairs.append((f, 0))  # 0 = bonafide/real
    for f in spoof_files:
        file_label_pairs.append((f, 1))  # 1 = spoof/fake
    
    # Shuffle
    np.random.shuffle(file_label_pairs)
    
    print(f"   Total: {len(file_label_pairs)} files")
    
    # Check for existing checkpoint
    checkpoint_file = Path(config['dataset']['output_path']) / f'checkpoint_{split_name}.pkl'
    start_idx = 0
    features_list = []
    labels_list = []
    
    if config['processing']['resume_from_checkpoint'] and checkpoint_file.exists():
        print(f"\n📂 Loading checkpoint...")
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
            features_list = checkpoint['features']
            labels_list = checkpoint['labels']
            start_idx = checkpoint['index']
        print(f"   Resuming from index {start_idx}/{len(file_label_pairs)}")
    
    # Determine if augmentation should be applied
    apply_aug = config['augmentation']['enabled'] and config['augmentation']['train_only'] and split_name == 'train'
    
    # Process files in parallel
    n_jobs = config['processing']['n_jobs']
    checkpoint_interval = config['processing']['checkpoint_interval']
    
    print(f"\n⚙️  Processing with {n_jobs} parallel jobs...")
    
    start_time = time.time()
    
    # Process in batches for checkpointing
    for batch_start in range(start_idx, len(file_label_pairs), checkpoint_interval):
        batch_end = min(batch_start + checkpoint_interval, len(file_label_pairs))
        batch = file_label_pairs[batch_start:batch_end]
        
        # Parallel processing
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_single_file)(file_path, label, config, apply_aug)
            for file_path, label in tqdm(batch, desc=f"   Batch {batch_start//checkpoint_interval + 1}")
        )
        
        # Collect results
        for result in results:
            if result is not None:
                features, label = result
                features_list.append(features)
                labels_list.append(label)
        
        # Save checkpoint
        checkpoint_data = {
            'features': features_list,
            'labels': labels_list,
            'index': batch_end
        }
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        print(f"   Checkpoint saved at index {batch_end}")
    
    elapsed = time.time() - start_time
    
    # Convert to numpy arrays
    features = np.array(features_list)
    labels = np.array(labels_list)
    
    print(f"\n✅ Extraction complete!")
    print(f"   Time elapsed: {elapsed/60:.2f} minutes")
    print(f"   Processing rate: {len(features)/elapsed:.1f} files/sec")
    
    # Remove checkpoint file
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    
    return features, labels

def main():
    config = load_config()
    
    print("\n" + "="*60)
    print("STEP 2: Feature Extraction Pipeline")
    print("="*60)
    
    # Get feature names
    feature_names = get_feature_names(config)
    print(f"\n📋 Extracting {len(feature_names)} features per sample")
    
    # Process each split
    datasets = {}
    
    for split in ['train', 'dev', 'eval']:
        features, labels = extract_features_for_split(split, config)
        datasets[split] = (features, labels)
        
        # Print stats
        print_dataset_stats(features, labels, feature_names, f"{split.upper()} Set")
    
    # Save all datasets
    output_path = Path(config['dataset']['output_path'])
    
    print(f"\n💾 Saving datasets...")
    
    # Save as NPZ (compressed)
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
    
    # Save metadata
    metadata = {
        'dataset_name': config['dataset']['name'],
        'extraction_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'config': config,
        'splits': {
            'train': datasets['train'][0].shape[0],
            'dev': datasets['dev'][0].shape[0],
            'eval': datasets['eval'][0].shape[0],
        }
    }
    
    with open(output_path / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"   ✓ Saved to: {output_path}/features_raw.npz")
    print(f"   ✓ Metadata: {output_path}/metadata.pkl")
    
    print("\n" + "="*60)
    print("✅ Feature extraction complete!")
    print("="*60)
    print("\n Next step: python 03_feature_analysis.py")

if __name__ == "__main__":
    main()