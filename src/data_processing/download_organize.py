"""
Step 1: Download and organize ASVspoof 2019 LA dataset
"""
import os
import shutil
import sys
from pathlib import Path
from tqdm import tqdm
from src.utils.config_loader import get_config

def organize_folder_dataset(config):
    """
    Organize folder-based datasets (like SceneFake) where data is already split into real/fake folders
    """
    base_path = Path(config['dataset']['base_path'])
    output_path = Path(config['dataset']['output_path'])
    file_ext = config['dataset'].get('file_extension', 'wav')
    
    print(f"\nProcessing folder-based dataset from: {base_path}")
    
    if not base_path.exists():
        print(f"❌ Dataset path not found: {base_path}")
        return False

    # Mapping: source_folder_name -> target_label
    # SceneFake uses 'real' and 'fake'. Map them to 'bonafide' and 'spoof' for consistency
    label_map = {
        'real': 'bonafide',
        'fake': 'spoof'
    }
    
    stats = {'bonafide': 0, 'spoof': 0}
    
    for split in ['train', 'dev', 'eval']:
        print(f"\n📁 Processing {split} split...")
        
        for src_label, target_label in label_map.items():
            # Source path: base_path/split/src_label (e.g., base/train/real)
            src_dir = base_path / split / src_label
            
            if not src_dir.exists():
                print(f"   ⚠️  Folder not found: {src_dir}")
                continue
                
            # Target path: output_path/split/target_label (e.g., processed/train/bonafide)
            dst_dir = output_path / split / target_label
            dst_dir.mkdir(parents=True, exist_ok=True)
            
            files = list(src_dir.glob(f'*.{file_ext}'))
            
            # Limit samples if configured
            max_samples = config['dataset'].get(f'{split}_samples')
            if max_samples:
                files = files[:max_samples]
                
            count = 0
            for src_file in tqdm(files, desc=f"   Copying {target_label}"):
                dst_file = dst_dir / src_file.name
                
                if not dst_file.exists():
                    try:
                        # Try symlink first for speed
                        os.symlink(src_file, dst_file)
                    except OSError:
                        shutil.copy2(src_file, dst_file)
                
                count += 1
            
            print(f"   ✓ {target_label}: {count} files")
            stats[target_label] += count
            
    print(f"\nFinal Stats: {stats}")
    return True

def organize_asvspoof_protocol(config):
    """
    Organize ASVspoof data using protocol files
    """
    base_path = Path(config['dataset']['base_path'])
    output_path = Path(config['dataset']['output_path'])
    
    # ... (Keep existing ASVspoof logic roughly same, but wrapped here)
    # For brevity, reusing the existing logic structure but adapting it slightly if needed.
    # Since I'm replacing the whole file content essentially, I will just paste the logic here.
    
    # Check if base path exists
    if not base_path.exists():
        print(f"\n❌ Dataset not found at: {base_path}")
        return False
    
    # Create output directories
    for split in ['train', 'dev', 'eval']:
        for label in ['bonafide', 'spoof']:
            (output_path / split / label).mkdir(parents=True, exist_ok=True)
    
    splits = [
        ('train', 'ASVspoof2019_LA_train', 'ASVspoof2019.LA.cm.train.trn.txt'),
        ('dev', 'ASVspoof2019_LA_dev', 'ASVspoof2019.LA.cm.dev.trl.txt'),
        ('eval', 'ASVspoof2019_LA_eval', 'ASVspoof2019.LA.cm.eval.trl.txt')
    ]
    
    stats = {'bonafide': 0, 'spoof': 0}
    
    for split_name, audio_folder, protocol_file in splits:
        print(f"\n📁 Processing {split_name} split...")
        
        protocol_path = base_path / 'ASVspoof2019_LA_cm_protocols' / protocol_file
        audio_path = base_path / audio_folder / 'flac'
        
        if not protocol_path.exists():
            # Try alternative path structure sometimes found in unzipped datasets
            protocol_path = base_path / 'LA' / 'ASVspoof2019_LA_cm_protocols' / protocol_file
            if not protocol_path.exists():
                print(f"   ⚠️  Protocol file not found: {protocol_file}")
                continue
        
        if not audio_path.exists():
             # Try alternative path
             audio_path = base_path / 'LA' / audio_folder / 'flac'
             if not audio_path.exists():
                 print(f"   ⚠️  Audio folder not found: {audio_path}")
                 continue

        # Read protocol file
        with open(protocol_path, 'r') as f:
            lines = f.readlines()
        
        # Limit samples if specified
        max_samples = config['dataset'].get(f'{split_name}_samples')
        if max_samples:
            print(f"   Limiting to {max_samples} samples")
            lines = lines[:max_samples * 2]
        
        bonafide_count = 0
        spoof_count = 0
        
        for line in tqdm(lines, desc=f"   Organizing {split_name}"):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            filename = parts[1] + '.flac'
            label = parts[4]  # 'bonafide' or 'spoof'
            
            src_file = audio_path / filename
            
            if not src_file.exists():
                continue
            
            # Determine destination
            dst_folder = output_path / split_name / label
            dst_file = dst_folder / filename
            
            # Copy file
            if not dst_file.exists():
                try:
                    os.symlink(src_file, dst_file)
                except OSError:
                    shutil.copy2(src_file, dst_file)
            
            if label == 'bonafide':
                bonafide_count += 1
            else:
                spoof_count += 1
        
        print(f"   ✓ {split_name}: {bonafide_count} bonafide, {spoof_count} spoof")
        stats['bonafide'] += bonafide_count
        stats['spoof'] += spoof_count
    return True

def organize_data(config=None):
    if config is None:
        config = get_config()
        
    print("=" * 60)
    print("STEP 1: Organizing Dataset")
    print("=" * 60)
    
    dataset_type = config['dataset'].get('type', 'protocol')
    
    if dataset_type == 'folder':
        return organize_folder_dataset(config)
    else:
        return organize_asvspoof_protocol(config)

if __name__ == "__main__":
    organize_data()