"""
Step 1: Download and organize ASVspoof 2019 LA dataset
"""
import os
import shutil
import sys
from pathlib import Path
from tqdm import tqdm
from src.utils.config_loader import get_config

def organize_asvspoof_data(config=None):
    """
    Organize ASVspoof data into structured folders
    """
    if config is None:
        config = get_config()
        
    base_path = Path(config['dataset']['base_path'])
    output_path = Path(config['dataset']['output_path'])
    
    print("=" * 60)
    print("STEP 1: Organizing ASVspoof 2019 LA Dataset")
    print("=" * 60)
    
    # Check if base path exists
    if not base_path.exists():
        print(f"\n❌ Dataset not found at: {base_path}")
        print("\nPlease download ASVspoof 2019 LA from:")
        print("https://datashare.ed.ac.uk/handle/10283/3336")
        return False
    
    # Create output directories
    for split in ['train', 'dev', 'eval']:
        for label in ['bonafide', 'spoof']:
            (output_path / split / label).mkdir(parents=True, exist_ok=True)
    
    # Process each split
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
            print(f"   ⚠️  Protocol file not found: {protocol_file}")
            continue
        
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

if __name__ == "__main__":
    organize_asvspoof_data()