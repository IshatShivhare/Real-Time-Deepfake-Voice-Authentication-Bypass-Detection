"""
Step 1: Download and organize ASVspoof 2019 LA dataset
"""
import os
import shutil
from pathlib import Path
import yaml
from tqdm import tqdm

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def organize_asvspoof_data(config):
    """
    Organize ASVspoof data into structured folders
    
    Structure:
    processed_data/
      train/
        bonafide/
        spoof/
      dev/
        bonafide/
        spoof/
      eval/
        bonafide/
        spoof/
    """
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
        print("\nExpected structure:")
        print("ASVspoof2019/LA/")
        print("  ├── ASVspoof2019_LA_train/")
        print("  ├── ASVspoof2019_LA_dev/")
        print("  ├── ASVspoof2019_LA_eval/")
        print("  └── ASVspoof2019_LA_cm_protocols/")
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
        
        # Read protocol file
        with open(protocol_path, 'r') as f:
            lines = f.readlines()
        
        # Limit samples if specified
        max_samples = config['dataset'].get(f'{split_name}_samples')
        if max_samples:
            print(f"   Limiting to {max_samples} samples")
            lines = lines[:max_samples * 2]  # Rough estimate (half bonafide, half spoof)
        
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
            
            # Copy file (symlink for speed if on same filesystem)
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
    
    print("\n" + "=" * 60)
    print("✅ Dataset organization complete!")
    print(f"   Total bonafide: {stats['bonafide']}")
    print(f"   Total spoof: {stats['spoof']}")
    print(f"   Total samples: {stats['bonafide'] + stats['spoof']}")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    config = load_config()
    success = organize_asvspoof_data(config)
    
    if success:
        print("\n✓ Ready for feature extraction!")
        print("  Run: python 02_extract_features.py")
    else:
        print("\n✗ Please download the dataset first")