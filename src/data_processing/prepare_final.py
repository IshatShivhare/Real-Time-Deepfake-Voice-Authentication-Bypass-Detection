"""
Step 4: Prepare final dataset with normalization and optional feature selection
"""
import numpy as np
import pickle
import time
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.utils.config_loader import get_config
from src.data_processing.utils.validation import validate_dataset, print_dataset_stats

def run_preparation(config=None):
    if config is None:
        config = get_config()
        
    output_path = Path(config['dataset']['output_path'])
    raw_file = output_path / 'features_raw.npz'
    
    if not raw_file.exists():
        print(f"Raw features not found at {raw_file}")
        return
        
    print("\n" + "="*60)
    print("STEP 4: Final Dataset Preparation")
    print("="*60)
    
    data = np.load(raw_file, allow_pickle=True)
    X_train, y_train = data['X_train'], data['y_train']
    X_dev, y_dev = data['X_dev'], data['y_dev']
    X_eval, y_eval = data['X_eval'], data['y_eval']
    feature_names = list(data['feature_names'])

    # TODO: Feature selection logic if needed
    
    print(f"\n📊 Normalizing features...")
    scaler = StandardScaler()
    if len(X_train) > 0:
        X_train_scaled = scaler.fit_transform(X_train)
        X_dev_scaled = scaler.transform(X_dev) if len(X_dev) > 0 else X_dev
        X_eval_scaled = scaler.transform(X_eval) if len(X_eval) > 0 else X_eval
    else:
        print("Warning: Training set empty.")
        X_train_scaled = X_train
        X_dev_scaled = X_dev
        X_eval_scaled = X_eval
        
    # Save final dataset
    np.savez_compressed(
        output_path / 'dataset_final.npz',
        X_train=X_train_scaled,
        y_train=y_train,
        X_dev=X_dev_scaled,
        y_dev=y_dev,
        X_eval=X_eval_scaled,
        y_eval=y_eval,
        feature_names=feature_names
    )
    
    with open(output_path / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    print(f"   ✓ Saved to: {output_path}/dataset_final.npz")

if __name__ == "__main__":
    run_preparation()