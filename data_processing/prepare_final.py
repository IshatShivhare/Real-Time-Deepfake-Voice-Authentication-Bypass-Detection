"""
Step 4: Prepare final dataset with normalization and optional feature selection
"""
import numpy as np
import pickle
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import time

from utils import print_dataset_stats, validate_dataset

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def prepare_final_dataset():
    config = load_config()
    output_path = Path(config['dataset']['output_path'])
    
    print("\n" + "="*60)
    print("STEP 4: Final Dataset Preparation")
    print("="*60)
    
    # Load raw features
    print("\n📂 Loading raw features...")
    data = np.load(output_path / 'features_raw.npz')
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_dev = data['X_dev']
    y_dev = data['y_dev']
    X_eval = data['X_eval']
    y_eval = data['y_eval']
    feature_names = list(data['feature_names'])
    
    print(f"   ✓ Train: {X_train.shape}")
    print(f"   ✓ Dev: {X_dev.shape}")
    print(f"   ✓ Eval: {X_eval.shape}")
    
    # Optional: Select top N features
    top_n = config['feature_selection'].get('top_n_features')
    
    if top_n and top_n < len(feature_names):
        print(f"\n🎯 Selecting top {top_n} features...")
        
        # Load feature importances
        with open(output_path / 'feature_importances.pkl', 'rb') as f:
            importance_data = pickle.load(f)
        
        importances = importance_data['importances']
        top_indices = np.argsort(importances)[::-1][:top_n]
        
        # Select features
        X_train = X_train[:, top_indices]
        X_dev = X_dev[:, top_indices]
        X_eval = X_eval[:, top_indices]
        feature_names = [feature_names[i] for i in top_indices]
        
        print(f"   ✓ Selected features: {feature_names[:5]}...")
    
    # Normalize features
    print(f"\n📊 Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)
    X_eval_scaled = scaler.transform(X_eval)
    
    print(f"   ✓ Normalization complete")
    print(f"   Mean: {X_train_scaled.mean():.6f}")
    print(f"   Std: {X_train_scaled.std():.6f}")
    
    # Validate datasets
    print(f"\n✅ Validating datasets...")
    
    for name, X, y in [('Train', X_train_scaled, y_train),
                        ('Dev', X_dev_scaled, y_dev),
                        ('Eval', X_eval_scaled, y_eval)]:
        is_valid, issues = validate_dataset(X, y, feature_names)
        if not is_valid:
            print(f"   ⚠️  {name} validation issues:")
            for issue in issues:
                print(f"      - {issue}")
        else:
            print(f"   ✓ {name}: PASSED")
    
    # Print final stats
    print_dataset_stats(X_train_scaled, y_train, feature_names, "FINAL TRAIN")
    print_dataset_stats(X_dev_scaled, y_dev, feature_names, "FINAL DEV")
    print_dataset_stats(X_eval_scaled, y_eval, feature_names, "FINAL EVAL")
    
    # Save final dataset
    print(f"\n💾 Saving final dataset...")
    
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
    
    # Save scaler (CRITICAL for inference)
    with open(output_path / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save comprehensive metadata
    metadata = {
        'dataset_name': config['dataset']['name'],
        'preparation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'scaler': 'StandardScaler',
        'config': config,
        'splits': {
            'train': {'samples': len(X_train_scaled), 'real': int((y_train==0).sum()), 'fake': int((y_train==1).sum())},
            'dev': {'samples': len(X_dev_scaled), 'real': int((y_dev==0).sum()), 'fake': int((y_dev==1).sum())},
            'eval': {'samples': len(X_eval_scaled), 'real': int((y_eval==0).sum()), 'fake': int((y_eval==1).sum())},
        },
        'feature_statistics': {
            'min': float(X_train_scaled.min()),
            'max': float(X_train_scaled.max()),
            'mean': float(X_train_scaled.mean()),
            'std': float(X_train_scaled.std())
        }
    }
    
    with open(output_path / 'metadata_final.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"   ✓ Dataset: {output_path}/dataset_final.npz")
    print(f"   ✓ Scaler: {output_path}/scaler.pkl")
    print(f"   ✓ Metadata: {output_path}/metadata_final.pkl")
    
    # Summary
    print("\n" + "="*60)
    print("✅ ETL PIPELINE COMPLETE!")
    print("="*60)
    print(f"\n📦 Final Dataset Summary:")
    print(f"   Total samples: {len(X_train_scaled) + len(X_dev_scaled) + len(X_eval_scaled)}")
    print(f"   Feature dimensions: {len(feature_names)}")
    print(f"   Train/Dev/Eval split: {len(X_train_scaled)}/{len(X_dev_scaled)}/{len(X_eval_scaled)}")
    print(f"\n🎯 Dataset ready for model training!")
    print(f"   Files to share with team:")
    print(f"   - dataset_final.npz")
    print(f"   - scaler.pkl")
    print(f"   - metadata_final.pkl")
    print(f"   - feature_importance.png")

if __name__ == "__main__":
    prepare_final_dataset()