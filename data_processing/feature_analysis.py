"""
Step 3: Analyze feature importance using Random Forest
Generate the feature importance graph like in your screenshot
"""
import numpy as np
import pickle
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def plot_feature_importance(importances, feature_names, output_path):
    """
    Plot feature importance bar chart (like your screenshot)
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create colormap (dark to light like your screenshot)
    colors = plt.cm.plasma(np.linspace(0.8, 0.2, len(sorted_features)))
    
    # Plot horizontal bar chart
    bars = plt.barh(range(len(sorted_features)), sorted_importances, color=colors)
    
    # Customize
    plt.yticks(range(len(sorted_features)), sorted_features, fontsize=9)
    plt.xlabel('Importance', fontsize=12)
    plt.title('Feature Importance from Random Forest Classifier', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # Highest importance at top
    
    # Add grid
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved plot: {output_path}/feature_importance.png")
    
    plt.close()

def analyze_features():
    config = load_config()
    output_path = Path(config['dataset']['output_path'])
    
    print("\n" + "="*60)
    print("STEP 3: Feature Importance Analysis")
    print("="*60)
    
    # Load data
    print("\n📂 Loading extracted features...")
    data = np.load(output_path / 'features_raw.npz')
    
    X_train = data['X_train']
    y_train = data['y_train']
    feature_names = data['feature_names']
    
    print(f"   ✓ Loaded: {X_train.shape[0]} training samples")
    print(f"   ✓ Features: {X_train.shape[1]}")
    
    # Normalize features
    print("\n📊 Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train Random Forest for feature importance
    print("\n🌲 Training Random Forest classifier...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    rf.fit(X_train_scaled, y_train)
    
    # Get feature importances
    importances = rf.feature_importances_
    
    print(f"\n✅ Training complete!")
    print(f"   Accuracy: {rf.score(X_train_scaled, y_train):.4f}")
    
    # Print top features
    print(f"\n🏆 Top 10 Most Important Features:")
    indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(indices, 1):
        print(f"   {i}. {feature_names[idx]}: {importances[idx]:.6f}")
    
    # Plot feature importance
    print(f"\n📈 Generating feature importance plot...")
    plot_feature_importance(importances, feature_names, output_path)
    
    # Save feature importances
    importance_data = {
        'feature_names': feature_names,
        'importances': importances,
        'top_10_indices': indices[:10].tolist(),
        'top_10_names': [feature_names[i] for i in indices[:10]],
        'top_10_scores': importances[indices[:10]].tolist()
    }
    
    with open(output_path / 'feature_importances.pkl', 'wb') as f:
        pickle.dump(importance_data, f)
    
    print(f"   ✓ Saved: {output_path}/feature_importances.pkl")
    
    print("\n" + "="*60)
    print("✅ Feature analysis complete!")
    print("="*60)
    print("\n📊 Check feature_importance.png for visualization")
    print("\n Next step: python 04_prepare_final.py")

if __name__ == "__main__":
    analyze_features()