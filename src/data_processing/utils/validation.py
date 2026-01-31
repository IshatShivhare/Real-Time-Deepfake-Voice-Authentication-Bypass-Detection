"""
Data quality validation functions
"""
import numpy as np

def validate_dataset(features, labels, feature_names):
    """
    Validate dataset quality
    Returns: (is_valid, issues)
    """
    issues = []
    
    # Check shape alignment
    if len(features) != len(labels):
        issues.append(f"Shape mismatch: {len(features)} features vs {len(labels)} labels")
    
    # Check for NaN
    nan_count = np.isnan(features).sum()
    if nan_count > 0:
        issues.append(f"Found {nan_count} NaN values")
    
    # Check for Inf
    inf_count = np.isinf(features).sum()
    if inf_count > 0:
        issues.append(f"Found {inf_count} Inf values")
    
    # Check feature dimensions
    if features.shape[1] != len(feature_names):
        issues.append(f"Feature dimension mismatch: {features.shape[1]} vs {len(feature_names)} names")
    
    # Check class balance
    unique, counts = np.unique(labels, return_counts=True)
    class_balance = dict(zip(unique, counts))
    imbalance_ratio = max(counts) / min(counts) if len(counts) > 1 else 1.0
    if imbalance_ratio > 2.0:
        issues.append(f"Class imbalance detected: {class_balance} (ratio: {imbalance_ratio:.2f})")
    
    is_valid = len(issues) == 0
    return is_valid, issues

def print_dataset_stats(features, labels, feature_names, split_name="Dataset"):
    """
    Print comprehensive dataset statistics
    """
    print(f"\n{'='*60}")
    print(f"{split_name} Statistics")
    print(f"{'='*60}")
    
    print(f"\n📊 Dimensions:")
    print(f"   Samples: {len(features)}")
    print(f"   Features: {features.shape[1]}")
    
    print(f"\n🏷️  Class Distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = "Bonafide (Real)" if label == 0 else "Spoof (Fake)"
        percentage = (count / len(labels)) * 100
        print(f"   {label_name}: {count} ({percentage:.1f}%)")
    
    print(f"\n📈 Feature Statistics:")
    print(f"   Min: {features.min():.6f}")
    print(f"   Max: {features.max():.6f}")
    print(f"   Mean: {features.mean():.6f}")
    print(f"   Std: {features.std():.6f}")
    
    print(f"\n✓ Feature Names ({len(feature_names)}):")
    print(f"   {', '.join(feature_names[:10])}...")
    
    # Validate
    is_valid, issues = validate_dataset(features, labels, feature_names)
    if is_valid:
        print(f"\n✅ Dataset validation: PASSED")
    else:
        print(f"\n⚠️  Dataset validation issues:")
        for issue in issues:
            print(f"   - {issue}")
    
    print(f"{'='*60}\n")