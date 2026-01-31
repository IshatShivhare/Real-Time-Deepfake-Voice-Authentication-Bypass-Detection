"""
Train deepfake detection model with progressive strategy
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from src.utils.config_loader import get_config
from src.models.custom.trainer import ModelTrainer

def plot_training_history(history, save_path):
    """
    Plot training curves
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Train')
        axes[1, 0].plot(history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Train')
        axes[1, 1].plot(history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def get_model_config(config):
    # Map new config structure to what logic expects if needed
    # Our trainer expects a dict with 'model' key containing parameters
    # The new config has 'model_custom'
    return {'model': config['model_custom']}

def progressive_training_strategy(config, X_train, y_train, X_val, y_val):
    print("\n" + "="*60)
    print("🎯 PROGRESSIVE TRAINING STRATEGY")
    print("="*60)
    
    model_cfg = config['model_custom']
    threshold = model_cfg['baseline_threshold']
    
    # Adapt config for trainer
    trainer_config = get_model_config(config)
    
    # Stage 1: Simple CNN baseline
    print("\n" + "="*60)
    print("STAGE 1: Simple CNN Baseline")
    print("="*60)
    
    trainer1 = ModelTrainer(trainer_config)
    
    # Prepare data
    X_train_prep, y_train_prep, X_val_prep, y_val_prep, class_weights = \
        trainer1.prepare_data(X_train, y_train, X_val, y_val)
    
    input_shape = X_train_prep.shape[1:]
    
    # Build simple CNN
    trainer1.build_and_compile(input_shape, 'simple_cnn')
    
    # Train
    history1 = trainer1.train(X_train_prep, y_train_prep, X_val_prep, y_val_prep, class_weights)
    
    # Get best validation accuracy
    best_acc_simple = max(history1.history['val_accuracy'])
    print(f"\n📊 Simple CNN best validation accuracy: {best_acc_simple:.4f}")
    
    if best_acc_simple >= threshold:
        print(f"\n✅ Simple CNN achieves {best_acc_simple:.4f} >= {threshold}")
        return trainer1, history1, 'simple_cnn'
    
    # Stage 2: CNN + GRU
    print(f"\n⚠️  Simple CNN accuracy {best_acc_simple:.4f} < {threshold}")
    print("\n" + "="*60)
    print("STAGE 2: CNN + GRU")
    print("="*60)
    
    trainer2 = ModelTrainer(trainer_config)
    trainer2.build_and_compile(input_shape, 'cnn_gru')
    history2 = trainer2.train(X_train_prep, y_train_prep, X_val_prep, y_val_prep, class_weights)
    
    best_acc_gru = max(history2.history['val_accuracy'])
    print(f"\n📊 CNN + GRU best validation accuracy: {best_acc_gru:.4f}")
    
    if best_acc_gru >= threshold:
        print(f"\n✅ CNN + GRU achieves {best_acc_gru:.4f} >= {threshold}")
        return trainer2, history2, 'cnn_gru'
    
    return trainer2, history2, 'cnn_gru'

def run_training(config=None):
    if config is None:
        config = get_config()
        
    output_path = Path(config['dataset']['output_path'])
    # Correct path for weights is ./weights/custom_model/ usually, but config has it.
    # Config keys have changed. let's respect config.
    # Note: trainer.py uses config['model']['model_save_path'].
    # We should update that in the mapped config.
    
    # Determine save path from config or default
    weights_path = Path(config['model_custom'].get('weights_path', './weights/custom_model/deepfake_detector.h5'))
    model_save_dir = weights_path.parent
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Update config passed to trainer to include model_save_path
    config['model_custom']['model_save_path'] = str(model_save_dir)
    
    print("\n" + "="*60)
    print("STEP 5: Model Training")
    print("="*60)
    
    # Load prepared dataset
    dataset_file = output_path / 'dataset_final.npz'
    if not dataset_file.exists():
        print(f"Dataset not found at {dataset_file}")
        return
        
    print("\n📂 Loading dataset...")
    data = np.load(dataset_file)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_dev']
    y_val = data['y_dev']
    X_test = data['X_eval']
    y_test = data['y_eval']
    
    # Training strategy
    use_progressive = config['model_custom'].get('use_progressive', False)
    
    if use_progressive:
        trainer, history, final_arch = progressive_training_strategy(
            config, X_train, y_train, X_val, y_val
        )
    else:
        trainer_config = get_model_config(config)
        trainer = ModelTrainer(trainer_config)
        architecture = config['model_custom']['architecture']
        
        X_train_prep, y_train_prep, X_val_prep, y_val_prep, class_weights = \
            trainer.prepare_data(X_train, y_train, X_val, y_val)
            
        trainer.build_and_compile(X_train_prep.shape[1:], architecture)
        history = trainer.train(X_train_prep, y_train_prep, X_val_prep, y_val_prep, class_weights)
        final_arch = architecture
    
    # Plot training curves
    plot_training_history(history, model_save_dir)
    
    # Evaluate on test set
    if len(X_test.shape) == 2:
        X_test = np.expand_dims(X_test, axis=-1)
        
    test_metrics = trainer.evaluate(X_test, y_test)
    
    # Save final model
    trainer.save_model(weights_path)
    
    # Save training metadata
    metadata = {
        'architecture': final_arch,
        'training_time_minutes': trainer.training_time / 60,
        'final_metrics': test_metrics,
        'best_val_accuracy': max(history.history['val_accuracy']),
        'config': config['model_custom'],
        'input_shape': X_train.shape[1:]
    }
    
    with open(model_save_dir / 'training_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
        
    print(f"\n✅ Training Completed. Model saved to {weights_path}")

if __name__ == "__main__":
    run_training()