# model/trainer.py
"""
Training logic with progressive strategy
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
import time
from pathlib import Path

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None
        self.training_time = 0
        
        # GPU Configuration
        if not self.config.get('app', {}).get('use_gpu', False):
            # Disable GPU for TensorFlow
            tf.config.set_visible_devices([], 'GPU')
            print("🚫 GPU disabled via config (using CPU for training)")
        else:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"🚀 Using GPU: {gpus[0]}")
            else:
                print("⚠️  GPU enabled in config but no GPU found. Using CPU.")
        
    def prepare_data(self, X_train, y_train, X_val, y_val):
        """
        Prepare data for training
        """
        print("\n📊 Preparing data...")
        
        # Reshape if needed (add channel dimension for Conv1D)
        if len(X_train.shape) == 2:
            # Shape: (samples, features) → (samples, timesteps, features)
            # Treat each feature as a timestep
            X_train = np.expand_dims(X_train, axis=-1)
            X_val = np.expand_dims(X_val, axis=-1)
            print(f"   Reshaped data: {X_train.shape}")
        
        # Calculate class weights for imbalanced data
        class_weights = None
        if self.config['model']['use_class_weights']:
            classes = np.unique(y_train)
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weights = dict(zip(classes, weights))
            print(f"   Class weights: {class_weights}")
        
        return X_train, y_train, X_val, y_val, class_weights
    
    def build_and_compile(self, input_shape, architecture='cnn_gru'):
        """
        Build and compile model
        """
        from .architecture import get_model
        
        print(f"\n🏗️  Building {architecture.upper()} model...")
        
        self.model = get_model(architecture, input_shape, self.config['model'])
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.config['model']['learning_rate']
            ),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        print("\n📋 Model Summary:")
        self.model.summary()
        
        # Count parameters
        trainable_params = np.sum([np.prod(v.shape) for v in self.model.trainable_weights])
        print(f"\n📊 Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def get_callbacks(self, model_save_path):
        """
        Create training callbacks
        """
        callbacks = []
        
        # Early stopping
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['model']['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ))
        
        # Reduce learning rate on plateau
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.config['model']['reduce_lr_patience'],
            min_lr=1e-6,
            verbose=1
        ))
        
        # Model checkpoint
        checkpoint_path = Path(model_save_path) / 'best_model.h5'
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        callbacks.append(keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=self.config['model']['save_best_only'],
            verbose=1
        ))
        
        # TensorBoard (optional but useful)
        log_dir = Path(model_save_path) / 'logs'
        callbacks.append(keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=0
        ))
        
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, class_weights=None):
        """
        Train the model
        """
        print("\n" + "="*60)
        print("🚀 Starting Training")
        print("="*60)
        
        start_time = time.time()
        
        callbacks = self.get_callbacks(self.config['model']['model_save_path'])
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['model']['epochs'],
            batch_size=self.config['model']['batch_size'],
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        self.training_time = time.time() - start_time
        
        print(f"\n✅ Training complete in {self.training_time/60:.2f} minutes")
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set
        """
        print("\n" + "="*60)
        print("📊 Evaluating Model")
        print("="*60)
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        metrics = dict(zip(self.model.metrics_names, results))
        
        print(f"\n🎯 Test Set Performance:")
        for metric, value in metrics.items():
            print(f"   {metric.capitalize()}: {value:.4f}")
        
        return metrics
    
    def save_model(self, path=None):
        """
        Save trained model
        """
        if path is None:
            path = Path(self.config['model']['model_save_path']) / 'final_model.h5'
        
        self.model.save(path)
        print(f"\n💾 Model saved: {path}")