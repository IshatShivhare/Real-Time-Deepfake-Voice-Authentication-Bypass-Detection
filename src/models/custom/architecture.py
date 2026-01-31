# model/architecture.py
"""
Model architecture definitions
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

def build_cnn_gru(input_shape, config):
    """
    CNN + GRU architecture (RECOMMENDED)
    Fast, effective, real-time capable
    """
    inputs = layers.Input(shape=input_shape, name='input')
    
    x = inputs
    
    # CNN layers for local pattern extraction
    for i, filters in enumerate(config['conv_filters']):
        x = layers.Conv1D(
            filters=filters,
            kernel_size=config['kernel_size'],
            padding='same',
            activation='relu',
            name=f'conv1d_{i+1}'
        )(x)
        x = layers.MaxPooling1D(
            pool_size=config['pool_size'],
            name=f'maxpool_{i+1}'
        )(x)
        x = layers.Dropout(config['dropout_rnn'], name=f'dropout_conv_{i+1}')(x)
    
    # GRU layer for temporal modeling
    x = layers.GRU(
        units=config['rnn_units'],
        return_sequences=False,  # Only final output
        name='gru'
    )(x)
    
    x = layers.Dropout(config['dropout_rate'], name='dropout_gru')(x)
    
    # Dense layers
    for i, units in enumerate(config['dense_units']):
        x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
        x = layers.Dropout(config['dropout_rate'], name=f'dropout_dense_{i+1}')(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='CNN_GRU')
    
    return model

def build_cnn_lstm(input_shape, config):
    """
    CNN + LSTM architecture (Alternative)
    Slightly slower than GRU but sometimes better performance
    """
    inputs = layers.Input(shape=input_shape)
    
    x = inputs
    
    # CNN layers
    for i, filters in enumerate(config['conv_filters']):
        x = layers.Conv1D(filters=filters, kernel_size=config['kernel_size'],
                         padding='same', activation='relu')(x)
        x = layers.MaxPooling1D(pool_size=config['pool_size'])(x)
        x = layers.Dropout(config['dropout_rnn'])(x)
    
    # LSTM layer
    x = layers.LSTM(units=config['rnn_units'], return_sequences=False)(x)
    x = layers.Dropout(config['dropout_rate'])(x)
    
    # Dense layers
    for units in config['dense_units']:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(config['dropout_rate'])(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='CNN_LSTM')
    return model

def build_simple_cnn(input_shape, config):
    """
    Simple CNN baseline (FASTEST)
    Use this first to see if complexity is needed
    """
    inputs = layers.Input(shape=input_shape)
    
    x = inputs
    
    # CNN layers only
    for i, filters in enumerate(config['conv_filters']):
        x = layers.Conv1D(filters=filters, kernel_size=config['kernel_size'],
                         padding='same', activation='relu')(x)
        x = layers.MaxPooling1D(pool_size=config['pool_size'])(x)
        x = layers.Dropout(config['dropout_rnn'])(x)
    
    # Global pooling instead of RNN
    x = layers.GlobalMaxPooling1D()(x)
    
    # Dense layers
    for units in config['dense_units']:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(config['dropout_rate'])(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='Simple_CNN')
    return model

def build_cnn_bilstm(input_shape, config):
    """
    CNN + BiLSTM (BEST ACCURACY but SLOWEST)
    Only use if simple models don't work
    """
    inputs = layers.Input(shape=input_shape)
    
    x = inputs
    
    # CNN layers
    for i, filters in enumerate(config['conv_filters']):
        x = layers.Conv1D(filters=filters, kernel_size=config['kernel_size'],
                         padding='same', activation='relu')(x)
        x = layers.MaxPooling1D(pool_size=config['pool_size'])(x)
        x = layers.Dropout(config['dropout_rnn'])(x)
    
    # BiLSTM layer
    x = layers.Bidirectional(
        layers.LSTM(units=config['rnn_units'], return_sequences=False)
    )(x)
    x = layers.Dropout(config['dropout_rate'])(x)
    
    # Dense layers
    for units in config['dense_units']:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(config['dropout_rate'])(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='CNN_BiLSTM')
    return model

def get_model(architecture_name, input_shape, config):
    """
    Factory function to get model by name
    """
    architectures = {
        'cnn_gru': build_cnn_gru,
        'cnn_lstm': build_cnn_lstm,
        'cnn_bilstm': build_cnn_bilstm,
        'simple_cnn': build_simple_cnn,
    }
    
    if architecture_name not in architectures:
        raise ValueError(f"Unknown architecture: {architecture_name}")
    
    return architectures[architecture_name](input_shape, config)