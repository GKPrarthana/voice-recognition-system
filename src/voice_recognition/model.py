"""
Model definition for the voice recognition system.
"""
from typing import Tuple, Optional, Dict, Any
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

def build_lstm_model(
    input_shape: Tuple[int, int], 
    num_classes: int,
    lstm_units: int = 128,
    dense_units: int = 64,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
    metrics: Optional[list] = None
) -> tf.keras.Model:
    """
    Build an LSTM model for speech command recognition.
    
    Args:
        input_shape: Shape of input features (time_steps, num_features)
        num_classes: Number of output classes
        lstm_units: Number of units in LSTM layer
        dense_units: Number of units in dense layer
        dropout_rate: Dropout rate
        learning_rate: Learning rate for optimizer
        metrics: List of metrics to track during training
        
    Returns:
        Compiled Keras model
    """
    """
    Build an LSTM-based model for speech command recognition.
    
    Args:
        input_shape: Shape of the input spectrograms (time_steps, freq_bins)
        num_classes: Number of output classes
        lstm_units: Number of units in the LSTM layer
        dense_units: Number of units in the dense layer
        dropout_rate: Dropout rate
        learning_rate: Learning rate for the optimizer
        metrics: List of metrics to track during training
        
    Returns:
        Compiled Keras model
    """
    if metrics is None:
        metrics = ['accuracy']
    
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Reshape(input_shape + (1,))(inputs)
    
    x = layers.BatchNormalization()(x)
    
    x = layers.Reshape((input_shape[0], -1))(x)
    
    x = layers.LSTM(lstm_units, return_sequences=False)(x)
    
    x = layers.Dense(dense_units, activation='relu')(x)
    
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=metrics
    )
    
    return model

def build_cnn_lstm_model(
    input_shape: Tuple[int, int], 
    num_classes: int,
    conv_filters: list = [32, 64],
    kernel_sizes: list = [3, 3],
    pool_sizes: list = [2, 2],
    lstm_units: int = 128,
    dense_units: int = 64,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
    metrics: Optional[list] = None
) -> tf.keras.Model:
    """
    Build a CNN-LSTM model for speech command recognition.
    
    Args:
        input_shape: Shape of input features (time_steps, num_features)
        num_classes: Number of output classes
        conv_filters: List of number of filters for each Conv1D layer
        kernel_sizes: List of kernel sizes for each Conv1D layer
        pool_sizes: List of pool sizes for each MaxPooling1D layer
        lstm_units: Number of units in LSTM layer
        dense_units: Number of units in dense layer
        dropout_rate: Dropout rate
        learning_rate: Learning rate for optimizer
        metrics: List of metrics to track during training
        
    Returns:
        Compiled Keras model
    """
    """
    Build a CNN-LSTM model for speech command recognition.
    
    Args:
        input_shape: Shape of the input spectrograms (time_steps, freq_bins)
        num_classes: Number of output classes
        conv_filters: List of number of filters for each Conv1D layer
        kernel_sizes: List of kernel sizes for each Conv1D layer
        pool_sizes: List of pool sizes for each MaxPooling1D layer
        lstm_units: Number of units in the LSTM layer
        dense_units: Number of units in the dense layer
        dropout_rate: Dropout rate
        learning_rate: Learning rate for the optimizer
        metrics: List of metrics to track during training
        
    Returns:
        Compiled Keras model
    """
    if metrics is None:
        metrics = ['accuracy']
    
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Reshape(input_shape + (1,))(inputs)
    
    for i, (filters, kernel_size, pool_size) in enumerate(zip(conv_filters, kernel_sizes, pool_sizes)):
        x = layers.Conv2D(
            filters=filters,
            kernel_size=(kernel_size, 1),
            padding='same',
            name=f'conv2d_{i}'
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(pool_size, 1), name=f'maxpool_{i}')(x)
    
    x = layers.Reshape((x.shape[1], -1))(x)
    
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False))(x)
    
    x = layers.Dense(dense_units, activation='relu')(x)
    
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=metrics
    )
    
    return model

def save_model(
    model: tf.keras.Model, 
    model_dir: str, 
    model_name: str = "voice_recognition_model"
) -> None:
    """
    Save the model to disk.
    
    Args:
        model: Trained Keras model
        model_dir: Directory to save the model
        model_name: Name of the model
    """
    import os
    import json
       
    model_path = os.path.join(model_dir, model_name)
    model.save(model_path)
    
    print(f"Model saved to {model_path}")

def load_model(
    model_path: str,
    custom_objects: Optional[Dict[str, Any]] = None
) -> tf.keras.Model:
    """
    Load a saved model from disk.
    
    Args:
        model_path: Path to the saved model
        custom_objects: Dictionary of custom objects needed for loading
        
    Returns:
        Loaded Keras model
    """
    if custom_objects is None:
        custom_objects = {}
    
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

if __name__ == "__main__":
    input_shape = (124, 129) 
    num_classes = 8  
    
    print("Testing LSTM model...")
    lstm_model = build_lstm_model(input_shape, num_classes)
    lstm_model.summary()
    
    print("\nTesting CNN-LSTM model...")
    cnn_lstm_model = build_cnn_lstm_model(input_shape, num_classes)
    cnn_lstm_model.summary()
