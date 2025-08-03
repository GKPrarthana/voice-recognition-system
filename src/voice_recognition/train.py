"""
Training script for the voice recognition model.
"""
import os
import argparse
import json
import pathlib
from typing import Dict, Any, Optional, Tuple
import numpy as np
import tensorflow as tf
from datetime import datetime

from .data_ingestion import load_dataset_files, split_dataset
from .preprocessing import create_tf_dataset
from .model import build_lstm_model, build_cnn_lstm_model, save_model

def setup_gpu() -> None:
    """Configure GPU settings if available."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU available. Training on CPU.")

def train(
    data_dir: str,
    model_dir: str,
    model_type: str = "lstm",
    batch_size: int = 32,
    epochs: int = 20,
    validation_split: float = 0.2,
    test_split: float = 0.1,
    learning_rate: float = 0.001,
    seed: int = 42,
    use_augmentation: bool = False,
    early_stopping: bool = True,
    tensorboard_logs: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train a voice recognition model.
    
    Args:
        data_dir: Path to the dataset directory
        model_dir: Directory to save the trained model
        model_type: Type of model to train ('lstm' or 'cnn_lstm')
        batch_size: Batch size for training
        epochs: Number of training epochs
        validation_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        learning_rate: Learning rate for the optimizer
        seed: Random seed for reproducibility
        use_augmentation: Whether to use data augmentation
        early_stopping: Whether to use early stopping
        tensorboard_logs: Directory to save TensorBoard logs
        
    Returns:
        Dictionary containing training history and metadata
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    setup_gpu()
    
    data_path = pathlib.Path(data_dir)
    files, labels, class_names = load_dataset_files(data_path)
    
    train_files, train_labels, val_files, val_labels, test_files, test_labels = split_dataset(
        files, labels, 
        validation_split=validation_split,
        test_split=test_split,
        seed=seed
    )
    
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    print(f"Test samples: {len(test_files)}")
    print(f"Class names: {class_names}")
    
    train_ds = create_tf_dataset(
        train_files, 
        train_labels, 
        batch_size=batch_size,
        shuffle=True,
        buffer_size=len(train_files)
    )
    
    val_ds = create_tf_dataset(
        val_files, 
        val_labels, 
        batch_size=batch_size,
        shuffle=False
    )
    
    for spectrogram, _ in train_ds.take(1):
        input_shape = spectrogram.shape[1:]
    
    print(f"Input shape: {input_shape}")
    
    if model_type == "lstm":
        model = build_lstm_model(
            input_shape=input_shape,
            num_classes=len(class_names),
            learning_rate=learning_rate
        )
    elif model_type == "cnn_lstm":
        model = build_cnn_lstm_model(
            input_shape=input_shape,
            num_classes=len(class_names),
            learning_rate=learning_rate
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.summary()
    
    callbacks = []
    
    if early_stopping:
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping_cb)
    
    if tensorboard_logs:
        log_dir = os.path.join(
            tensorboard_logs, 
            f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        tensorboard_cb = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
        callbacks.append(tensorboard_cb)
        print(f"TensorBoard logs: {log_dir}")
    
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.ckpt")
    
    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    callbacks.append(model_checkpoint_cb)
    
    print(f"Training {model_type.upper()} model for {epochs} epochs...")
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )
    
    model.load_weights(checkpoint_path)
    
    test_ds = create_tf_dataset(
        test_files, 
        test_labels, 
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    model_name = f"voice_recognition_{model_type}"
    save_model(model, model_dir, model_name)
    
    metadata = {
        'model_type': model_type,
        'input_shape': input_shape,
        'num_classes': len(class_names),
        'class_names': class_names.tolist(),
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'seed': seed,
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'training_date': datetime.now().isoformat(),
        'dataset_size': {
            'train': len(train_files),
            'validation': len(val_files),
            'test': len(test_files)
        }
    }
    
    metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Training complete. Model and metadata saved to {model_dir}")
    
    return {
        'model': model,
        'history': history.history,
        'metadata': metadata,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss
    }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a voice recognition model.')
    
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory to save the trained model')
    
    parser.add_argument('--model-type', type=str, default='lstm',
                        choices=['lstm', 'cnn_lstm'],
                        help='Type of model to train')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate for the optimizer')
    
    parser.add_argument('--validation-split', type=float, default=0.2,
                        help='Fraction of data to use for validation')
    parser.add_argument('--test-split', type=float, default=0.1,
                        help='Fraction of data to use for testing')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--no-early-stopping', action='store_false', dest='early_stopping',
                        help='Disable early stopping')
    parser.add_argument('--tensorboard-logs', type=str, default=None,
                        help='Directory to save TensorBoard logs')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    os.makedirs(args.model_dir, exist_ok=True)
    
    train(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        model_type=args.model_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.validation_split,
        test_split=args.test_split,
        learning_rate=args.learning_rate,
        seed=args.seed,
        early_stopping=args.early_stopping,
        tensorboard_logs=args.tensorboard_logs
    )
