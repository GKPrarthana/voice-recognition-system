"""
Data ingestion module for loading and preparing the speech commands dataset.
"""
import os
import pathlib
from typing import List, Tuple, Optional
import numpy as np
import tensorflow as tf

def list_commands(data_dir) -> np.ndarray:
    """
    List all command directories in the dataset directory.
    
    Args:
        data_dir: Path to the dataset directory (str or pathlib.Path)
        
    Returns:
        Sorted array of command names
    """

    data_path = pathlib.Path(data_dir) if isinstance(data_dir, str) else data_dir
    
    commands = np.array(
        [
            name
            for name in os.listdir(data_path)
            if (data_path / name).is_dir() and not name.startswith(".")
        ]
    )
    
    if len(commands) == 0:
        raise ValueError(f"No valid command directories found in {data_path}")
        
    commands.sort()
    print(f"Found command directories: {commands}")
    return commands

def load_dataset_files(data_dir) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load audio file paths and their corresponding labels.
    
    Args:
        data_dir: Path to the dataset directory (str or pathlib.Path)
        
    Returns:
        Tuple of (files, labels, class_names)
    """
    data_path = pathlib.Path(data_dir) if isinstance(data_dir, str) else data_dir
    
    class_names = list_commands(data_path)
    files = []
    labels = []
    
    for label, command in enumerate(class_names):
        command_dir = data_path / command
        command_files = [
            str(command_dir / fname) 
            for fname in os.listdir(command_dir) 
            if fname.lower().endswith('.wav')
        ]
        if not command_files:
            print(f"Warning: No WAV files found in {command_dir}")
        files.extend(command_files)
        labels.extend([label] * len(command_files))
    
    if not files:
        raise ValueError(f"No WAV files found in dataset directory {data_path}")
    
    print(f"Loaded {len(files)} audio files for {len(class_names)} commands")
    return np.array(files), np.array(labels), class_names

def split_dataset(
    files: np.ndarray, 
    labels: np.ndarray, 
    validation_split: float = 0.2,
    test_split: float = 0.1,
    seed: int = 123
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into training, validation, and test sets.
    
    Args:
        files: Array of file paths
        labels: Array of labels
        validation_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_files, train_labels, val_files, val_labels, test_files, test_labels)
    """
    num_val = int(len(files) * validation_split)
    num_test = int(len(files) * test_split)
    num_train = len(files) - num_val - num_test
    
    rng = np.random.default_rng(seed=seed)
    indices = np.arange(len(files))
    rng.shuffle(indices)
    
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]
    
    train_files = files[train_indices]
    train_labels = labels[train_indices]
    
    val_files = files[val_indices]
    val_labels = labels[val_indices]
    
    test_files = files[test_indices]
    test_labels = labels[test_indices]
    
    return train_files, train_labels, val_files, val_labels, test_files, test_labels

if __name__ == "__main__":
    data_dir = pathlib.Path("data/raw/mini_speech_commands")
    files, labels, class_names = load_dataset_files(data_dir)
    print(f"Total files: {len(files)}")
    print(f"Class names: {class_names}")
    
    train_files, train_labels, val_files, val_labels, test_files, test_labels = split_dataset(
        files, labels
    )
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    print(f"Test samples: {len(test_files)}")
