import os
import argparse
import json
import tensorflow as tf
import numpy as np
from voice_recognition.data_ingestion import load_dataset_files, split_dataset
from voice_recognition.preprocessing import create_tf_dataset, decode_audio, get_spectrogram
from voice_recognition.model import build_lstm_model, build_cnn_lstm_model

def predict_custom_audio(model, audio_path: str, class_names):
    """Predict the class of a custom audio file."""
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    audio = decode_audio(audio_path)
    spec = get_spectrogram(audio)
    spec = tf.expand_dims(spec, 0)  # Add batch dimension
    
    predictions = model.predict(spec)
    predicted_index = tf.argmax(predictions[0]).numpy()
    
    print(f"Prediction probabilities: {dict(zip(class_names, predictions[0]))}")
    return class_names[predicted_index]

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate voice command recognition model.')
    parser.add_argument('--data-dir', type=str, default='data/raw/mini_speech_commands',
                        help='Path to the dataset directory')
    parser.add_argument('--model-dir', type=str, default='models/command_recognition',
                        help='Directory to save the trained model')
    parser.add_argument('--model-type', type=str, default='lstm',
                        choices=['lstm', 'cnn_lstm'],
                        help='Type of model to train')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    
    args = parser.parse_args()
    
    os.makedirs(args.model_dir, exist_ok=True)
    
    print("Loading dataset...")
    files, labels, class_names = load_dataset_files(args.data_dir)
    train_files, train_labels, val_files, val_labels, test_files, test_labels = split_dataset(
        files, labels, validation_split=0.2, test_split=0.1
    )
    
    print("Creating datasets...")
    train_ds = create_tf_dataset(train_files, train_labels, batch_size=args.batch_size)
    val_ds = create_tf_dataset(val_files, val_labels, batch_size=args.batch_size, shuffle=False)
    
    for spectrograms, _ in train_ds.take(1):
        input_shape = spectrograms.shape[1:]
    
    print(f"Building {args.model_type.upper()} model...")
    if args.model_type == 'lstm':
        model = build_lstm_model(input_shape, len(class_names))
    else:  
        model = build_cnn_lstm_model(input_shape, len(class_names))
    
    os.makedirs(args.model_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.model_dir, 'model.keras')
    model_path = os.path.join(args.model_dir, 'model.keras')
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=False,  
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    np.save(os.path.join(args.model_dir, 'class_names.npy'), class_names)
    
    model.class_names = class_names.tolist() if hasattr(class_names, 'tolist') else list(class_names)
    model.save(model_path, save_format='keras')
    
    input_shape_list = input_shape.as_list() if hasattr(input_shape, 'as_list') else list(input_shape)
    metadata = {
        'class_names': model.class_names,
        'input_shape': input_shape_list,
        'model_type': args.model_type
    }
    with open(os.path.join(args.model_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModel saved to {model_path}")
    print("Class names:", class_names)
    
    test_files = [f for f in files if 'down' in f and '0a9f9af7_nohash_1.wav' in f]
    if test_files:
        test_file = test_files[0]
        print(f"\nTesting with file: {test_file}")
        prediction = predict_custom_audio(model, test_file, class_names)
        print(f"Predicted command: {prediction}")

if __name__ == "__main__":
    main()