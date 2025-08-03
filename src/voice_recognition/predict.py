"""
Prediction module for the voice recognition system.
"""
import os
import argparse
import json
import pathlib
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Optional

from .preprocessing import decode_audio, get_spectrogram

class VoiceCommandRecognizer:
    """A class for recognizing voice commands using a trained model."""
    
    def __init__(self, model_path: str, metadata_path: Optional[str] = None):
        """
        Initialize the voice command recognizer.
        
        Args:
            model_path: Path to the saved model directory or file
            metadata_path: Optional path to the metadata JSON file
        """
        self.model_path = model_path
        self.model = None
        self.metadata = {}
        self.class_names = []
        self.input_shape = None
        
        if metadata_path is None:
            model_dir = os.path.dirname(model_path) if os.path.isfile(model_path) else model_path
            metadata_path = self._find_metadata(model_dir)
        
        if metadata_path and os.path.exists(metadata_path):
            self._load_metadata(metadata_path)
        
        if not self.model:        self._load_model()
        
        if not self.class_names and os.path.exists(os.path.join(os.path.dirname(model_path), 'class_names.npy')):
            try:
                self.class_names = np.load(os.path.join(os.path.dirname(model_path), 'class_names.npy')).tolist()
                print(f"Loaded class names from file: {self.class_names}")
            except Exception as e:
                print(f"Warning: Could not load class names: {e}")
        
        if not self.class_names:
            self.class_names = sorted(['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes'])
            print(f"Warning: Using default class names in alphabetical order: {self.class_names}")
        else:
            print(f"Using class names: {self.class_names}")
            
        if self.class_names != sorted(self.class_names):
            print(f"Warning: Class names are not in alphabetical order. This may cause incorrect predictions.")
            print(f"  Current order: {self.class_names}")
            print(f"  Expected order: {sorted(self.class_names)}")
    
    def _find_metadata(self, model_dir: str) -> Optional[str]:
        """Find the metadata file in the model directory."""
        model_name = os.path.basename(model_dir)
        if model_name.endswith('_metadata.json'):
            return model_dir
        
        possible_paths = [
            os.path.join(model_dir, "metadata.json"),
            os.path.join(model_dir, f"{model_name}_metadata.json"),
            os.path.join(os.path.dirname(model_dir), f"{model_name}_metadata.json")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _load_metadata(self, metadata_path: str) -> None:
        """Load model metadata from a JSON file."""
        try:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            self.class_names = self.metadata.get('class_names', [])
            self.input_shape = tuple(self.metadata.get('input_shape', (124, 129)))
            
            print(f"Loaded metadata for model: {self.metadata.get('model_type', 'unknown')}")
            print(f"Class names: {self.class_names}")
            print(f"Input shape: {self.input_shape}")
            
        except Exception as e:
            print(f"Error loading metadata: {e}")
    
    def _load_model(self) -> None:
        """Load the trained model and try to get class names from it."""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Model loaded from {self.model_path}")
            
            if hasattr(self.model, 'class_names'):
                self.class_names = self.model.class_names
                print(f"Found class names in model: {self.class_names}")
            
            if hasattr(self.model, 'input_shape') and self.model.input_shape[1:]:
                self.input_shape = self.model.input_shape[1:]
                print(f"Input shape: {self.input_shape}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """
        Preprocess an audio file for prediction.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Preprocessed spectrogram as a numpy array
        """
        audio = decode_audio(audio_path)
        
        spectrogram = get_spectrogram(
            audio,
            desired_length=16000,
            frame_length=256,
            frame_step=128
        )
        
        if self.input_shape and spectrogram.shape != self.input_shape:
            if spectrogram.shape[0] > self.input_shape[0]:
                spectrogram = spectrogram[:self.input_shape[0], :]
            if spectrogram.shape[1] > self.input_shape[1]:
                spectrogram = spectrogram[:, :self.input_shape[1]]
            
            if spectrogram.shape[0] < self.input_shape[0] or spectrogram.shape[1] < self.input_shape[1]:
                padded = np.zeros(self.input_shape, dtype=spectrogram.dtype)
                padded[:spectrogram.shape[0], :spectrogram.shape[1]] = spectrogram
                spectrogram = padded
        
        return np.expand_dims(spectrogram, axis=0)
    
    def predict_file(self, audio_path: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Predict the command from an audio file.
        
        Args:
            audio_path: Path to the audio file
            top_k: Number of top predictions to return
            
        Returns:
            List of dictionaries containing class names and their probabilities
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        spectrogram = self.preprocess_audio(audio_path)
        
        predictions = self.model.predict(spectrogram, verbose=0)
        
        top_k = min(top_k, len(self.class_names))
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if hasattr(self, 'class_names') and idx < len(self.class_names):
                class_name = self.class_names[idx]
            else:
                class_name = str(idx)
                
            prob = float(predictions[0][idx])
            results.append({
                'class': class_name,
                'probability': prob,
                'index': int(idx)
            })
        
        return results
    
    def predict_raw_audio(self, audio_data: np.ndarray, sample_rate: int = 16000, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Predict the command from raw audio data.
        
        Args:
            audio_data: 1D numpy array of audio samples
            sample_rate: Sample rate of the audio data
            
        Returns:
            List of dictionaries containing class names and their probabilities
        """
        audio_tensor = tf.convert_to_tensor(audio_data, dtype=tf.float32)
        
        if sample_rate != 16000:
            audio_tensor = tf.audio.resample(
                audio_tensor,
                sample_rate,
                16000
            )
        
        spectrogram = get_spectrogram(audio_tensor)
        
        if self.input_shape and spectrogram.shape != self.input_shape:
            if spectrogram.shape[0] > self.input_shape[0]:
                spectrogram = spectrogram[:self.input_shape[0], :]
            if spectrogram.shape[1] > self.input_shape[1]:
                spectrogram = spectrogram[:, :self.input_shape[1]]
            
            if spectrogram.shape[0] < self.input_shape[0] or spectrogram.shape[1] < self.input_shape[1]:
                padded = np.zeros(self.input_shape, dtype=spectrogram.numpy().dtype)
                padded[:spectrogram.shape[0], :spectrogram.shape[1]] = spectrogram.numpy()
                spectrogram = tf.convert_to_tensor(padded)
        
        spectrogram = tf.expand_dims(spectrogram, axis=0)
        
        predictions = self.model.predict(spectrogram, verbose=0)
        
        top_idx = np.argmax(predictions[0])
        class_name = self.class_names[top_idx] if top_idx < len(self.class_names) else str(top_idx)
        probability = float(predictions[0][top_idx])
        
        return [{
            'class': class_name,
            'probability': probability,
            'index': int(top_idx)
        }]

def record_audio(duration: int = 3, sample_rate: int = 16000) -> Optional[np.ndarray]:
    """
    Record audio from the default microphone.
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate for the recording
        
    Returns:
        NumPy array containing the recorded audio, or None if recording failed
    """
    try:
        import sounddevice as sd
        import soundfile as sf
        
        print(f"Recording for {duration} seconds...")
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        print("Recording finished")
        return audio.flatten()
        
    except ImportError:
        print("Recording requires sounddevice and soundfile packages. Install with:")
        print("pip install sounddevice soundfile")
        return None
    except Exception as e:
        print(f"Error during recording: {e}")
        return None

def main():
    """Command-line interface for voice command recognition."""
    parser = argparse.ArgumentParser(description='Recognize voice commands using a trained model.')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the saved model directory or file')
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--file', type=str, 
                           help='Path to an audio file for prediction')
    input_group.add_argument('--record', action='store_true',
                           help='Record audio from microphone')
    
    parser.add_argument('--metadata', type=str, 
                       help='Path to the metadata JSON file (default: auto-detect)')
    parser.add_argument('--duration', type=int, default=3,
                       help='Recording duration in seconds (default: 3)')
    parser.add_argument('--top-k', type=int, default=3,
                       help='Number of top predictions to show (default: 3)')
    
    args = parser.parse_args()
    
    try:
        recognizer = VoiceCommandRecognizer(args.model, args.metadata)
    except Exception as e:
        print(f"Error initializing recognizer: {e}")
        return
    
    if args.file:
        try:
            print(f"Processing file: {args.file}")
            predictions = recognizer.predict_file(args.file, top_k=args.top_k)
            
            print("\nPredictions:")
            for i, pred in enumerate(predictions, 1):
                print(f"{i}. {pred['class']}: {pred['probability']:.4f}")
                
        except Exception as e:
            print(f"Error processing file: {e}")
    
    elif args.record:
        audio_data = record_audio(duration=args.duration)
        if audio_data is not None:
            try:
                print("Processing recording...")
                predictions = recognizer.predict_raw_audio(audio_data, top_k=3)
                
                print("\nTop predictions:")
                for i, pred in enumerate(predictions, 1):
                    print(f"{i}. {pred['class']}: {pred['probability']:.2%}")
                
                top_prediction = predictions[0]
                print(f"\nPredicted command: {top_prediction['class']} (confidence: {top_prediction['probability']:.2%})")
                
                output_dir = "recordings"
                os.makedirs(output_dir, exist_ok=True)
                
                import soundfile as sf
                from datetime import datetime
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(output_dir, f"recording_{timestamp}.wav")
                sf.write(output_file, audio_data, 16000)
                print(f"Recording saved to {output_file}")
                
            except Exception as e:
                print(f"Error processing recording: {e}")

if __name__ == "__main__":
    main()
