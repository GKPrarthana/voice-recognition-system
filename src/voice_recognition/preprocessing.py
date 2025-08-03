"""
Audio preprocessing module for the voice recognition system.
"""
from typing import Tuple, Union, List, Optional
import tensorflow as tf
import numpy as np

def decode_audio(file_path: Union[str, tf.Tensor], target_sample_rate: int = 16000) -> tf.Tensor:
    """
    Read and decode a WAV file to a 1D float tensor and resample if needed.
    
    Args:
        file_path: Path to the audio file or a tensor containing the path
        target_sample_rate: Target sample rate in Hz
        
    Returns:
        Tensor containing the audio waveform
    """
    if isinstance(file_path, tf.Tensor):
        file_path = tf.strings.as_string(file_path)
    
    audio_binary = tf.io.read_file(file_path)
    audio, sample_rate = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=-1)
    sample_rate = tf.cast(sample_rate, tf.int32)
    
    if sample_rate != target_sample_rate:
        audio = tf.cast(audio, tf.float32)
        original_length = tf.shape(audio)[0]
        resample_indices = tf.linspace(
            0.0, 
            float(original_length - 1), 
            target_sample_rate
        )
        audio = tf.gather(audio, tf.cast(resample_indices, tf.int32), axis=0)
    
    return audio

def get_spectrogram(
    waveform: tf.Tensor, 
    desired_length: int = 16000,
    frame_length: int = 256,
    frame_step: int = 128,
    fft_length: Optional[int] = None
) -> tf.Tensor:
    """
    Convert a waveform into a log-magnitude spectrogram.
    
    Args:
        waveform: 1D tensor containing the audio waveform
        desired_length: Desired length of the waveform in samples
        frame_length: Length of each frame in samples
        frame_step: Number of samples between frames
        fft_length: Size of the FFT to apply
        
    Returns:
        Log-magnitude spectrogram with shape (time_steps, freq_bins)
    """
    waveform = tf.cast(waveform, tf.float32)
    
    waveform = waveform[:desired_length]
    
    padding = tf.maximum(desired_length - tf.shape(waveform)[0], 0)
    waveform = tf.pad(waveform, [[0, padding]])
    
    stft = tf.signal.stft(
        waveform,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length or frame_length,
        window_fn=tf.signal.hann_window,
        pad_end=True
    )
    
    spectrogram = tf.abs(stft)
    
    spectrogram = tf.math.pow(spectrogram, 0.3)
    
    spectrogram = (spectrogram - tf.reduce_min(spectrogram)) / \
                 (tf.reduce_max(spectrogram) - tf.reduce_min(spectrogram) + 1e-10)
    
    spectrogram = tf.transpose(spectrogram, perm=[1, 0])
    
    return spectrogram

def create_tf_dataset(
    files: List[str], 
    labels: List[int], 
    batch_size: int = 32,
    shuffle: bool = True,
    buffer_size: int = 1000,
    prefetch: bool = True,
    cache: bool = False,
    seed: Optional[int] = None
) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset from file paths and labels.
    
    Args:
        files: List of file paths
        labels: List of labels
        batch_size: Batch size
        shuffle: Whether to shuffle the dataset
        buffer_size: Buffer size for shuffling
        prefetch: Whether to prefetch batches
        cache: Whether to cache the dataset in memory
        seed: Random seed for shuffling
        
    Returns:
        TensorFlow Dataset yielding (spectrogram, label) pairs
    """
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    
    dataset = tf.data.Dataset.zip((files_ds, labels_ds))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size, seed=seed)
    
    def process_path(file_path, label):
        audio = decode_audio(file_path)
        spectrogram = get_spectrogram(audio)
        return spectrogram, label
    
    dataset = dataset.map(
        process_path, 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    if cache:
        dataset = dataset.cache()
    
    dataset = dataset.batch(batch_size)
    
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

if __name__ == "__main__":
    import pathlib
    
    data_dir = pathlib.Path("data/raw/mini_speech_commands")
    test_file = next(data_dir.glob("**/*.wav"))
    
    print(f"Testing with file: {test_file}")
    
    audio = decode_audio(str(test_file))
    print(f"Audio shape: {audio.shape}, dtype: {audio.dtype}")
    
    spectrogram = get_spectrogram(audio)
    print(f"Spectrogram shape: {spectrogram.shape}, dtype: {spectrogram.dtype}")
