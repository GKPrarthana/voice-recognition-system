import tensorflow as tf
import librosa
import numpy as np

def decode_audio(file_path: tf.Tensor) -> tf.Tensor:
    file_path = tf.strings.as_string(file_path)
    audio_binary = tf.io.read_file(file_path)
    audio, sample_rate = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=-1)
    sample_rate = tf.cast(sample_rate, tf.int32)

    desired_sample_rate = 16000
    def resample():
        audio64 = tf.cast(audio, tf.float32)
        original_length = tf.shape(audio64)[0]
        resample_indices = tf.linspace(0.0, float(original_length - 1), desired_sample_rate)
        return tf.gather(audio64, tf.cast(resample_indices, tf.int32), axis=0)

    audio = tf.cond(sample_rate == desired_sample_rate, lambda: audio, resample)
    return audio

def get_spectrogram(waveform: tf.Tensor) -> tf.Tensor:
    desired_length = 16000
    waveform = waveform[:desired_length]
    zero_padding = tf.zeros([desired_length] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.concat([waveform, zero_padding], 0)
    frame_length = 256
    frame_step = 128
    stft = tf.signal.stft(waveform, frame_length=frame_length, frame_step=frame_step, fft_length=frame_length)
    spectrogram = tf.abs(stft)
    log_spectrogram = tf.math.log(spectrogram + 1e-7)
    return log_spectrogram

def augment_audio(audio: tf.Tensor, sample_rate: int = 16000) -> tf.Tensor:
    audio_np = audio.numpy()
    if np.random.rand() < 0.5:
        audio_np = librosa.effects.pitch_shift(audio_np, sr=sample_rate, n_steps=np.random.uniform(-2, 2))
    if np.random.rand() < 0.5:
        noise = np.random.normal(0, 0.005, audio_np.shape)
        audio_np += noise
    return tf.convert_to_tensor(audio_np, dtype=tf.float32)

def preprocess_dataset(files: list, labels: list) -> tf.data.Dataset:
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)

    def _process(file_path: tf.Tensor, label: tf.Tensor):
        audio = decode_audio(file_path)
        audio = tf.py_function(augment_audio, [audio], tf.float32)
        spec = get_spectrogram(audio)
        return spec, label

    dataset = tf.data.Dataset.zip((files_ds, labels_ds))
    dataset = dataset.map(_process, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset