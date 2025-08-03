"""
Configuration settings for the voice recognition system.
"""
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw' / 'mini_speech_commands'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Model paths
MODELS_DIR = BASE_DIR / 'models'
CHECKPOINTS_DIR = MODELS_DIR / 'checkpoints'
TENSORBOARD_LOGS_DIR = MODELS_DIR / 'logs'

# Audio settings
SAMPLE_RATE = 16000  # Hz
DURATION = 1.0  # seconds
HOP_LENGTH = 128  # samples
N_FFT = 256  # FFT window size
N_MELS = 128  # Number of Mel bands
FMAX = 8000  # Maximum frequency (Hz)

# Model settings
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_TEST_SPLIT = 0.1

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, CHECKPOINTS_DIR, TENSORBOARD_LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
