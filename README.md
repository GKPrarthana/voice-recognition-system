# Voice Command Recognition System

A deep learning-based system for recognizing spoken commands using LSTM and CNN-LSTM models. This project provides a modular and extensible framework for training and deploying voice command recognition models.

## Features

- Support for both LSTM and CNN-LSTM architectures
- Data preprocessing pipeline for audio files
- Training and evaluation scripts
- Real-time prediction from audio files or microphone input
- Model saving and loading functionality
- Configurable hyperparameters

## Project Structure

```
voice-recognition-system/
├── config/                    # Configuration files
│   └── config.py             # Project configuration settings
├── data/                      # Data directory
│   └── raw/                   # Raw audio data
│       └── mini_speech_commands/
│           ├── down/
│           ├── go/
│           ├── left/
│           ├── no/
│           ├── right/
│           ├── stop/
│           ├── up/
│           └── yes/
├── models/                    # Saved models and checkpoints
│   ├── checkpoints/           # Training checkpoints
│   └── logs/                  # TensorBoard logs
├── notebooks/                 # Jupyter notebooks for exploration
│   └── voice-recognition.ipynb
├── src/                       # Source code
│   └── voice_recognition/     # Main package
│       ├── __init__.py
│       ├── data_ingestion.py  # Data loading and preparation
│       ├── preprocessing.py   # Audio preprocessing utilities
│       ├── model.py           # Model architecture definitions
│       ├── train.py           # Training script
│       └── predict.py         # Prediction utilities
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/voice-recognition-system.git
   cd voice-recognition-system
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training a Model

To train a new model:

```bash
python -m voice_recognition.train \
    --data-dir data/raw/mini_speech_commands \
    --model-dir models/command_recognition \
    --model-type lstm \
    --epochs 20 \
    --batch-size 32
```

### Making Predictions

To predict from an audio file:

```bash
python -m voice_recognition.predict \
    --model models/command_recognition \
    --file path/to/audio.wav
```

For real-time prediction using your microphone:

```bash
python -m voice_recognition.predict \
    --model models/command_recognition \
    --record
```

## Dataset

This project uses the [Speech Commands Dataset](https://www.tensorflow.org/datasets/catalog/speech_commands) by default. Place the dataset in the `data/raw/mini_speech_commands` directory with the following structure:

```
data/raw/mini_speech_commands/
├── down/
├── go/
├── left/
├── no/
├── right/
├── stop/
├── up/
└── yes/
```

## Customization

### Adding New Commands

1. Create a new directory for each command in the `data/raw/mini_speech_commands` directory
2. Add your audio files (WAV format) to the respective directories
3. Retrain the model with the updated dataset

### Model Architecture

You can choose between different model architectures using the `--model-type` argument:

- `lstm`: Standard LSTM model
- `cnn_lstm`: CNN-LSTM hybrid model

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [TensorFlow Speech Commands Dataset](https://www.tensorflow.org/datasets/catalog/speech_commands)
- [TensorFlow Audio Recognition Tutorial](https://www.tensorflow.org/tutorials/audio/simple_audio)