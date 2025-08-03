import os

directories = [
    "src/voice_recognition/data_ingestion",
    "src/voice_recognition/preprocessing",
    "src/voice_recognition/model",
    "src/voice_recognition/training",
    "config",
    "logs",
    "models",
    "data/raw",
    "data/processed",
    "notebooks"
]
files = {
    "src/__init__.py": "",
    "src/voice_recognition/__init__.py": "",
    "src/voice_recognition/data_ingestion/__init__.py": "",
    "src/voice_recognition/data_ingestion/data_loader.py": "",
    "src/voice_recognition/preprocessing/__init__.py": "",
    "src/voice_recognition/preprocessing/audio_preprocessing.py": "",
    "src/voice_recognition/model/__init__.py": "",
    "src/voice_recognition/model/model_builder.py": "",
    "src/voice_recognition/training/__init__.py": "",
    "src/voice_recognition/training/train.py": "",
    "config/config.yaml": "",
    "config/params.yaml": "",
    "logs/__init__.py": "",
    "main.py": "",
    "requirements.txt": "tensorflow\nnumpy\nlibrosa\npyyaml\nmlflow\nmatplotlib",
    "README.md": "# Voice Recognition Project\n\nA deep learning project for classifying spoken commands using LSTM."
}

def create_project_structure():
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    for file_path, content in files.items():
        with open(file_path, "w") as f:
            f.write(content)
        print(f"Created file: {file_path}")

if __name__ == "__main__":
    create_project_structure()