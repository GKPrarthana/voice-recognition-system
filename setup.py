from setuptools import setup, find_packages

setup(
    name="voice_recognition",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "tensorflow>=2.8.0",
        "numpy>=1.19.5",
        "librosa>=0.9.2",
        "soundfile>=0.10.3",
        "sounddevice>=0.4.4",
    ],
    entry_points={
        "console_scripts": [
            "voice-train=voice_recognition.train:main",
            "voice-predict=voice_recognition.predict:main",
        ],
    },
)
