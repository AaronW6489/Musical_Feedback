import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import tempfile
import os

def record_audio(duration=10, fs=44100):
    """
    Records audio from the default microphone for the given duration (seconds).
    Returns the path to the saved WAV file.
    """
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    temp_dir = tempfile.gettempdir()
    wav_path = os.path.join(temp_dir, "user_recording.wav")
    write(wav_path, fs, audio)
    print(f"Saved recording to {wav_path}")
    return wav_path
