import numpy as np
from scipy.io import wavfile

# Parameters
fs = 22050  # Sampling frequency
duration = 30  # Seconds
filename = "test_audio.wav"

print(f"Generating {duration}s of synthetic audio...")

# Create a time array
t = np.linspace(0, duration, int(fs * duration))

# Generate a mix of frequencies that change over time (to create "peaks")
# This creates a sliding tone (chirp) + some harmonics
audio = (np.sin(2 * np.pi * (440 + 100 * np.sin(2 * np.pi * 0.1 * t)) * t) +
         0.5 * np.sin(2 * np.pi * 880 * t) +
         0.2 * np.sin(2 * np.pi * 1500 * t))

# Normalize to 16-bit PCM range
audio = (audio * 32767 / np.max(np.abs(audio))).astype(np.int16)

# Save as WAV
wavfile.write(filename, fs, audio)

print(f"Success! Created: {filename}")
print("You can now use this file path in the indexer.")
