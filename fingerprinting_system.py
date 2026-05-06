import librosa
import numpy as np
import hashlib
import logging
from scipy.ndimage import maximum_filter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Constants for stability
SAMPLING_RATE = 22050
WINDOW_SIZE = 4096
HOP_SIZE = 1024
PEAK_NEIGHBORHOOD_SIZE = 20
FAN_VALUE = 15
# Target zone for pairing (in frames) - Enhancement: constraints
MIN_HASH_TIME_DELTA = 10
MAX_HASH_TIME_DELTA = 200
# Threshold for peak detection (in dB relative to the loudest peak) - Enhancement: noise robustness
THRESHOLD_DB = -35

def get_peaks(audio):
    """
    Extracts stable frequency peaks from an audio signal.
    Prioritizes strongest peaks for robustness in noise.
    """
    # 1. Generate Spectrogram (STFT)
    S = np.abs(librosa.stft(audio, n_fft=WINDOW_SIZE, hop_length=HOP_SIZE))

    # 2. Convert to dB scale (stable scaling)
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    # 3. Find local maxima using a neighborhood filter
    data_max = maximum_filter(S_db, size=PEAK_NEIGHBORHOOD_SIZE)
    maxima = (S_db == data_max)

    # 4. Use a threshold to pick only strong peaks
    peak_indices = np.where(maxima & (S_db > THRESHOLD_DB))
    freqs, times = peak_indices
    amps = S_db[peak_indices]

    # Enhancement: Prioritize strongest peaks
    peaks_with_amps = sorted(zip(times, freqs, amps), key=lambda x: x[2], reverse=True)

    # Keep top peaks for robustness
    MAX_PEAKS = 3000
    if len(peaks_with_amps) > MAX_PEAKS:
        peaks_with_amps = peaks_with_amps[:MAX_PEAKS]

    # Sort back by time for hashing
    peaks = sorted([(t, f) for t, f, a in peaks_with_amps], key=lambda x: x[0])

    logger.info(f"Extracted {len(peaks)} stable peaks.")
    return peaks

def generate_hashes(peaks):
    """
    Generates SHA1 hashes by pairing peaks.
    Includes time delta constraints.
    """
    hashes = []

    for i in range(len(peaks)):
        for j in range(1, FAN_VALUE + 1):
            if (i + j) < len(peaks):
                t1, f1 = peaks[i]
                t2, f2 = peaks[i + j]
                t_delta = t2 - t1

                # Enhancement: Ensure pairing is within the target time delta zone
                if MIN_HASH_TIME_DELTA <= t_delta <= MAX_HASH_TIME_DELTA:
                    h_string = f"{int(f1)}|{int(f2)}|{int(t_delta)}"
                    h = hashlib.sha1(h_string.encode('utf-8')).hexdigest()
                    hashes.append((h, t1))

    logger.info(f"Generated {len(hashes)} unique-alignment hashes.")
    return hashes

# Example Usage Block
if __name__ == "__main__":
    # Test with dummy data
    duration = 5 # seconds
    dummy_audio = np.random.uniform(-1, 1, SAMPLING_RATE * duration)

    logger.info("Starting fingerprinting pipeline...")
    peaks = get_peaks(dummy_audio)
    hashes = generate_hashes(peaks)

    if hashes:
        logger.info(f"Sample hash result: {hashes[0]}")
