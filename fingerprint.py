import numpy as np
import librosa
from scipy.ndimage import maximum_filter
import hashlib
import os
import logging

logger = logging.getLogger(__name__)

# Constants for the algorithm
FAN_VALUE = 15
PEAK_NEIGHBORHOOD_SIZE = 20
# Threshold for peak detection (in dB relative to the loudest peak)
# Increasing this makes detection more selective, decreasing noise/hashes.
THRESHOLD_DB = -50

def get_hashes(file_path):
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # 1. Load & Convert to Mono
        y, sr = librosa.load(file_path, sr=22050, mono=True)

        if len(y) == 0:
            logger.warning(f"Empty audio file: {file_path}")
            return []

        # 2. Generate Spectrogram (STFT)
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))

        # Convert to dB scale with reference to the maximum value
        # This makes the loudest peak 0dB and all others negative.
        S_db = librosa.amplitude_to_db(S, ref=np.max)

        # 3. Identify Peaks (Constellation Map)
        # Find local maxima in a 20x20 window
        struct = np.ones((PEAK_NEIGHBORHOOD_SIZE, PEAK_NEIGHBORHOOD_SIZE))
        local_max = maximum_filter(S_db, footprint=struct) == S_db

        # Filter peaks: Must be a local maximum AND above the threshold
        # This prevents picking up millions of tiny noise peaks
        detected_peaks = local_max & (S_db > THRESHOLD_DB)

        freq_bins, time_bins = np.where(detected_peaks)

        # Group and sort by time
        peaks = sorted(zip(time_bins, freq_bins), key=lambda x: x[0])

        # Limit total peaks to avoid overwhelming Firestore (e.g., 1500 peaks max)
        if len(peaks) > 1500:
            logger.info(f"Limiting peaks from {len(peaks)} to 1500 for performance")
            peaks = peaks[:1500]

        logger.info(f"Processing {len(peaks)} peaks for hashing")

        # 4. Combinatorial Hashing
        hashes = []
        for i in range(len(peaks)):
            for j in range(1, FAN_VALUE + 1):
                if (i + j) < len(peaks):
                    t1, f1 = peaks[i]
                    t2, f2 = peaks[i + j]
                    t_delta = t2 - t1

                    # Ensure peaks are within a 4.6 second window (200 frames)
                    if 0 < t_delta <= 200:
                        # Create hash using frequencies and time delta
                        h_input = f"{f1}|{f2}|{t_delta}"
                        h = hashlib.sha1(h_input.encode()).hexdigest()
                        hashes.append((h, t1))

        return hashes
    except Exception as e:
        logger.exception(f"Error processing audio file {file_path}")
        raise e
