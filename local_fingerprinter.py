import librosa
import numpy as np
import hashlib
import logging
from scipy.ndimage import maximum_filter
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SAMPLING_RATE = 22050
PEAK_NEIGHBORHOOD_SIZE = 20
FAN_VALUE = 15
# Enhancement: Time delta constraints
MIN_HASH_TIME_DELTA = 10
MAX_HASH_TIME_DELTA = 200
# Enhancement: Noise threshold
THRESHOLD_DB = -35

# In-memory database
# Structure: { hash_value: [(song_name, offset_in_frames), ...], ... }
FINGERPRINT_DB = defaultdict(list)
SONG_METADATA = {}

def get_peaks(y):
    """
    Generate spectrogram and extract local peaks.
    Prioritizes strongest peaks for robustness in noise.
    """
    # 1. Generate Spectrogram (STFT)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    # 2. Extract Peaks using local maximum filter
    data_max = maximum_filter(S_db, size=PEAK_NEIGHBORHOOD_SIZE)
    maxima = (S_db == data_max)

    # 3. Filter by threshold and prioritize strongest peaks
    peak_indices = np.where(maxima & (S_db > THRESHOLD_DB))
    freqs, times = peak_indices
    amps = S_db[peak_indices]

    # Combine and sort by amplitude descending
    peaks_with_amps = sorted(zip(times, freqs, amps), key=lambda x: x[2], reverse=True)

    # Keep top peaks for robustness
    MAX_PEAKS = 3000
    if len(peaks_with_amps) > MAX_PEAKS:
        peaks_with_amps = peaks_with_amps[:MAX_PEAKS]

    # Sort back by time for hashing
    peaks = sorted([(t, f) for t, f, a in peaks_with_amps], key=lambda x: x[0])

    logger.info(f"Extracted {len(peaks)} peaks.")
    return peaks

def generate_hashes(peaks):
    """
    Create hashes by pairing peaks within a target zone.
    Includes time delta constraints.
    """
    hashes = []
    for i in range(len(peaks)):
        for j in range(1, FAN_VALUE + 1):
            if (i + j) < len(peaks):
                t1, f1 = peaks[i]
                t2, f2 = peaks[i + j]
                t_delta = t2 - t1

                # Enhancement: Apply time delta constraints
                if MIN_HASH_TIME_DELTA <= t_delta <= MAX_HASH_TIME_DELTA:
                    data = f"{f1}|{f2}|{t_delta}"
                    h = hashlib.sha1(data.encode('utf-8')).hexdigest()
                    hashes.append((h, t1))

    logger.info(f"Generated {len(hashes)} hashes.")
    return hashes

def add_song(file_path):
    """
    Load, fingerprint, and store a song in the memory database.
    """
    song_name = file_path.split('/')[-1]
    logger.info(f"Adding song: {song_name}")

    try:
        y, _ = librosa.load(file_path, sr=SAMPLING_RATE, mono=True)
        peaks = get_peaks(y)
        hashes = generate_hashes(peaks)

        for h, offset in hashes:
            FINGERPRINT_DB[h].append((song_name, offset))

        SONG_METADATA[song_name] = len(hashes)
        logger.info(f"Successfully added {song_name} to database.")
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")

def detect_song(query_file):
    """
    Identify a song from a query clip.
    Tolerates missing hashes by focusing on temporal alignment.
    """
    logger.info(f"Detecting song from query: {query_file}")

    try:
        y, _ = librosa.load(query_file, sr=SAMPLING_RATE, mono=True)
        peaks = get_peaks(y)
        query_hashes = generate_hashes(peaks)

        # Structure: { song_name: { offset_diff: count } }
        matches = defaultdict(lambda: defaultdict(int))

        total_query_hashes = len(query_hashes)
        if total_query_hashes == 0:
            return {"song_name": "Unknown", "confidence": 0.0}

        for h_query, t_query in query_hashes:
            if h_query in FINGERPRINT_DB:
                for song_name, t_db in FINGERPRINT_DB[h_query]:
                    diff = t_db - t_query
                    matches[song_name][diff] += 1

        # Analyze matches
        best_song = "Unknown"
        max_score = 0

        for song_name, offsets in matches.items():
            # The highest count at a specific offset difference is our "match score"
            song_score = max(offsets.values())

            if song_score > max_score:
                max_score = song_score
                best_song = song_name

        # Enhancement: Confidence logic tolerates missing hashes.
        # Score is relative to total query hashes.
        confidence = max_score / total_query_hashes if total_query_hashes > 0 else 0.0

        logger.info(f"Best match found: {best_song} with {max_score} aligned matches. Confidence: {confidence:.4f}")

        return {
            "song_name": best_song,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        logger.error(f"Error during detection: {e}")
        return {"song_name": "Error", "confidence": 0.0}

if __name__ == "__main__":
    print("--- Local Music Fingerprinting System ---")
    # Example Usage:
    # add_song("path/to/song.mp3")
    # result = detect_song("path/to/clip.mp3")
