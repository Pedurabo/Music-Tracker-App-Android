import numpy as np
import librosa
import hashlib
import os
import math
import logging
from collections import defaultdict
from scipy.ndimage import maximum_filter

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# =========================
# CONFIGURATION
# =========================
# Audio Processing
SAMPLING_RATE = 22050
WINDOW_SIZE = 4096
HOP_SIZE = 1024

# Peak Detection
PEAK_NEIGHBORHOOD_SIZE = 20  # Distance between peaks in spectrogram
THRESHOLD_DB = -35           # Minimum amplitude relative to max (0 dB)
TOP_N_PEAKS_PER_TIME = 5    # Max peaks per time slice for noise robustness

# Hashing (Combinatorial Pairing)
FAN_VALUE = 15               # Max pairings per anchor peak
MIN_HASH_TIME_DELTA = 10     # Min frames between paired peaks (~0.2s)
MAX_HASH_TIME_DELTA = 200    # Max frames between paired peaks (~4.6s)

# Global In-Memory Database
# Structure: { hash_value: [(song_id, offset_in_frames), ...], ... }
database = defaultdict(list)
song_metadata = {}

# =========================
# CORE ALGORITHM
# =========================

def get_peaks(y):
    """
    Extracts high-energy spectral peaks (Constellation Map).
    Uses maximum_filter for 2D local maxima detection.
    """
    # 1. Generate Spectrogram (Magnitude)
    S = np.abs(librosa.stft(y, n_fft=WINDOW_SIZE, hop_length=HOP_SIZE))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    # 2. Find local maxima in 2D neighborhood
    struct = np.ones((PEAK_NEIGHBORHOOD_SIZE, PEAK_NEIGHBORHOOD_SIZE))
    local_max = maximum_filter(S_db, footprint=struct) == S_db

    # 3. Apply amplitude threshold
    peak_mask = local_max & (S_db > THRESHOLD_DB)

    # Extract coordinates
    freq_bins, time_bins = np.where(peak_mask)
    amps = S_db[freq_bins, time_bins]

    # 4. Filter for noise: Keep only the strongest peaks per time slice
    peaks_by_time = defaultdict(list)
    for f, t, a in zip(freq_bins, time_bins, amps):
        peaks_by_time[t].append((f, a))

    final_peaks = []
    for t, candidates in peaks_by_time.items():
        # Sort by amplitude descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        for f, a in candidates[:TOP_N_PEAKS_PER_TIME]:
            final_peaks.append((int(f), int(t)))

    # Sort peaks by time for consistent hashing
    final_peaks.sort(key=lambda x: x[1])

    return final_peaks

def generate_hashes(peaks):
    """
    Creates robust fingerprints by pairing nearby peaks.
    Hash format: SHA1(freq1 | freq2 | time_delta)
    """
    hashes = []
    # peaks: [(f, t), ...] sorted by time
    for i in range(len(peaks)):
        f1, t1 = peaks[i]

        # Look for peaks in a "target zone" ahead of the anchor
        count = 0
        for j in range(i + 1, len(peaks)):
            f2, t2 = peaks[j]
            t_delta = t2 - t1

            if t_delta < MIN_HASH_TIME_DELTA:
                continue
            if t_delta > MAX_HASH_TIME_DELTA:
                break # Optimization: peaks are sorted by time

            # Create a unique SHA-1 hash for the pair
            h_string = f"{f1}|{f2}|{t_delta}".encode()
            h = hashlib.sha1(h_string).hexdigest()[:20]

            # Store (hash, offset_t1)
            hashes.append((h, int(t1)))

            count += 1
            if count >= FAN_VALUE:
                break

    return hashes

# =========================
# SYSTEM INTERFACE
# =========================

def add_song(file_path, song_name=None):
    """
    Indexes a song into the global database.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return

    song_id = hashlib.md5(file_path.encode()).hexdigest()[:8]
    if song_name is None:
        song_name = os.path.basename(file_path)

    logger.info(f"\n[System] Indexing Song: '{song_name}'")

    try:
        # Load audio (Mono, 22050Hz)
        y, _ = librosa.load(file_path, sr=SAMPLING_RATE, mono=True)

        # Process
        peaks = get_peaks(y)
        hashes = generate_hashes(peaks)

        # Store
        for h, offset in hashes:
            database[h].append((song_id, offset))

        song_metadata[song_id] = song_name
        logger.info(f"  - Peaks count: {len(peaks)}")
        logger.info(f"  - Hash count: {len(hashes)}")
        logger.info(f"[Success] Added to database.")

    except Exception as e:
        logger.error(f"[Error] Failed to index {file_path}: {e}")

def detect_song(file_path):
    """
    Identifies a song from a query clip using temporal alignment.
    """
    if not os.path.exists(file_path):
        logger.error(f"Clip not found: {file_path}")
        return None

    logger.info(f"\n[System] Analyzing Clip: {os.path.basename(file_path)}")

    try:
        # 1. Fingerprint the query
        y, _ = librosa.load(file_path, sr=SAMPLING_RATE, mono=True)
        peaks = get_peaks(y)
        query_hashes = generate_hashes(peaks)

        logger.info(f"  - Peaks count: {len(peaks)}")
        logger.info(f"  - Hash count: {len(query_hashes)}")

        if not query_hashes:
            logger.warning("[Warn] No hashes generated from query.")
            return "Unknown", 0.0

        total_query_hashes = len(query_hashes)

        # 2. Find matches and compute deltas (Temporal Alignment)
        # song_delta_counts[song_id][delta] = count
        song_delta_counts = defaultdict(lambda: defaultdict(int))

        # Group query hashes for efficient lookup
        query_map = defaultdict(list)
        for h, t_q in query_hashes:
            query_map[h].append(t_q)

        match_found = 0
        for h, offsets_q in query_map.items():
            if h in database:
                for s_id, t_db in database[h]:
                    for t_q in offsets_q:
                        delta = t_db - t_q
                        song_delta_counts[s_id][delta] += 1
                        match_found += 1

        logger.info(f"  - Matches: {match_found}")

        if not song_delta_counts:
            logger.info("[Result] No matches found in database.")
            return "Unknown", 0.0

        # 3. Find the best match
        best_song_id = None
        max_aligned_matches = 0

        for s_id, deltas in song_delta_counts.items():
            song_max_aligned = max(deltas.values())
            if song_max_aligned > max_aligned_matches:
                max_aligned_matches = song_max_aligned
                best_song_id = s_id

        # 4. Confidence Score Calculation
        # Raw density: ratio of aligned matches to total query fingerprints
        confidence = max_aligned_matches / total_query_hashes

        song_name = song_metadata.get(best_song_id, "Unknown")

        # 5. Result Summary
        logger.info(f"  - Alignment score: {max_aligned_matches}")
        logger.info(f"  - Confidence: {confidence:.4f}")

        if confidence < 0.1:
            logger.info("[Result] No confident match found.")
            return "Unknown", 0.0

        logger.info(f"[Match Found] {song_name}")
        return song_name, round(confidence, 4)

    except Exception as e:
        logger.error(f"[Error] Detection failed: {e}")
        return "Error", 0.0

# =========================
# TEST SUITE
# =========================
if __name__ == "__main__":
    # Point this to a real audio file on your system to test
    test_song = r"C:\Users\HP\music-backend\songs\test_song.mp3"

    if os.path.exists(test_song):
        logger.info("=== STARTING SHAZAM-LIKE FINGERPRINT TEST ===")

        # 1. Ingest
        add_song(test_song, "Master Track")

        # 2. Detect (Exact match should be 1.0)
        detect_song(test_song)
    else:
        logger.warning(f"Test file not found at: {test_song}")
        logger.info("Update the 'test_song' path in working_detector.py to run a full test.")
