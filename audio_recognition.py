import os
import sqlite3
import hashlib
import numpy as np
import librosa
from scipy.ndimage import maximum_filter
from scipy.ndimage import binary_erosion

# --- Configuration ---
DB_NAME = "fingerprints.db"
SAMPLING_RATE = 44100
WINDOW_SIZE = 4096
HOP_SIZE = 1024
# Connectivity for peak finding (distance between peaks)
PEAK_NEIGHBORHOOD_SIZE = 20
# Fan-out for combinatorial hashing (how many neighbors to pair with)
FAN_VALUE = 15
# Enhancement: Time delta constraints
MIN_HASH_TIME_DELTA = 10
MAX_HASH_TIME_DELTA = 200
# Enhancement: Noise threshold
THRESHOLD_DB = -35

class AudioFingerprinter:
    def __init__(self, db_path=DB_NAME):
        """Initializes the system and the SQLite database."""
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Creates the database schema if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS songs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    file_path TEXT,
                    total_hashes INTEGER
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fingerprints (
                    hash TEXT,
                    song_id INTEGER,
                    offset INTEGER,
                    FOREIGN KEY(song_id) REFERENCES songs(id)
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_hash ON fingerprints (hash)')
            conn.commit()

    def _get_peaks(self, audio_data):
        """
        Finds prominent spectral peaks in the audio signal (Constellation Map).
        Prioritizes strongest peaks for robustness in noise.
        """
        # Compute Spectrogram
        S = librosa.stft(audio_data, n_fft=WINDOW_SIZE, hop_length=HOP_SIZE)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

        # Local Maxima Filter
        struct = np.ones((PEAK_NEIGHBORHOOD_SIZE, PEAK_NEIGHBORHOOD_SIZE))
        local_max = maximum_filter(S_db, footprint=struct) == S_db

        # Enhancement: Prioritize strongest peaks and use relative threshold
        peak_indices = np.where(local_max & (S_db > THRESHOLD_DB))
        freqs, times = peak_indices
        amps = S_db[peak_indices]

        # Combine and sort by amplitude descending
        peaks_with_amps = sorted(zip(times, freqs, amps), key=lambda x: x[2], reverse=True)

        # Limit total peaks to keep the system responsive while retaining best features
        MAX_PEAKS = 3000
        if len(peaks_with_amps) > MAX_PEAKS:
            peaks_with_amps = peaks_with_amps[:MAX_PEAKS]

        # Sort back by time index for hash generation
        peaks = sorted([(f, t) for t, f, a in peaks_with_amps], key=lambda x: x[1])
        return peaks

    def _generate_hashes(self, peaks):
        """
        Generates combinatorial hashes from pairs of peaks.
        Shift-invariant by using time differences.
        Includes time delta constraints.
        """
        hashes = []
        for i in range(len(peaks)):
            for j in range(1, FAN_VALUE + 1):
                if (i + j) < len(peaks):
                    f1, t1 = peaks[i]
                    f2, t2 = peaks[i + j]
                    t_delta = t2 - t1

                    # Enhancement: Apply time delta constraints
                    if MIN_HASH_TIME_DELTA <= t_delta <= MAX_HASH_TIME_DELTA:
                        # Combine frequencies and time delta into a hash
                        h_string = f"{f1}|{f2}|{t_delta}"
                        h = hashlib.sha1(h_string.encode()).hexdigest()[:20]
                        hashes.append((h, int(t1)))
        return hashes

    def add_song(self, file_path):
        """
        Indexes a new song into the database.
        """
        if not os.path.exists(file_path):
            print(f"[Error] File not found: {file_path}")
            return

        song_name = os.path.basename(file_path)
        print(f"[System] Fingerprinting: {song_name}...")

        try:
            # 1. Load and Normalize (Mono, 44100 Hz)
            audio, _ = librosa.load(file_path, sr=SAMPLING_RATE, mono=True)

            # 2. Extract Peaks and Hashes
            peaks = self._get_peaks(audio)
            hashes = self._generate_hashes(peaks)

            if not hashes:
                print(f"[Warn] No fingerprints generated for {song_name}.")
                return

            # 3. Save to Database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(
                        "INSERT INTO songs (name, file_path, total_hashes) VALUES (?, ?, ?)",
                        (song_name, file_path, len(hashes))
                    )
                    song_id = cursor.lastrowid

                    cursor.executemany(
                        "INSERT INTO fingerprints (hash, song_id, offset) VALUES (?, ?, ?)",
                        [(h, song_id, offset) for h, offset in hashes]
                    )
                    conn.commit()
                    print(f"[Success] Added '{song_name}' with {len(hashes)} hashes.")
                except sqlite3.IntegrityError:
                    print(f"[Info] Song '{song_name}' already exists.")
        except Exception as e:
            print(f"[Error] Processing failed: {e}")

    def detect_song(self, file_path):
        """
        Detects a song from a short audio clip.
        Tolerates missing hashes by focusing on temporal alignment.
        """
        if not os.path.exists(file_path):
            print(f"[Error] Clip not found: {file_path}")
            return None

        print(f"[System] Analyzing clip: {os.path.basename(file_path)}...")

        try:
            # 1. Fingerprint the input clip
            audio, _ = librosa.load(file_path, sr=SAMPLING_RATE, mono=True)
            peaks = self._get_peaks(audio)
            hashes = self._generate_hashes(peaks)

            if not hashes:
                print("[Warn] Clip is too silent or short.")
                return None

            # 2. Search for matches in database
            matches = {} # song_id -> list of offset differences

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for h, t_clip in hashes:
                    cursor.execute("SELECT song_id, offset FROM fingerprints WHERE hash = ?", (h,))
                    for song_id, t_song in cursor.fetchall():
                        if song_id not in matches:
                            matches[song_id] = []
                        # Time alignment check: t_song - t_clip should be constant for matches
                        matches[song_id].append(t_song - t_clip)

            if not matches:
                print("[Result] No matches found.")
                return None

            # 3. Find song with the highest aligned matches
            best_song_id = None
            max_aligned_matches = 0

            for song_id, diffs in matches.items():
                if not diffs: continue
                # Calculate the mode of the offset differences
                counts = {}
                for d in diffs:
                    counts[d] = counts.get(d, 0) + 1

                song_max = max(counts.values())
                if song_max > max_aligned_matches:
                    max_aligned_matches = song_max
                    best_song_id = song_id

            if not best_song_id:
                print("[Result] No significant match found.")
                return None

            # 4. Fetch details and calculate confidence
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name, total_hashes FROM songs WHERE id = ?", (best_song_id,))
                song_name, total_hashes = cursor.fetchone()

                # Confidence = (Matched Aligned Hashes / Total Query Hashes)
                # Enhancement: Matching tolerates missing hashes.
                # A score of 0.3 (30%) is considered strong for short/noisy clips.
                confidence = max_aligned_matches / len(hashes)

                print(f"[Match Found] {song_name}")
                print(f"  - Aligned Matches: {max_aligned_matches}")
                print(f"  - Confidence Score: {confidence:.4f}")

                return {"song": song_name, "confidence": confidence, "matches": max_aligned_matches}

        except Exception as e:
            print(f"[Error] Detection failed: {e}")
            return None

# --- Function Interface & CLI ---
if __name__ == "__main__":
    recognizer = AudioFingerprinter()

    print("\n=== Shazam-style Audio Detection System ===")
    print("1. Add Song (Index a full track)")
    print("2. Detect Song (Identify a clip)")
    print("3. Exit")

    while True:
        choice = input("\nEnter choice (1-3): ")
        if choice == '1':
            path = input("Enter path to full song file: ").strip('"')
            recognizer.add_song(path)
        elif choice == '2':
            path = input("Enter path to audio clip: ").strip('"')
            recognizer.detect_song(path)
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice.")
