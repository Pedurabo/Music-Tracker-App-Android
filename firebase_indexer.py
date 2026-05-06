import os
import hashlib
import uuid
import librosa
import numpy as np
from scipy.ndimage import maximum_filter
from google.cloud import firestore

# --- Configuration ---
TARGET_SR = 22050
WINDOW_SIZE = 4096
HOP_SIZE = 1024
PEAK_NEIGHBORHOOD_SIZE = 20
FAN_VALUE = 15
# Enhancement: Time delta constraints
MIN_HASH_TIME_DELTA = 10
MAX_HASH_TIME_DELTA = 200
FIRESTORE_BATCH_LIMIT = 500
# Enhancement: Noise threshold
THRESHOLD_DB = -35
AUDIO_EXTENSIONS = ('.mp3', '.wav', '.m4a', '.flac', '.ogg')

class MusicTrackerIndexer:
    def __init__(self, service_account_json=None):
        try:
            if service_account_json and os.path.exists(service_account_json):
                print(f"[System] Successfully located: {service_account_json}")
                self.db = firestore.Client.from_service_account_json(service_account_json)
            else:
                self.db = firestore.Client()
        except Exception as e:
            print(f"\n[ERROR] Connection failed: {e}")
            raise

    def _get_peaks(self, audio_data):
        """
        Extracts spectral peaks. Prioritizes strongest peaks for robustness.
        """
        S = np.abs(librosa.stft(audio_data, n_fft=WINDOW_SIZE, hop_length=HOP_SIZE))
        S_db = librosa.amplitude_to_db(S, ref=np.max)

        struct = np.ones((PEAK_NEIGHBORHOOD_SIZE, PEAK_NEIGHBORHOOD_SIZE))
        local_max = maximum_filter(S_db, footprint=struct) == S_db

        peak_indices = np.where(local_max & (S_db > THRESHOLD_DB))
        freqs, times = peak_indices
        amps = S_db[peak_indices]

        # Enhancement: Prioritize strongest peaks
        peaks_with_amps = sorted(zip(times, freqs, amps), key=lambda x: x[2], reverse=True)

        MAX_PEAKS = 3000
        if len(peaks_with_amps) > MAX_PEAKS:
            peaks_with_amps = peaks_with_amps[:MAX_PEAKS]

        # Sort back by time for hashing
        peaks = sorted([(f, t) for t, f, a in peaks_with_amps], key=lambda x: x[1])
        return peaks

    def _generate_hashes(self, peaks):
        """
        Generates robust hashes with time delta constraints.
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
                        h_string = f"{f1}|{f2}|{t_delta}".encode()
                        h = hashlib.sha1(h_string).hexdigest()[:20]
                        hashes.append((h, int(t1)))
        return hashes

    def index_song(self, file_path, song_title):
        if not os.path.exists(file_path):
            print(f"[Error] Audio file not found: {file_path}")
            return False

        song_id = str(uuid.uuid4())
        print(f"\n[System] Processing: '{song_title}'")

        try:
            audio, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)
            peaks = self._get_peaks(audio)
            hashes = self._generate_hashes(peaks)

            if not hashes:
                print("  [Warn] No fingerprints generated.")
                return False

            print(f"  [Info] Generated {len(hashes)} hashes. Starting batch upload...")
            self._store_in_firestore(hashes, song_id, song_title)
            print(f"  [Success] Indexed '{song_title}'")
            return True
        except Exception as e:
            print(f"  [Error] Failed to process {song_title}: {e}")
            return False

    def index_folder(self, folder_path):
        if not os.path.isdir(folder_path):
            print(f"[Error] Folder not found: {folder_path}")
            return

        files = [f for f in os.listdir(folder_path) if f.lower().endswith(AUDIO_EXTENSIONS)]
        print(f"\n--- Found {len(files)} audio files in folder ---")

        success_count = 0
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            title = os.path.splitext(filename)[0].replace('_', ' ').title()
            if self.index_song(file_path, title):
                success_count += 1

        print(f"\n--- Bulk Indexing Complete: {success_count}/{len(files)} succeeded ---")

    def _store_in_firestore(self, hashes, song_id, song_title):
        batch = self.db.batch()
        count = 0
        for h_val, offset in hashes:
            doc_ref = self.db.collection('song_hashes').document()
            batch.set(doc_ref, {
                'hash': h_val,
                'offset': offset,
                'song_id': song_id,
                'song_title': song_title
            })
            count += 1
            if count >= FIRESTORE_BATCH_LIMIT:
                batch.commit()
                batch = self.db.batch()
                count = 0
        if count > 0:
            batch.commit()

if __name__ == "__main__":
    print("\n--- MusicTracker Bulk Indexer ---")
    possible_names = ["service_account.json.json", "service_account.json", "service-account.json"]
    cred_path = next((n for n in possible_names if os.path.exists(n)), None)

    if not cred_path:
        cred_path = input("Enter path to service-account.json: ").strip().strip('"')

    if cred_path and os.path.exists(cred_path):
        indexer = MusicTrackerIndexer(cred_path)
        print("\nOptions:")
        print("1. Index a single file")
        print("2. Index an entire folder")
        choice = input("Enter choice (1/2): ").strip()

        if choice == '1':
            path = input("Enter file path: ").strip().strip('"')
            title = input("Enter song title: ").strip()
            indexer.index_song(path, title)
        elif choice == '2':
            folder = input("Enter folder path: ").strip().strip('"')
            indexer.index_folder(folder)
        else:
            print("Invalid choice.")
    else:
        print("[Error] Could not find credentials file.")
