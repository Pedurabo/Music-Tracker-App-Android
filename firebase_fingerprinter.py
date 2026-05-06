import os
import hashlib
import uuid
import librosa
import numpy as np
import math
from scipy.ndimage import maximum_filter
from google.cloud import firestore
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

# --- Configuration ---
SAMPLING_RATE = 22050
WINDOW_SIZE = 4096
HOP_SIZE = 1024
PEAK_NEIGHBORHOOD_SIZE = 20
FAN_VALUE = 15  # Max number of pairs per anchor peak
MIN_HASH_TIME_DELTA = 10
MAX_HASH_TIME_DELTA = 200
THRESHOLD_DB = -35
FIRESTORE_QUERY_LIMIT = 10  # Optimized to 10 per query as requested
FIRESTORE_BATCH_LIMIT = 500
TOP_N_PEAKS_PER_TIME = 5 # Number of peaks to keep per time window for noise reduction

class FirebaseFingerprinter:
    def __init__(self, service_account_json=None):
        if service_account_json and os.path.exists(service_account_json):
            self.db = firestore.Client.from_service_account_json(service_account_json)
        else:
            self.db = firestore.Client()

    def _get_peaks(self, audio_data):
        S = np.abs(librosa.stft(audio_data, n_fft=WINDOW_SIZE, hop_length=HOP_SIZE))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        struct = np.ones((PEAK_NEIGHBORHOOD_SIZE, PEAK_NEIGHBORHOOD_SIZE))
        local_max = maximum_filter(S_db, footprint=struct) == S_db
        peak_mask = local_max & (S_db > THRESHOLD_DB)
        freq_bins, time_bins = np.where(peak_mask)
        amps = S_db[freq_bins, time_bins]
        peaks_by_time = defaultdict(list)
        for f, t, a in zip(freq_bins, time_bins, amps):
            peaks_by_time[t].append((f, a))
        final_peaks = []
        for t, candidates in peaks_by_time.items():
            candidates.sort(key=lambda x: x[1], reverse=True)
            for f, a in candidates[:TOP_N_PEAKS_PER_TIME]:
                final_peaks.append((int(f), int(t), float(a)))
        MAX_TOTAL_PEAKS = 1500
        final_peaks.sort(key=lambda x: x[2], reverse=True)
        if len(final_peaks) > MAX_TOTAL_PEAKS:
            final_peaks = final_peaks[:MAX_TOTAL_PEAKS]
        return sorted([(f, t) for f, t, a in final_peaks], key=lambda x: x[1])

    def _generate_hashes(self, peaks):
        hashes = []
        for i in range(len(peaks)):
            f1, t1 = peaks[i]
            count = 0
            for j in range(i + 1, len(peaks)):
                f2, t2 = peaks[j]
                t_delta = t2 - t1
                if t_delta < MIN_HASH_TIME_DELTA: continue
                if t_delta > MAX_HASH_TIME_DELTA: break
                h_string = f"{f1}|{f2}|{t_delta}".encode()
                h = hashlib.sha1(h_string).hexdigest()[:20]
                hashes.append((h, int(t1)))
                count += 1
                if count >= FAN_VALUE: break
        return hashes

    def _query_firestore_chunk(self, chunk):
        """Helper for parallel Firestore lookup using 'in' query."""
        matches = []
        try:
            # Optimized: Query up to 10 hashes in a single call
            docs = self.db.collection('song_hashes').where('hash', 'in', chunk).stream()
            for doc in docs:
                matches.append(doc.to_dict())
        except Exception as e:
            print(f"Firestore chunk query error: {e}")
        return matches

    def _get_all_db_matches(self, unique_hashes):
        """
        Batches hashes and queries Firestore in parallel.
        Returns a list of all matching documents from DB.
        """
        # Chunk unique hashes to respect Firestore 'in' query limit (10)
        chunks = [unique_hashes[i:i + FIRESTORE_QUERY_LIMIT]
                  for i in range(0, len(unique_hashes), FIRESTORE_QUERY_LIMIT)]

        print(f"[Info] Executing {len(chunks)} batched Firestore queries...")

        all_db_matches = []
        with ThreadPoolExecutor(max_workers=15) as executor:
            results = list(executor.map(self._query_firestore_chunk, chunks))
            for result_list in results:
                all_db_matches.extend(result_list)
        return all_db_matches

    def detect_song(self, file_path):
        """
        Identifies a song by batch-querying Firestore for hash matches.
        Uses temporal alignment to find the best match.
        Returns (song_name, confidence, match_level, song_id)
        """
        if not os.path.exists(file_path):
            print(f"[Error] Clip not found: {file_path}")
            return None, 0.0, "Unknown", None

        print(f"[System] Analyzing clip: {os.path.basename(file_path)}...")

        try:
            # 1. Local Processing
            audio, _ = librosa.load(file_path, sr=SAMPLING_RATE, mono=True)
            peaks = self._get_peaks(audio)
            query_hashes = self._generate_hashes(peaks)

            if not query_hashes:
                print("[Warn] Clip is too silent or short.")
                return None, 0.0, "Unknown", None

            total_query_hashes = len(query_hashes)
            unique_query_hashes = list(set([h for h, _ in query_hashes]))

            # 2. Optimized Database Lookup (Batched)
            all_db_matches = self._get_all_db_matches(unique_query_hashes)

            if not all_db_matches:
                print("[Result] No matches found in database.")
                return None, 0.0, "Unknown", None

            # 3. Local Aggregation and Temporal Alignment
            # Organize matches by song_id
            song_delta_counts = defaultdict(lambda: defaultdict(int))

            query_offset_map = defaultdict(list)
            for h, t_q in query_hashes:
                query_offset_map[h].append(t_q)

            for db_match in all_db_matches:
                h = db_match['hash']
                s_id = db_match['song_id']
                t_db = db_match['offset']

                # Compute time differences for all occurrences of this hash
                for t_q in query_offset_map[h]:
                    delta = t_db - t_q
                    song_delta_counts[s_id][delta] += 1

            # 4. Find the best matching song_id
            best_song_id = None
            max_aligned_matches = 0

            for s_id, deltas in song_delta_counts.items():
                song_max_aligned = max(deltas.values())
                if song_max_aligned > max_aligned_matches:
                    max_aligned_matches = song_max_aligned
                    best_song_id = s_id

            if not best_song_id:
                return None, 0.0, "Unknown", None

            # 5. Scoring and Metadata Retrieval
            confidence = max_aligned_matches / total_query_hashes
            match_level = self._get_match_level(confidence)

            # Fetch song name from Firestore
            song_doc = self.db.collection('songs').document(best_song_id).get()
            detected_name = song_doc.to_dict().get('name', 'Unknown') if song_doc.exists else "Unknown"

            print(f"[Match Found] {detected_name} (Confidence: {confidence:.4f}, Level: {match_level})")
            return detected_name, round(confidence, 4), match_level, best_song_id

        except Exception as e:
            print(f"[Error] Detection failed: {e}")
            return None, 0.0, "Error", None

    def _get_match_level(self, confidence):
        if confidence < 0.1: return "Unknown"
        if confidence < 0.3: return "Weak match"
        if confidence < 0.7: return "Moderate match"
        return "Strong match"

    def index_song(self, file_path, song_title):
        if not os.path.exists(file_path): return False
        song_id = str(uuid.uuid4())
        print(f"[System] Indexing: '{song_title}' (ID: {song_id})")
        try:
            self.db.collection('songs').document(song_id).set({
                'name': song_title,
                'id': song_id,
                'indexed_at': firestore.SERVER_TIMESTAMP
            })
            audio, _ = librosa.load(file_path, sr=SAMPLING_RATE, mono=True)
            peaks = self._get_peaks(audio)
            hashes = self._generate_hashes(peaks)
            if not hashes: return False
            batch = self.db.batch()
            count = 0
            for h_val, offset in hashes:
                doc_ref = self.db.collection('song_hashes').document()
                batch.set(doc_ref, {'hash': h_val, 'song_id': song_id, 'offset': offset})
                count += 1
                if count >= FIRESTORE_BATCH_LIMIT:
                    batch.commit()
                    batch = self.db.batch()
                    count = 0
            if count > 0: batch.commit()
            print(f"[Success] Indexed '{song_title}' with {len(hashes)} hashes.")
            return True
        except Exception as e:
            print(f"[Error] Ingestion failed: {e}")
            return False

if __name__ == "__main__":
    detector = FirebaseFingerprinter("service_account.json" if os.path.exists("service_account.json") else None)
    print("1. Index a Song\n2. Detect a Song")
    choice = input("Choice: ")
    if choice == '1':
        path = input("Path: ").strip().strip('"')
        title = input("Title: ").strip()
        detector.index_song(path, title)
    elif choice == '2':
        path = input("Path: ").strip().strip('"')
        detector.detect_song(path)
