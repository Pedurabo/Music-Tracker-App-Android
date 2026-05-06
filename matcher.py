from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def match_query(query_hashes, database_fingerprints):
    """
    Improved matching algorithm using temporal alignment and threshold-based scoring.

    Logic:
    1. For each matching hash: compute time difference delta = db_offset - query_offset
    2. Group matches by (song_id, delta)
    3. Find the most common delta per song (aligned matches)
    4. confidence = aligned_matches / total_query_hashes
    5. Return the song with the highest aligned score and its descriptive match level.
    """

    # Group matches by (song_id, delta)
    # song_delta_counts[song_id][delta] = count
    song_delta_counts = defaultdict(lambda: defaultdict(int))

    total_query_hashes = len(query_hashes)
    if total_query_hashes == 0:
        return None, 0.0, "Unknown"

    total_matches_found = 0
    for q_hash, q_offset in query_hashes:
        if q_hash in database_fingerprints:
            # database_fingerprints[q_hash] is a list of (song_id, db_offset)
            for song_id, db_offset in database_fingerprints[q_hash]:
                # 1. Compute time difference: delta = db_offset - query_offset
                delta = db_offset - q_offset

                # 2. Group matches by (song_id, delta)
                song_delta_counts[song_id][delta] += 1
                total_matches_found += 1

    logger.info(f"Total raw hash matches: {total_matches_found}")

    if not song_delta_counts:
        return None, 0.0, "Unknown"

    best_song_id = None
    max_aligned_matches = 0

    # 3. Find the most common delta per song and global best
    for song_id, deltas in song_delta_counts.items():
        # 4. Score = size of largest aligned group (the "peak" in the delta histogram)
        aligned_matches = max(deltas.values())

        if aligned_matches > max_aligned_matches:
            max_aligned_matches = aligned_matches
            best_song_id = song_id

    # Calculate confidence based on aligned hash density
    # Requirement: confidence = aligned_matches / total_query_hashes
    confidence = max_aligned_matches / total_query_hashes

    # Threshold-based match level
    # Requirement:
    # if confidence < 0.1 -> Unknown
    # if 0.1-0.3 -> Weak match
    # if 0.3-0.7 -> Moderate match
    # if >0.7 -> Strong match
    # Ensure identical files return 1.0
    match_level = "Unknown"
    if confidence >= 0.1:
        if confidence < 0.3:
            match_level = "Weak match"
        elif confidence < 0.7:
            match_level = "Moderate match"
        else:
            match_level = "Strong match"
    else:
        # if confidence < 0.1 -> Unknown
        best_song_id = None
        confidence = 0.0
        match_level = "Unknown"

    logger.info(f"Best Match: {best_song_id}, Confidence: {confidence:.4f}, Level: {match_level}")

    # Return exactly 3 values: song_id, confidence, match_level
    return best_song_id, round(float(confidence), 4), match_level
