import logging
from datetime import datetime, timezone
from google.cloud import firestore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RoyaltyEngine")

class RoyaltyEngine:
    def __init__(self, db=None):
        self.db = db or firestore.Client()
        # Constants for partial payments
        self.PARTIAL_PAYMENT_FACTOR = 0.5  # 50% payout if business is unlicensed (RED)
        self.WARNING_PAYMENT_FACTOR = 1.0  # 100% payout for YELLOW/GREEN

    def process_royalties(self):
        """
        Main engine loop:
        1. Fetches unprocessed usage logs.
        2. Calculates earnings based on compliance status.
        3. Updates artist balances.
        4. Marks logs as processed.
        """
        logger.info("Starting Royalty Processing Engine...")

        # Get unprocessed logs
        logs_ref = self.db.collection('usage_logs').where('processed', '==', False)
        logs = logs_ref.get()

        if not logs:
            logger.info("No new logs to process.")
            return

        # artist_id -> { 'earned': float, 'logs': [doc_refs] }
        artist_updates = {}

        for doc in logs:
            data = doc.to_dict()
            artist_id = data.get('artist_id')
            base_royalty = data.get('royalty_due', 0.0)
            compliance = data.get('compliance_at_detection', 'RED')

            if not artist_id:
                continue

            # Apply Compliance Multiplier
            # Logic: If business is RED (Unlicensed), the artist gets a partial payment.
            # This incentivizes artists to ensure their venues are licensed.
            multiplier = self.PARTIAL_PAYMENT_FACTOR if compliance == 'RED' else self.WARNING_PAYMENT_FACTOR
            actual_earning = base_royalty * multiplier

            if artist_id not in artist_updates:
                artist_updates[artist_id] = {'earned': 0.0, 'log_refs': []}

            artist_updates[artist_id]['earned'] += actual_earning
            artist_updates[artist_id]['log_refs'].append(doc.reference)

        # Batch Update Artists and Logs
        batch = self.db.batch()
        processed_count = 0

        for artist_id, info in artist_updates.items():
            artist_ref = self.db.collection('artists').document(artist_id)

            # Use increment to avoid race conditions
            batch.set(artist_ref, {
                'total_earnings': firestore.Increment(info['earned']),
                'pending_balance': firestore.Increment(info['earned']),
                'last_updated': firestore.SERVER_TIMESTAMP
            }, merge=True)

            for log_ref in info['log_refs']:
                batch.update(log_ref, {
                    'processed': True,
                    'final_payout': base_royalty * (self.PARTIAL_PAYMENT_FACTOR if compliance == 'RED' else 1.0),
                    'processed_at': firestore.SERVER_TIMESTAMP
                })
                processed_count += 1

        batch.commit()
        logger.info(f"Successfully processed {processed_count} logs for {len(artist_updates)} artists.")

    def run_realtime_trigger(self, log_data, log_ref):
        """
        Can be called by a Cloud Function trigger or immediately after detection.
        """
        artist_id = log_data.get('artist_id')
        if not artist_id: return

        compliance = log_data.get('compliance_at_detection', 'RED')
        base_royalty = log_data.get('royalty_due', 0.0)

        multiplier = self.PARTIAL_PAYMENT_FACTOR if compliance == 'RED' else 1.0
        actual_earning = base_royalty * multiplier

        batch = self.db.batch()
        artist_ref = self.db.collection('artists').document(artist_id)
        batch.set(artist_ref, {
            'total_earnings': firestore.Increment(actual_earning),
            'pending_balance': firestore.Increment(actual_earning),
            'last_updated': firestore.SERVER_TIMESTAMP
        }, merge=True)

        batch.update(log_ref, {
            'processed': True,
            'final_payout': actual_earning,
            'processed_at': firestore.SERVER_TIMESTAMP
        })
        batch.commit()

if __name__ == "__main__":
    engine = RoyaltyEngine()
    engine.process_royalties()
