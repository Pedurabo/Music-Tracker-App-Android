import os
import tempfile
import logging
import uuid
import time
from datetime import datetime, timezone, timedelta
from flask import Flask, request, jsonify
from firebase_fingerprinter import FirebaseFingerprinter
from google.cloud import firestore
from mtn_momo import MTNMoMoClient
from airtel_money import airtel_pay, airtel_status

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize Clients
service_json = "service_account.json"
if not os.path.exists(service_json):
    service_json = "service_account.json.json"

fingerprinter = FirebaseFingerprinter(service_json if os.path.exists(service_json) else None)
db = fingerprinter.db
mtn_client = MTNMoMoClient()

def get_compliance_status(business_id):
    """Checks the business subscription status in Firestore."""
    if not business_id or business_id == "unknown_business":
        return "RED"
    try:
        docs = db.collection('subscriptions').where('businessId', '==', business_id).where('status', '==', 'active').get()
        if not docs: return "RED"

        # Get latest by expiry date
        latest_sub = max([d.to_dict() for d in docs], key=lambda x: x.get('expiryDate') or datetime.min.replace(tzinfo=timezone.utc))
        expiry_date = latest_sub.get('expiryDate')

        if not expiry_date: return "GREEN"

        now = datetime.now(timezone.utc)
        if expiry_date.tzinfo is None: expiry_date = expiry_date.replace(tzinfo=timezone.utc)

        delta = expiry_date - now
        if delta.total_seconds() < 0: return "RED"
        if delta.days <= 3: return "YELLOW"
        return "GREEN"
    except Exception as e:
        logger.error(f"Compliance check failed: {e}")
        return "RED"

@app.route('/detect', methods=['POST'])
def detect():
    """Song detection endpoint with business tracking and royalty logging."""
    audio_file = request.files.get('file') or request.files.get('audio')
    business_id = request.form.get('business_id') or "unknown_business"

    if not audio_file: return jsonify({"error": "No audio file"}), 400

    fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
    try:
        compliance_status = get_compliance_status(business_id)

        with os.fdopen(fd, 'wb') as tmp:
            audio_file.save(tmp)

        song_name, confidence, match_level, song_id = fingerprinter.detect_song(tmp_path)

        response_data = {
            "song_name": song_name or "Unknown",
            "confidence": float(confidence),
            "match_level": match_level,
            "song_id": song_id,
            "license_status": compliance_status
        }

        if song_id:
            # 1. Fetch Song Metadata (Artist & Royalty Rate)
            song_doc = db.collection('songs').document(song_id).get()
            s_data = song_doc.to_dict() if song_doc.exists else {}
            artist_id = s_data.get('artist_id', 'unknown_artist')
            royalty_rate = s_data.get('royalty_rate', 0.05) # Default 0.05 per play

            # 2. Log Detection for Royalty Calculation
            detection_ref = db.collection('usage_logs').document()
            detection_ref.set({
                'business_id': business_id,
                'song_id': song_id,
                'artist_id': artist_id,
                'song_name': song_name,
                'timestamp': firestore.SERVER_TIMESTAMP,
                'confidence': confidence,
                'compliance_at_detection': compliance_status,
                'royalty_due': royalty_rate
            })

            response_data.update({
                "artist_id": artist_id,
                "royalties_due": royalty_rate
            })

            # 3. Trigger Fraud Check (Module logic)
            # (In production, this would be a background task)
            check_fraud_logic(business_id, compliance_status)

        return jsonify(response_data)
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

def check_fraud_logic(business_id, status):
    """Internal check for unlicensed high-volume usage."""
    if status == "RED":
        # Check detections in last hour
        one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
        recent_detections = db.collection('usage_logs') \
            .where('business_id', '==', business_id) \
            .where('timestamp', '>=', one_hour_ago).get()

        if len(recent_detections) > 10:
            db.collection('violations').add({
                'business_id': business_id,
                'type': 'UNLICENSED_USAGE',
                'count': len(recent_detections),
                'timestamp': firestore.SERVER_TIMESTAMP,
                'status': 'PENDING'
            })

@app.route('/license-status/<business_id>', methods=['GET'])
def license_check(business_id):
    """Specific endpoint for license status check."""
    return jsonify({
        "business_id": business_id,
        "status": get_compliance_status(business_id)
    })

@app.route('/pay', methods=['POST'])
def pay():
    """Initiates Payment (MTN or Airtel)."""
    data = request.json
    phone = data.get('phone_number')
    amount = data.get('amount')
    business_id = data.get('business_id')
    tier = data.get('tier', 'SMALL_BAR')
    provider = data.get('provider', 'MTN').upper()

    if not all([phone, amount, business_id]):
        return jsonify({"error": "Missing fields"}), 400

    success, reference_id = False, None
    if provider == "AIRTEL":
        success, reference_id = airtel_pay(phone, amount)
    else:
        success, reference_id = mtn_client.initiate_mtn_payment(phone, amount)

    if success:
        db.collection('pending_payments').document(reference_id).set({
            'business_id': business_id, 'tier': tier, 'amount': amount,
            'status': 'PENDING', 'provider': provider, 'created_at': firestore.SERVER_TIMESTAMP
        })
        return jsonify({"status": "initiated", "transaction_id": reference_id}), 202
    return jsonify({"error": reference_id}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
