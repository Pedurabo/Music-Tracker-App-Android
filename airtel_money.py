import os
import requests
import time
import logging
import uuid
from datetime import datetime, timedelta

# Configuration & Logging
logger = logging.getLogger("Airtel_Money")

class AirtelMoneyClient:
    def __init__(self):
        # Configuration - Load from environment variables
        self.base_url = os.getenv("AIRTEL_BASE_URL", "https://openapiuat.airtel.africa")
        self.client_id = os.getenv("AIRTEL_CLIENT_ID")
        self.client_secret = os.getenv("AIRTEL_CLIENT_SECRET")
        self.country = os.getenv("AIRTEL_COUNTRY", "UG")
        self.currency = os.getenv("AIRTEL_CURRENCY", "UGX")

        # Token Caching
        self._token = None
        self._token_expiry = None

    def get_token(self):
        """Fetches a valid OAuth token using client_id and client_secret."""
        now = datetime.now()
        if self._token and self._token_expiry and now < self._token_expiry:
            return self._token

        logger.info("Fetching new Airtel access token...")
        url = f"{self.base_url}/auth/oauth2/token"
        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials"
        }
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

            self._token = data.get("access_token")
            # Buffer of 60 seconds
            expires_in = int(data.get("expires_in", 3600)) - 60
            self._token_expiry = now + timedelta(seconds=expires_in)

            return self._token
        except Exception as e:
            logger.error(f"Airtel Auth failed: {e}")
            return None

    def airtel_pay(self, phone, amount):
        """
        Initiates a payment (Collection) request.
        Returns: (success, transaction_id or error_message)
        """
        token = self.get_token()
        if not token:
            return False, "Authentication failed"

        url = f"{self.base_url}/merchant/v1/payments/"

        # Unique ID for this specific request
        external_id = str(uuid.uuid4())

        headers = {
            "Content-Type": "application/json",
            "Accept": "*/*",
            "X-Country": self.country,
            "X-Currency": self.currency,
            "Authorization": f"Bearer {token}"
        }

        payload = {
            "reference": "MusicTracker Payment",
            "subscriber": {
                "msisdn": phone
            },
            "transaction": {
                "amount": amount,
                "id": external_id
            }
        }

        try:
            logger.info(f"Initiating Airtel payment for {phone} amount {amount}")
            response = requests.post(url, json=payload, headers=headers)
            res_data = response.json()

            # Airtel's response structure varies by region, but usually includes a status object
            if response.status_code == 200 and res_data.get('status', {}).get('success'):
                return True, external_id
            else:
                error_msg = res_data.get('status', {}).get('message', 'Request failed')
                logger.error(f"Airtel Pay Error: {error_msg}")
                return False, error_msg
        except Exception as e:
            logger.error(f"Airtel Pay Exception: {e}")
            return False, str(e)

    def airtel_status(self, tx_id):
        """
        Checks the status of a transaction.
        Returns: (status_string, full_response)
        """
        token = self.get_token()
        if not token:
            return "AUTH_FAILED", {}

        url = f"{self.base_url}/standard/v1/payments/{tx_id}"
        headers = {
            "Content-Type": "application/json",
            "Accept": "*/*",
            "X-Country": self.country,
            "X-Currency": self.currency,
            "Authorization": f"Bearer {token}"
        }

        try:
            response = requests.get(url, headers=headers)
            res_data = response.json()

            # Status mapping: TS (Success), TF (Failed), TP (Pending)
            raw_status = res_data.get('data', {}).get('transaction', {}).get('status')

            status_map = {
                "TS": "SUCCESSFUL",
                "TF": "FAILED",
                "TP": "PENDING"
            }

            final_status = status_map.get(raw_status, "UNKNOWN")
            return final_status, res_data
        except Exception as e:
            logger.error(f"Airtel Status Check Error: {e}")
            return "ERROR", {"message": str(e)}

# Plug-and-play helper instance
_client = AirtelMoneyClient()

def airtel_pay(phone, amount):
    return _client.airtel_pay(phone, amount)

def airtel_status(tx_id):
    return _client.airtel_status(tx_id)
