import os
import requests
import uuid
import base64
import time
import logging
from datetime import datetime, timedelta

# Configuration & Logging
logger = logging.getLogger("MTN_MoMo")

class MTNMoMoClient:
    def __init__(self):
        # Sandbox credentials - in production, these MUST be in .env
        self.base_url = "https://sandbox.momodeveloper.mtn.com"
        self.subscription_key = os.getenv("MTN_SUBSCRIPTION_KEY")
        self.user_id = os.getenv("MTN_USER_ID")  # X-Reference-Id for API User
        self.api_key = os.getenv("MTN_API_KEY")
        self.environment = os.getenv("MTN_ENVIRONMENT", "sandbox")

        # Token Caching
        self._token = None
        self._token_expiry = None

    def _get_auth_header(self):
        """Generates Basic Auth header for token generation."""
        auth_str = f"{self.user_id}:{self.api_key}"
        encoded_auth = base64.b64encode(auth_str.encode()).decode()
        return {"Authorization": f"Basic {encoded_auth}"}

    def get_token(self):
        """Fetches a valid OAuth token, using cache if available."""
        now = datetime.now()
        if self._token and self._token_expiry and now < self._token_expiry:
            return self._token

        logger.info("Fetching new MTN MoMo access token...")
        url = f"{self.base_url}/collection/token/"
        headers = {
            "Ocp-Apim-Subscription-Key": self.subscription_key,
            **self._get_auth_header()
        }

        try:
            response = requests.post(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            self._token = data.get("access_token")
            # Buffer of 60 seconds for expiry
            expires_in = int(data.get("expires_in", 3600)) - 60
            self._token_expiry = now + timedelta(seconds=expires_in)

            return self._token
        except Exception as e:
            logger.error(f"MTN Token generation failed: {e}")
            return None

    def initiate_mtn_payment(self, phone, amount, external_id=None):
        """
        Sends a 'Request to Pay' to the customer's phone.
        Returns: (success, reference_id or error_message)
        """
        token = self.get_token()
        if not token:
            return False, "Auth failed"

        reference_id = str(uuid.uuid4())
        url = f"{self.base_url}/collection/v1_0/requesttopay"

        headers = {
            "Authorization": f"Bearer {token}",
            "X-Reference-Id": reference_id,
            "X-Target-Environment": self.environment,
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": self.subscription_key
        }

        # Mobile numbers in sandbox usually follow specific formats (e.g., 46733123453)
        # Ensure it is a string and potentially format it for your region
        payload = {
            "amount": str(amount),
            "currency": "EUR", # Sandbox uses EUR
            "externalId": external_id or str(int(time.time())),
            "payer": {
                "partyIdType": "MSISDN",
                "partyId": phone
            },
            "payerMessage": "Subscription Payment",
            "payeeNote": "MusicTracker Licensing"
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            # 202 Accepted is the success code for this async operation
            if response.status_code == 202:
                logger.info(f"Payment prompt sent. Ref: {reference_id}")
                return True, reference_id
            else:
                logger.error(f"MTN Pay Request failed: {response.text}")
                return False, response.text
        except Exception as e:
            logger.error(f"MTN Pay Request error: {e}")
            return False, str(e)

    def check_payment_status(self, reference_id):
        """
        Checks the status of a specific transaction.
        Returns: (status_code, response_data)
        """
        token = self.get_token()
        if not token:
            return "ERROR", {"message": "Auth failed"}

        url = f"{self.base_url}/collection/v1_0/requesttopay/{reference_id}"
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Target-Environment": self.environment,
            "Ocp-Apim-Subscription-Key": self.subscription_key
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            # Status can be: PENDING, SUCCESSFUL, FAILED
            status = data.get("status", "FAILED")
            logger.info(f"MTN Transaction {reference_id} status: {status}")
            return status, data
        except Exception as e:
            logger.error(f"MTN Status check failed: {e}")
            return "ERROR", {"message": str(e)}
