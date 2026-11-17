"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Internal Gateway-to-Microservice HMAC-SHA256 Authentication Module (Milestone 2 - Step 7.4B-1)

Implements cryptographic request verification:
- Canonical request construction: METHOD + "\n" + PATH + "\n" + TIMESTAMP + "\n" + NONCE + "\n" + SHA256(BODY)
- Strict timestamp freshness window (default: 300 seconds)
- In-memory thread-safe Nonce cache for replay attack detection
- Constant-time signature comparison (hmac.compare_digest)
- Key rotation support (active + previous key fallback)
- Never logs or exposes secret values
"""

import hmac
import hashlib
import time
import logging
import threading
from typing import Optional, Tuple, Set, Dict

logger = logging.getLogger("sts_ml_service.hmac_auth")


class NonceReplayStore:
    """
    Thread-safe in-memory cache for request nonces to prevent replay attacks.
    Automatically purges expired nonces beyond the validity window.
    """

    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        # Maps nonce -> expiration_timestamp
        self._nonces: Dict[str, float] = {}

    def is_nonce_seen_or_record(self, nonce: str, current_time: Optional[float] = None) -> bool:
        """
        Atomically checks if a nonce has been seen.
        If already seen, returns True (replay detected).
        If new, records it with an expiration timestamp and returns False.
        """
        now = current_time if current_time is not None else time.time()
        with self._lock:
            # Purge expired entries
            expired = [n for n, exp in self._nonces.items() if exp < now]
            for n in expired:
                del self._nonces[n]

            # Replay check
            if nonce in self._nonces:
                return True

            # Record new nonce
            self._nonces[nonce] = now + self.ttl_seconds
            return False

    def clear(self):
        """Clears all stored nonces (used for testing)."""
        with self._lock:
            self._nonces.clear()


# Global nonce cache instance
nonce_store = NonceReplayStore(ttl_seconds=300)


def build_canonical_string(
    method: str,
    path: str,
    timestamp: str,
    nonce: str,
    body: bytes
) -> str:
    """
    Constructs the canonical string to be signed.
    Format:
        METHOD\n
        PATH\n
        TIMESTAMP\n
        NONCE\n
        SHA256(BODY_BYTES)
    """
    body_sha256 = hashlib.sha256(body).hexdigest()
    canonical_str = (
        f"{method.upper().strip()}\n"
        f"{path.strip()}\n"
        f"{str(timestamp).strip()}\n"
        f"{nonce.strip()}\n"
        f"{body_sha256}"
    )
    return canonical_str


def generate_hmac_signature(
    secret_key: str,
    method: str,
    path: str,
    timestamp: str,
    nonce: str,
    body: bytes
) -> str:
    """
    Generates a hex-encoded HMAC-SHA256 signature for a request.
    Used by client and tests to generate matching signatures.
    """
    if not secret_key:
        raise ValueError("Cannot sign request with empty or missing secret key.")

    canonical_str = build_canonical_string(method, path, timestamp, nonce, body)
    signature = hmac.new(
        secret_key.encode("utf-8"),
        canonical_str.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    return signature


def verify_hmac_request(
    method: str,
    path: str,
    timestamp_str: Optional[str],
    nonce: Optional[str],
    signature: Optional[str],
    body: bytes,
    active_key: Optional[str],
    previous_key: Optional[str] = None,
    replay_window_seconds: int = 300,
    current_time: Optional[float] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validates an incoming internal request signature.

    Returns:
        (True, None) if verified successfully.
        (False, error_code) if verification fails.
    """
    now = current_time if current_time is not None else time.time()

    # 1. Check Secret Key Configuration
    if not active_key or len(active_key.strip()) < 32:
        logger.error("Authentication rejected: STS_INTERNAL_SERVICE_KEY is not configured or too short.")
        return False, "AUTH_KEY_NOT_CONFIGURED"

    # 2. Check Required Headers Presence
    if not timestamp_str or not nonce or not signature:
        logger.warning("Authentication rejected: Missing required security headers (Timestamp, Nonce, or Signature).")
        return False, "MISSING_SECURITY_HEADERS"

    # 3. Timestamp Validation (Replay Window)
    try:
        ts = float(timestamp_str)
    except (ValueError, TypeError):
        return False, "INVALID_TIMESTAMP_FORMAT"

    if abs(now - ts) > replay_window_seconds:
        logger.warning("Authentication rejected: Request timestamp %s outside allowed window of %ds (delta: %fs).", timestamp_str, replay_window_seconds, abs(now - ts))
        return False, "TIMESTAMP_EXPIRED"

    # 4. Nonce Replay Detection
    if nonce_store.is_nonce_seen_or_record(nonce, now):
        logger.warning("Authentication rejected: Replayed nonce detected: %s", nonce)
        return False, "REPLAY_DETECTED"

    # 5. Compute and Compare Signature
    canonical_str = build_canonical_string(method, path, timestamp_str, nonce, body)
    expected_active_sig = hmac.new(
        active_key.encode("utf-8"),
        canonical_str.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    # Constant-time comparison
    if hmac.compare_digest(expected_active_sig, signature):
        return True, None

    # Fallback to previous key for zero-downtime rotation
    if previous_key and len(previous_key.strip()) >= 32:
        expected_prev_sig = hmac.new(
            previous_key.encode("utf-8"),
            canonical_str.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        if hmac.compare_digest(expected_prev_sig, signature):
            logger.info("Request authenticated successfully using previous rotation key.")
            return True, None

    logger.warning("Authentication rejected: Signature mismatch for path %s.", path)
    return False, "INVALID_SIGNATURE"
