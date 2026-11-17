"""
STS FINTECH AI FINANCIAL INTELLIGENCE
HMAC-SHA256 Cryptographic Authentication Tests (Milestone 2 - Step 7.4B-1)

Verifies:
1. Canonical string format consistency.
2. Cryptographic signature generation and verification.
3. Timestamp freshness window enforcement (<= 300 seconds).
4. In-memory Nonce replay detection.
5. Constant-time comparison and tamper rejection.
6. Zero-downtime key rotation with fallback key.
7. Rejection of unconfigured or weak (<32 bytes) service keys.
"""

import time
import pytest
from ml.service.security.hmac_auth import (
    build_canonical_string,
    generate_hmac_signature,
    verify_hmac_request,
    NonceReplayStore,
    nonce_store
)

SAMPLE_ACTIVE_KEY = "test_internal_service_key_active_32_bytes_long_secret_12345"
SAMPLE_PREV_KEY = "test_internal_service_key_prev_32_bytes_long_secret_67890"


class TestHmacAuth:
    """Test suite for HMAC request authentication and replay defense."""

    def setup_method(self):
        """Clears global nonce store before each test."""
        nonce_store.clear()

    def test_canonical_string_builder(self):
        """Verifies canonical string generation format."""
        import hashlib
        body = b'{"amount": 100.5}'
        canonical = build_canonical_string(
            method="post",
            path="/internal/v2/predict/expense",
            timestamp="1700000000",
            nonce="nonce-abc-123",
            body=body
        )
        expected_lines = [
            "POST",
            "/internal/v2/predict/expense",
            "1700000000",
            "nonce-abc-123",
            hashlib.sha256(body).hexdigest()
        ]
        assert canonical == "\n".join(expected_lines)

    def test_valid_request_verification(self):
        """Verifies that a validly signed request passes authentication."""
        now = time.time()
        timestamp_str = str(int(now))
        nonce = f"nonce-{now}-unique-1"
        body = b'{"query": "health_check"}'
        path = "/internal/v2/health"
        method = "GET"

        signature = generate_hmac_signature(
            secret_key=SAMPLE_ACTIVE_KEY,
            method=method,
            path=path,
            timestamp=timestamp_str,
            nonce=nonce,
            body=body
        )

        is_valid, err = verify_hmac_request(
            method=method,
            path=path,
            timestamp_str=timestamp_str,
            nonce=nonce,
            signature=signature,
            body=body,
            active_key=SAMPLE_ACTIVE_KEY,
            current_time=now
        )
        assert is_valid is True
        assert err is None

    def test_missing_or_weak_key_rejected(self):
        """Verifies rejection when service key is missing or under 32 characters."""
        now = time.time()
        # Weak key (< 32 chars)
        is_valid, err = verify_hmac_request(
            method="GET",
            path="/internal/v2/health",
            timestamp_str=str(int(now)),
            nonce="nonce-1",
            signature="sig",
            body=b"",
            active_key="short_key",
            current_time=now
        )
        assert is_valid is False
        assert err == "AUTH_KEY_NOT_CONFIGURED"

        # None key
        is_valid, err = verify_hmac_request(
            method="GET",
            path="/internal/v2/health",
            timestamp_str=str(int(now)),
            nonce="nonce-1",
            signature="sig",
            body=b"",
            active_key=None,
            current_time=now
        )
        assert is_valid is False
        assert err == "AUTH_KEY_NOT_CONFIGURED"

    def test_missing_headers_rejected(self):
        """Verifies rejection when required headers are missing."""
        now = time.time()
        # Missing timestamp
        is_valid, err = verify_hmac_request(
            method="GET",
            path="/internal/v2/health",
            timestamp_str=None,
            nonce="nonce-1",
            signature="sig",
            body=b"",
            active_key=SAMPLE_ACTIVE_KEY,
            current_time=now
        )
        assert is_valid is False
        assert err == "MISSING_SECURITY_HEADERS"

        # Missing nonce
        is_valid, err = verify_hmac_request(
            method="GET",
            path="/internal/v2/health",
            timestamp_str=str(int(now)),
            nonce=None,
            signature="sig",
            body=b"",
            active_key=SAMPLE_ACTIVE_KEY,
            current_time=now
        )
        assert is_valid is False
        assert err == "MISSING_SECURITY_HEADERS"

        # Missing signature
        is_valid, err = verify_hmac_request(
            method="GET",
            path="/internal/v2/health",
            timestamp_str=str(int(now)),
            nonce="nonce-1",
            signature=None,
            body=b"",
            active_key=SAMPLE_ACTIVE_KEY,
            current_time=now
        )
        assert is_valid is False
        assert err == "MISSING_SECURITY_HEADERS"

    def test_timestamp_freshness_window_enforcement(self):
        """Verifies that requests older than 300s or in future > 300s are rejected."""
        now = 1000000.0
        active_key = SAMPLE_ACTIVE_KEY

        # Case 1: Fresh request (delta 10s) -> Valid
        fresh_ts = str(int(now - 10))
        sig = generate_hmac_signature(active_key, "GET", "/path", fresh_ts, "nonce-fresh", b"")
        is_valid, err = verify_hmac_request(
            "GET", "/path", fresh_ts, "nonce-fresh", sig, b"", active_key, current_time=now
        )
        assert is_valid is True

        # Case 2: Expired request (delta 301s in past) -> Rejected
        expired_ts = str(int(now - 301))
        sig = generate_hmac_signature(active_key, "GET", "/path", expired_ts, "nonce-exp", b"")
        is_valid, err = verify_hmac_request(
            "GET", "/path", expired_ts, "nonce-exp", sig, b"", active_key, current_time=now
        )
        assert is_valid is False
        assert err == "TIMESTAMP_EXPIRED"

        # Case 3: Future request (delta 305s in future) -> Rejected
        future_ts = str(int(now + 305))
        sig = generate_hmac_signature(active_key, "GET", "/path", future_ts, "nonce-future", b"")
        is_valid, err = verify_hmac_request(
            "GET", "/path", future_ts, "nonce-future", sig, b"", active_key, current_time=now
        )
        assert is_valid is False
        assert err == "TIMESTAMP_EXPIRED"

    def test_nonce_replay_attack_prevention(self):
        """Verifies that reusing the same nonce within the freshness window is rejected."""
        now = 1000000.0
        ts = str(int(now - 5))
        reused_nonce = "replayed-nonce-uuid-999"
        sig = generate_hmac_signature(SAMPLE_ACTIVE_KEY, "GET", "/path", ts, reused_nonce, b"")

        # First request must succeed
        is_valid_1, err_1 = verify_hmac_request(
            "GET", "/path", ts, reused_nonce, sig, b"", SAMPLE_ACTIVE_KEY, current_time=now
        )
        assert is_valid_1 is True
        assert err_1 is None

        # Second request with the same nonce must be detected and rejected
        is_valid_2, err_2 = verify_hmac_request(
            "GET", "/path", ts, reused_nonce, sig, b"", SAMPLE_ACTIVE_KEY, current_time=now
        )
        assert is_valid_2 is False
        assert err_2 == "REPLAY_DETECTED"

    def test_key_rotation_fallback(self):
        """Verifies that request signed with previous key succeeds during rotation."""
        now = time.time()
        ts = str(int(now))
        nonce = f"nonce-rotation-{now}"
        body = b"rotate_payload"

        # Request signed with PREVIOUS key
        prev_sig = generate_hmac_signature(
            secret_key=SAMPLE_PREV_KEY,
            method="POST",
            path="/internal/v2/predict",
            timestamp=ts,
            nonce=nonce,
            body=body
        )

        # Microservice configured with active key + fallback previous key
        is_valid, err = verify_hmac_request(
            method="POST",
            path="/internal/v2/predict",
            timestamp_str=ts,
            nonce=nonce,
            signature=prev_sig,
            body=body,
            active_key=SAMPLE_ACTIVE_KEY,
            previous_key=SAMPLE_PREV_KEY,
            current_time=now
        )
        assert is_valid is True
        assert err is None

    def test_body_tampering_rejected(self):
        """Verifies that altering 1 byte in the payload invalidates the signature."""
        now = time.time()
        ts = str(int(now))
        nonce = "tamper-nonce-1"
        original_body = b'{"amount": 100.00}'
        tampered_body = b'{"amount": 900.00}'

        sig = generate_hmac_signature(
            SAMPLE_ACTIVE_KEY, "POST", "/api", ts, nonce, original_body
        )

        is_valid, err = verify_hmac_request(
            "POST", "/api", ts, nonce, sig, tampered_body, SAMPLE_ACTIVE_KEY, current_time=now
        )
        assert is_valid is False
        assert err == "INVALID_SIGNATURE"
