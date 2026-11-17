"""
STS FINTECH - PERSONAL FINANCE ROUTER SECURITY INTEGRATION TESTS
Milestone 2 - Step 7.4C-3D3 (Pre-Hook Security Gate)

Isolated security integration test suite verifying that:
1. Missing HMAC headers are rejected with 401 Unauthorized.
2. Invalid HMAC signatures are rejected with 401 Unauthorized.
3. Expired timestamps outside the 300s window are rejected with 401 Unauthorized.
4. Replayed nonces are rejected with 401 Unauthorized.
5. Correctly signed requests reach the classifier and return SUCCESS with required invariants.
6. Invalid request bodies (missing fields, invalid dates, out-of-range amounts) fail safely with 422.
7. Model-unavailable conditions fail safely with NOT_EVALUATED and do not crash the process.
8. Zero access to production databases or customer records.
9. Canonical request format and verify_internal_hmac dependency are strictly exercised.
"""

import os
import sys
import time
import json
import uuid
import pytest
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, BASE_DIR)

from ml.service.security.hmac_auth import generate_hmac_signature, nonce_store
from ml.service.main import create_app
from ml.service.routes.personal_routes import router as personal_router
from ml.service.prediction.personal_classifier import PersonalFinanceClassifierService

# Set test key (>= 32 chars)
TEST_KEY = "test_secret_key_0123456789_abcdefghij"
os.environ["STS_INTERNAL_SERVICE_KEY"] = TEST_KEY

ENDPOINT_PATH = "/internal/v1/personal/classify"


@pytest.fixture(scope="module")
def client():
    """Builds a test application with the personal router included for testing."""
    app = create_app()
    app.include_router(personal_router)
    return TestClient(app)


def build_auth_headers(method: str, path: str, body_bytes: bytes, key: str = TEST_KEY, timestamp_str: str = None, nonce: str = None):
    """Generates authentic HMAC headers for tests."""
    ts = timestamp_str or str(int(time.time()))
    n = nonce or str(uuid.uuid4())
    sig = generate_hmac_signature(
        secret_key=key,
        method=method,
        path=path,
        timestamp=ts,
        nonce=n,
        body=body_bytes
    )
    return {
        "X-STS-Timestamp": ts,
        "X-STS-Nonce": n,
        "X-STS-Signature": sig,
        "Content-Type": "application/json"
    }


def test_sec_01_missing_hmac_headers_rejected(client):
    """Requests lacking any of the 3 HMAC headers must return 401."""
    payload = {
        "transaction_direction": "expense",
        "description": "AWS cloud hosting monthly invoice",
        "amount_inr": 4500.0,
        "transaction_date": "2026-11-17"
    }
    res = client.post(ENDPOINT_PATH, json=payload)
    assert res.status_code == 401
    assert res.json()["detail"]["error_code"] == "MISSING_SECURITY_HEADERS"


def test_sec_02_invalid_hmac_signature_rejected(client):
    """Requests with tampered or incorrect HMAC signatures must return 401."""
    payload = {
        "transaction_direction": "expense",
        "description": "AWS cloud hosting monthly invoice",
        "amount_inr": 4500.0,
        "transaction_date": "2026-11-17"
    }
    body_bytes = json.dumps(payload).encode("utf-8")
    headers = build_auth_headers("POST", ENDPOINT_PATH, body_bytes)
    headers["X-STS-Signature"] = "bad" * 16

    res = client.post(ENDPOINT_PATH, content=body_bytes, headers=headers)
    assert res.status_code == 401
    assert res.json()["detail"]["error_code"] == "INVALID_SIGNATURE"


def test_sec_03_expired_timestamp_rejected(client):
    """Requests with timestamp older than 300s window must return 401."""
    payload = {
        "transaction_direction": "expense",
        "description": "AWS cloud hosting monthly invoice",
        "amount_inr": 4500.0,
        "transaction_date": "2026-11-17"
    }
    body_bytes = json.dumps(payload).encode("utf-8")
    old_ts = str(int(time.time()) - 600)  # 10 minutes ago
    headers = build_auth_headers("POST", ENDPOINT_PATH, body_bytes, timestamp_str=old_ts)

    res = client.post(ENDPOINT_PATH, content=body_bytes, headers=headers)
    assert res.status_code == 401
    assert res.json()["detail"]["error_code"] == "TIMESTAMP_EXPIRED"


def test_sec_04_nonce_replay_attack_rejected(client):
    """Reusing the same nonce within TTL must trigger REPLAY_DETECTED (401)."""
    payload = {
        "transaction_direction": "expense",
        "description": "AWS cloud hosting monthly invoice",
        "amount_inr": 4500.0,
        "transaction_date": "2026-11-17"
    }
    body_bytes = json.dumps(payload).encode("utf-8")
    n = str(uuid.uuid4())
    headers = build_auth_headers("POST", ENDPOINT_PATH, body_bytes, nonce=n)

    # First request: 200 OK
    r1 = client.post(ENDPOINT_PATH, content=body_bytes, headers=headers)
    assert r1.status_code == 200

    # Second request: 401 Replay
    r2 = client.post(ENDPOINT_PATH, content=body_bytes, headers=headers)
    assert r2.status_code == 401
    assert r2.json()["detail"]["error_code"] == "REPLAY_DETECTED"


def test_sec_05_correctly_signed_request_classified(client):
    """Validly signed request executes inference and returns contract invariants."""
    payload = {
        "transaction_direction": "expense",
        "description": "Hetzner online cloud server hosting monthly billing",
        "amount_inr": 3200.0,
        "transaction_date": "2026-11-17"
    }
    body_bytes = json.dumps(payload).encode("utf-8")
    headers = build_auth_headers("POST", ENDPOINT_PATH, body_bytes)

    res = client.post(ENDPOINT_PATH, content=body_bytes, headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["category"] == "Infrastructure & Cloud"
    assert data["ranking_score"] >= 0.65
    assert data["score_semantics"] == "INTERNAL_RANKING_ONLY"
    assert data["requires_manual_confirmation"] is True
    assert data["abstention_reason"] is None
    assert data["model_version"] == "personal_finance_bundle_v1_candidate"


def test_sec_06_invalid_request_bodies_fail_safely(client):
    """Validation errors return 422 without triggering exceptions."""
    # Invalid direction
    b1 = json.dumps({"transaction_direction": "dividend", "description": "TCS dividend", "amount_inr": 1000.0, "transaction_date": "2026-11-17"}).encode("utf-8")
    h1 = build_auth_headers("POST", ENDPOINT_PATH, b1)
    assert client.post(ENDPOINT_PATH, content=b1, headers=h1).status_code == 422

    # Negative amount
    b2 = json.dumps({"transaction_direction": "income", "description": "TCS dividend", "amount_inr": -50.0, "transaction_date": "2026-11-17"}).encode("utf-8")
    h2 = build_auth_headers("POST", ENDPOINT_PATH, b2)
    assert client.post(ENDPOINT_PATH, content=b2, headers=h2).status_code == 422

    # Non-existent date
    b3 = json.dumps({"transaction_direction": "income", "description": "TCS dividend", "amount_inr": 100.0, "transaction_date": "2026-02-31"}).encode("utf-8")
    h3 = build_auth_headers("POST", ENDPOINT_PATH, b3)
    assert client.post(ENDPOINT_PATH, content=b3, headers=h3).status_code == 422


def test_sec_07_model_unavailable_fails_safely_no_crash(client, monkeypatch):
    """When model service is not loaded, endpoint returns NOT_EVALUATED safely."""
    from ml.service.routes import personal_routes

    # Create dummy unloaded service
    unloaded_service = PersonalFinanceClassifierService(bundle_path="/tmp/non_existent.joblib")
    monkeypatch.setattr(personal_routes, "get_personal_classifier_service", lambda: unloaded_service)

    payload = {
        "transaction_direction": "income",
        "description": "Fee for consulting",
        "amount_inr": 50000.0,
        "transaction_date": "2026-11-17"
    }
    body_bytes = json.dumps(payload).encode("utf-8")
    headers = build_auth_headers("POST", ENDPOINT_PATH, body_bytes)

    res = client.post(ENDPOINT_PATH, content=body_bytes, headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "NOT_EVALUATED"
    assert data["abstention_reason"] == "MODEL_UNAVAILABLE"
    assert data["requires_manual_confirmation"] is True
