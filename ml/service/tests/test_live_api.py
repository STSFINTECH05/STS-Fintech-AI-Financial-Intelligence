"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Live FastAPI Microservice HTTP & Security Verification (Milestone 2 - Step 7.4B-1A)

Verifies using FastAPI TestClient:
1. Public liveness probe (/health/live) returns 200 without authentication.
2. Missing HMAC headers on protected routes returns 401 UNAUTHORIZED.
3. Invalid / tampered HMAC signatures return 401 UNAUTHORIZED.
4. Expired timestamps (> 300s) return 401 UNAUTHORIZED.
5. Replayed nonces return 401 UNAUTHORIZED.
6. Valid HMAC on /internal/v2/health returns 200 READY with SYNTHETIC_V2_DATASET provenance.
7. Valid HMAC on /internal/v2/models/status returns 200 with approved, deferred, and excluded models.
8. Zero filesystem path leakage in responses.
"""

import time
import uuid
import pytest
from starlette.testclient import TestClient

from ml.service.main import app
from ml.service.security.hmac_auth import generate_hmac_signature, nonce_store

TEST_SECRET_KEY = "test_ephemeral_dev_key_at_least_32_bytes_long_secret_12345"


@pytest.fixture(autouse=True)
def configure_test_auth(monkeypatch):
    """Sets development secret key and clears nonce store before each test."""
    monkeypatch.setenv("STS_INTERNAL_SERVICE_KEY", TEST_SECRET_KEY)
    nonce_store.clear()


@pytest.fixture
def client():
    """Initializes TestClient with lifespan context (loading approved models)."""
    with TestClient(app) as test_client:
        yield test_client


def make_hmac_headers(method: str, path: str, body: bytes = b"", secret_key: str = TEST_SECRET_KEY, timestamp_offset: float = 0.0, custom_nonce: str = None):
    """Helper to generate valid HMAC security headers."""
    now = time.time() + timestamp_offset
    timestamp_str = str(int(now))
    nonce = custom_nonce or f"test-nonce-{uuid.uuid4().hex}"
    signature = generate_hmac_signature(
        secret_key=secret_key,
        method=method,
        path=path,
        timestamp=timestamp_str,
        nonce=nonce,
        body=body
    )
    return {
        "X-STS-Timestamp": timestamp_str,
        "X-STS-Nonce": nonce,
        "X-STS-Signature": signature
    }


class TestLiveApiEndpoints:
    """Test suite for live HTTP endpoints using TestClient."""

    def test_health_live_public(self, client):
        """Public liveness probe succeeds without authentication."""
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["service_status"] == "ALIVE"
        assert "timestamp" in data

    def test_internal_health_missing_headers_rejected(self, client):
        """Internal health probe rejects request without security headers."""
        response = client.get("/internal/v2/health")
        assert response.status_code == 401
        data = response.json()
        assert data["detail"]["error_code"] == "MISSING_SECURITY_HEADERS"

    def test_internal_health_invalid_signature_rejected(self, client):
        """Internal health probe rejects request with forged/invalid signature."""
        headers = {
            "X-STS-Timestamp": str(int(time.time())),
            "X-STS-Nonce": f"nonce-{uuid.uuid4().hex}",
            "X-STS-Signature": "0000000000000000000000000000000000000000000000000000000000000000"
        }
        response = client.get("/internal/v2/health", headers=headers)
        assert response.status_code == 401
        data = response.json()
        assert data["detail"]["error_code"] == "INVALID_SIGNATURE"

    def test_internal_health_expired_timestamp_rejected(self, client):
        """Internal health probe rejects requests outside 300-second freshness window."""
        headers = make_hmac_headers("GET", "/internal/v2/health", timestamp_offset=-350.0)
        response = client.get("/internal/v2/health", headers=headers)
        assert response.status_code == 401
        data = response.json()
        assert data["detail"]["error_code"] == "TIMESTAMP_EXPIRED"

    def test_internal_health_valid_signature_success(self, client):
        """Internal health probe returns 200 READY with synthetic provenance when authenticated."""
        headers = make_hmac_headers("GET", "/internal/v2/health")
        response = client.get("/internal/v2/health", headers=headers)
        assert response.status_code == 200
        data = response.json()

        assert data["service_status"] == "READY"
        assert data["model_version"] == "2.0.0"
        assert data["models_loaded_count"] == 4
        assert data["artifact_verification_status"] == "ALL_HASHES_VERIFIED"
        assert data["data_provenance"] == "SYNTHETIC_V2_DATASET"
        assert "timestamp" in data

    def test_nonce_replay_attack_rejected(self, client):
        """Replaying a request with the exact same nonce within the validity window is rejected."""
        fixed_nonce = f"replayed-nonce-{uuid.uuid4().hex}"
        headers = make_hmac_headers("GET", "/internal/v2/health", custom_nonce=fixed_nonce)

        # First request succeeds
        res1 = client.get("/internal/v2/health", headers=headers)
        assert res1.status_code == 200

        # Second request with identical nonce is rejected
        res2 = client.get("/internal/v2/health", headers=headers)
        assert res2.status_code == 401
        data2 = res2.json()
        assert data2["detail"]["error_code"] == "REPLAY_DETECTED"

    def test_internal_models_status_authenticated(self, client):
        """Internal models status returns full model suite metadata with strict security filtering."""
        headers = make_hmac_headers("GET", "/internal/v2/models/status")
        response = client.get("/internal/v2/models/status", headers=headers)
        assert response.status_code == 200
        data = response.json()

        assert data["service_status"] == "READY"
        assert data["model_version"] == "2.0.0"
        assert data["data_provenance"] == "SYNTHETIC_V2_DATASET"

        models = data["models"]
        # Approved active models
        for approved_id in ["model_a_classifier", "model_b_anomaly_detector", "model_c_forecaster", "model_d_budget_predictor"]:
            assert approved_id in models
            assert models[approved_id]["status"] == "LOADED"
            assert models[approved_id]["sha256_verified"] is True
            assert models[approved_id]["data_provenance"] == "SYNTHETIC_V2_DATASET"

        # Excluded models
        assert models["model_b_supervised_benchmark"]["status"] == "EXCLUDED"
        assert models["model_e2_late_payment"]["status"] == "EXCLUDED"

        # Deferred models
        assert models["model_e1_duplicate_matcher"]["status"] == "DEFERRED"
        assert models["model_e3_vendor_screening"]["status"] == "DEFERRED"

        # Security check: Zero filesystem paths leaked
        assert "/Users/" not in response.text
        assert ".joblib" not in response.text
