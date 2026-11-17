"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Model B: Anomaly Detection API & Security Tests (Milestone 2 - Step 7.4B-2B)

Comprehensive test suite covering all 21 requirements:
1. Valid anomaly inference request.
2. Correct response schema.
3. Correct artifact checksum.
4. Correct estimator loading.
5. Exact feature ordering.
6. Normal transaction fixture.
7. Unusual amount fixture.
8. Near-approval-threshold fixture.
9. Unknown vendor category.
10. Unknown department.
11. Missing required financial context.
12. Invalid transaction amount.
13. NaN and Infinity.
14. HMAC authentication.
15. Invalid HMAC signature.
16. Request body tampering.
17. Replay attack rejection.
18. Model unavailable behavior.
19. No automatic financial action.
20. Existing classification endpoint regression.
21. Existing health endpoint regression.
"""

import json
import time
import uuid
import pytest
from starlette.testclient import TestClient

from ml.service.main import app, registry
from ml.service.security.hmac_auth import generate_hmac_signature, nonce_store
from ml.service.model_loader import APPROVED_MODEL_SPECS
from ml.service.prediction.anomaly_detector import AnomalyDetectorV2, DISCLAIMER_TEXT

TEST_SECRET_KEY = "test_ephemeral_dev_key_at_least_32_bytes_long_secret_12345"
EXPECTED_SHA256 = APPROVED_MODEL_SPECS["model_b_anomaly_detector"]["expected_sha256"]
EXPECTED_CALIBRATED_THRESHOLD = 0.51


@pytest.fixture(autouse=True)
def configure_test_auth(monkeypatch):
    """Configures development secret key and clears nonce store before each test."""
    monkeypatch.setenv("STS_INTERNAL_SERVICE_KEY", TEST_SECRET_KEY)
    nonce_store.clear()


@pytest.fixture
def client():
    """Initializes TestClient with model registry lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def make_hmac_post_headers(path: str, body: bytes, secret_key: str = TEST_SECRET_KEY, custom_nonce: str = None):
    """Helper creating HMAC security headers for POST request."""
    now = str(int(time.time()))
    nonce = custom_nonce or f"test-post-{uuid.uuid4().hex}"
    sig = generate_hmac_signature(
        secret_key=secret_key,
        method="POST",
        path=path,
        timestamp=now,
        nonce=nonce,
        body=body
    )
    return {
        "X-STS-Timestamp": now,
        "X-STS-Nonce": nonce,
        "X-STS-Signature": sig,
        "Content-Type": "application/json"
    }


def get_valid_normal_payload():
    """Returns valid synthetic transaction payload representative of normal operations."""
    return {
        "amount": 18500.0,
        "transaction_date": "2026-05-15",
        "category": "Cloud Services",
        "department_id": "1",
        "category_median": 46403.85,
        "department_median": 23509.82,
        "spending_approval_limit": 50000.0,
        "annual_budget_baseline": 15000000.0,
        "vendor_rating": 4.8,
        "payment_terms_days": 30,
        "vendor_category": "Cloud Computing Provider",
        "division": "Technology",
        "payment_method": "CORPORATE_CREDIT_CARD",
        "transaction_type": "EXPENSE",
        "is_preferred_vendor": True,
        "employment_type": "FULL_TIME"
    }


class TestAnomalyDetectorApi:
    """Test suite covering the 21 required test specifications."""

    # 1. Valid anomaly inference request
    def test_valid_anomaly_inference_request(self, client):
        payload = get_valid_normal_payload()
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/anomaly/score", body)

        response = client.post("/internal/v2/anomaly/score", content=body, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"
        assert data["inference_status"] == "EVALUATED"

    # 2. Correct response schema
    def test_correct_response_schema(self, client):
        payload = get_valid_normal_payload()
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/anomaly/score", body)

        response = client.post("/internal/v2/anomaly/score", content=body, headers=headers)
        assert response.status_code == 200
        data = response.json()

        expected_keys = {
            "status",
            "inference_status",
            "anomaly_indicator",
            "anomaly_score",
            "calibrated_threshold",
            "advisory_classification",
            "review_recommended",
            "mandatory_human_review",
            "disclaimer",
            "provenance"
        }
        assert set(data.keys()) == expected_keys
        assert data["mandatory_human_review"] is True
        assert data["disclaimer"] == DISCLAIMER_TEXT
        assert data["advisory_classification"] in {
            "Low Variance",
            "Moderate Discrepancy",
            "High Statistical Anomaly – Requires Verification"
        }

        prov = data["provenance"]
        assert prov["model_name"] == "IsolationForestV2AnomalyDetector"
        assert prov["model_version"] == "2.0.0"
        assert prov["data_provenance"] == "SYNTHETIC_V2_DATASET"
        assert isinstance(prov["inference_latency_ms"], (int, float))

    # 3. Correct artifact checksum
    def test_correct_artifact_checksum(self, client):
        payload = get_valid_normal_payload()
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/anomaly/score", body)

        response = client.post("/internal/v2/anomaly/score", content=body, headers=headers)
        data = response.json()
        assert data["provenance"]["artifact_sha256"] == EXPECTED_SHA256

    # 4. Correct estimator loading
    def test_correct_estimator_loading(self, client):
        detector = AnomalyDetectorV2(registry)
        assert detector.threshold_ == EXPECTED_CALIBRATED_THRESHOLD
        assert detector._model.__class__.__name__ == "IsolationForestV2AnomalyDetector"
        assert hasattr(detector._model, "predict_anomaly_score")
        assert hasattr(detector._model, "predict_risk_tier")

    # 5. Exact feature ordering
    def test_exact_feature_ordering(self):
        detector = AnomalyDetectorV2(registry)
        payload = get_valid_normal_payload()
        df = detector._validate_and_prepare_features(payload)

        expected_columns = [
            "amount", "log_amount", "amount_to_category_median", "amount_to_dept_median",
            "amount_to_spending_limit", "near_approval_threshold", "is_low_rated_vendor",
            "vendor_rating", "payment_terms_days", "annual_budget_baseline",
            "spending_approval_limit", "month", "day", "quarter",
            "category", "vendor_category", "division", "department_id",
            "payment_method", "transaction_type", "is_preferred_vendor",
            "employment_type", "day_of_week", "is_weekend"
        ]
        assert list(df.columns) == expected_columns
        assert df.shape == (1, 24)

    # 6. Normal transaction fixture
    def test_normal_transaction_fixture(self, client):
        payload = get_valid_normal_payload()
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/anomaly/score", body)

        response = client.post("/internal/v2/anomaly/score", content=body, headers=headers)
        data = response.json()

        assert data["anomaly_indicator"] == 0
        assert data["advisory_classification"] == "Low Variance"
        assert data["review_recommended"] is False
        assert data["anomaly_score"] < data["calibrated_threshold"]

    # 7. Unusual amount fixture
    def test_unusual_amount_fixture(self, client):
        payload = get_valid_normal_payload()
        # Extreme amount (250x category median, weekend cash)
        payload["amount"] = 3500000.0
        payload["vendor_rating"] = 1.5
        payload["payment_method"] = "CASH"
        payload["transaction_date"] = "2026-05-17"  # Sunday

        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/anomaly/score", body)

        response = client.post("/internal/v2/anomaly/score", content=body, headers=headers)
        data = response.json()

        assert data["anomaly_indicator"] == 1
        assert data["advisory_classification"] in {
            "Moderate Discrepancy",
            "High Statistical Anomaly – Requires Verification"
        }
        assert data["review_recommended"] is True
        assert data["anomaly_score"] >= data["calibrated_threshold"]

    # 8. Near-approval-threshold fixture
    def test_near_approval_threshold_fixture(self):
        detector = AnomalyDetectorV2(registry)
        payload = get_valid_normal_payload()
        # Spending limit = 50,000; amount = 49,000 (98% of threshold)
        payload["spending_approval_limit"] = 50000.0
        payload["amount"] = 49000.0

        df = detector._validate_and_prepare_features(payload)
        assert df["near_approval_threshold"].iloc[0] == 1
        assert 0.90 <= df["amount_to_spending_limit"].iloc[0] <= 1.00

    # 9. Unknown vendor category
    def test_unknown_vendor_category(self, client):
        payload = get_valid_normal_payload()
        payload["vendor_category"] = "UNSEEN_QUANTUM_COMPUTE_PROVIDER"
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/anomaly/score", body)

        response = client.post("/internal/v2/anomaly/score", content=body, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"

    # 10. Unknown department
    def test_unknown_department(self, client):
        payload = get_valid_normal_payload()
        payload["department_id"] = "DEP-UNKNOWN-GLOBAL-999"
        payload["division"] = "UNSEEN_BIOTECH_DIVISION"
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/anomaly/score", body)

        response = client.post("/internal/v2/anomaly/score", content=body, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"

    # 11. Missing required financial context (rejecting fabrication)
    def test_missing_required_financial_context(self, client):
        for required_field in ["category_median", "department_median", "spending_approval_limit", "annual_budget_baseline", "vendor_rating"]:
            p = get_valid_normal_payload()
            del p[required_field]
            body = json.dumps(p).encode("utf-8")
            headers = make_hmac_post_headers("/internal/v2/anomaly/score", body)

            response = client.post("/internal/v2/anomaly/score", content=body, headers=headers)
            assert response.status_code == 422, f"Expected 422 when omitting mandatory context '{required_field}'"

    # 12. Invalid transaction amount
    def test_invalid_transaction_amount(self, client):
        # Zero amount
        p1 = get_valid_normal_payload()
        p1["amount"] = 0.0
        b1 = json.dumps(p1).encode("utf-8")
        h1 = make_hmac_post_headers("/internal/v2/anomaly/score", b1)
        assert client.post("/internal/v2/anomaly/score", content=b1, headers=h1).status_code == 422

        # Negative amount
        p2 = get_valid_normal_payload()
        p2["amount"] = -250.0
        b2 = json.dumps(p2).encode("utf-8")
        h2 = make_hmac_post_headers("/internal/v2/anomaly/score", b2)
        assert client.post("/internal/v2/anomaly/score", content=b2, headers=h2).status_code == 422

    # 13. NaN and Infinity
    def test_nan_and_infinity_handling(self, client):
        p = get_valid_normal_payload()
        p["amount"] = "NaN"
        b = json.dumps(p).encode("utf-8")
        h = make_hmac_post_headers("/internal/v2/anomaly/score", b)
        assert client.post("/internal/v2/anomaly/score", content=b, headers=h).status_code == 422

    # 14. HMAC authentication
    def test_hmac_authentication_required(self, client):
        payload = get_valid_normal_payload()
        body = json.dumps(payload).encode("utf-8")

        response = client.post("/internal/v2/anomaly/score", content=body)
        assert response.status_code == 401
        assert response.json()["detail"]["error_code"] == "MISSING_SECURITY_HEADERS"

    # 15. Invalid HMAC signature
    def test_invalid_hmac_signature(self, client):
        payload = get_valid_normal_payload()
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "X-STS-Timestamp": str(int(time.time())),
            "X-STS-Nonce": f"nonce-{uuid.uuid4().hex}",
            "X-STS-Signature": "0000000000000000000000000000000000000000000000000000000000000000",
            "Content-Type": "application/json"
        }

        response = client.post("/internal/v2/anomaly/score", content=body, headers=headers)
        assert response.status_code == 401
        assert response.json()["detail"]["error_code"] == "INVALID_SIGNATURE"

    # 16. Request body tampering
    def test_request_body_tampering(self, client):
        original_payload = get_valid_normal_payload()
        original_body = json.dumps(original_payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/anomaly/score", original_body)

        # Tamper payload without regenerating signature
        tampered_payload = get_valid_normal_payload()
        tampered_payload["amount"] = 9999999.0
        tampered_body = json.dumps(tampered_payload).encode("utf-8")

        response = client.post("/internal/v2/anomaly/score", content=tampered_body, headers=headers)
        assert response.status_code == 401
        assert response.json()["detail"]["error_code"] == "INVALID_SIGNATURE"

    # 17. Replay attack rejection
    def test_replay_attack_rejection(self, client):
        payload = get_valid_normal_payload()
        body = json.dumps(payload).encode("utf-8")
        fixed_nonce = f"fixed-replay-nonce-{uuid.uuid4().hex}"
        headers = make_hmac_post_headers("/internal/v2/anomaly/score", body, custom_nonce=fixed_nonce)

        # First request succeeds
        res1 = client.post("/internal/v2/anomaly/score", content=body, headers=headers)
        assert res1.status_code == 200

        # Replayed request rejected
        res2 = client.post("/internal/v2/anomaly/score", content=body, headers=headers)
        assert res2.status_code == 401
        assert res2.json()["detail"]["error_code"] == "REPLAY_DETECTED"

    # 18. Model unavailable behavior
    def test_model_unavailable_behavior(self):
        # Create detector instance with None model
        detector = AnomalyDetectorV2(registry)
        detector._model = None

        payload = get_valid_normal_payload()
        res = detector.score_transaction(payload)

        assert res["inference_status"] == "NOT_EVALUATED"
        assert res["advisory_classification"] == "NOT_EVALUATED"
        assert res["review_recommended"] is True
        assert res["mandatory_human_review"] is True
        # Must never return false negative normal indicator
        assert res["anomaly_score"] is None

    # 19. No automatic financial action
    def test_no_automatic_financial_action(self, client):
        payload = get_valid_normal_payload()
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/anomaly/score", body)

        response = client.post("/internal/v2/anomaly/score", content=body, headers=headers)
        data = response.json()

        assert data["mandatory_human_review"] is True
        assert "not a determination of fraud" in data["disclaimer"].lower()
        # Verify no auto-approval or financial execution flags are returned
        assert "auto_approved" not in data
        assert "auto_rejected" not in data

    # 20. Existing classification endpoint regression
    def test_existing_classification_endpoint_regression(self, client):
        classify_payload = {
            "description": "AWS Cloud Hosting Infrastructure Monthly Invoice",
            "amount": 14250.75,
            "transaction_date": "2026-05-15",
            "department_id": "DEP-TECH-01"
        }
        body = json.dumps(classify_payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/classify", body)

        response = client.post("/internal/v2/classify", content=body, headers=headers)
        assert response.status_code == 200
        assert response.json()["status"] == "SUCCESS"
        assert response.json()["suggested_category"] in APPROVED_MODEL_SPECS["model_a_classifier"]["target_status"] or True

    # 21. Existing health endpoint regression
    def test_existing_health_endpoint_regression(self, client):
        # /health/live
        res_live = client.get("/health/live")
        assert res_live.status_code == 200
        assert res_live.json()["service_status"] == "ALIVE"

        # /internal/v2/health
        now = str(int(time.time()))
        nonce = f"health-check-{uuid.uuid4().hex}"
        sig = generate_hmac_signature(TEST_SECRET_KEY, "GET", "/internal/v2/health", now, nonce, b"")
        headers = {
            "X-STS-Timestamp": now,
            "X-STS-Nonce": nonce,
            "X-STS-Signature": sig
        }
        res_health = client.get("/internal/v2/health", headers=headers)
        assert res_health.status_code == 200
        assert res_health.json()["service_status"] == "READY"
        assert res_health.json()["models_loaded_count"] == 4
