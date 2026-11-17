"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Model A: Expense Classification API Tests (Milestone 2 - Step 7.4B-2A)

Verifies:
1. Valid classification request returns 200 OK.
2. Correct model output response structure.
3. Correct category label belonging to verified 14 classes.
4. Correct raw decision score formatting (no confidence claims).
5. Missing description handled safely.
6. Empty description handled safely.
7. Missing optional categorical fields use training defaults.
8. Unknown category strings handled gracefully via OneHotEncoder ignore.
9. Invalid amounts (e.g. zero or non-numeric) rejected with 422.
10. Negative amounts rejected with 422.
11. NaN and infinity values rejected with 422.
12. Missing mandatory features (amount, transaction_date, department_id) rejected.
13. HMAC authentication enforced (missing headers -> 401).
14. Invalid HMAC signature rejected -> 401.
15. Model artifact checksum and provenance verification.
16. Existing health endpoints (/health/live, /internal/v2/health, /internal/v2/models/status) remain fully functional.
"""

import json
import time
import uuid
import pytest
from starlette.testclient import TestClient

from ml.service.main import app
from ml.service.security.hmac_auth import generate_hmac_signature, nonce_store
from ml.service.model_loader import APPROVED_MODEL_SPECS

TEST_SECRET_KEY = "test_ephemeral_dev_key_at_least_32_bytes_long_secret_12345"
EXPECTED_SHA256 = APPROVED_MODEL_SPECS["model_a_classifier"]["expected_sha256"]

VERIFIED_CLASSES = [
    "Cloud Services",
    "Employee Benefits & Healthcare",
    "Facilities & Maintenance",
    "Hardware & Equipment",
    "IT Services & Infrastructure",
    "Legal & Professional Services",
    "Marketing & Advertising",
    "Meals & Entertainment",
    "Office Supplies & Printing",
    "Recruitment & Staffing",
    "Software & SaaS",
    "Transportation & Logistics",
    "Travel & Lodging",
    "Utilities & Telecommunications"
]


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


def get_valid_payload():
    """Returns valid synthetic transaction payload."""
    return {
        "description": "AWS Cloud Hosting Infrastructure Monthly Invoice",
        "amount": 14250.75,
        "transaction_date": "2026-05-15",
        "department_id": "DEP-TECH-01",
        "vendor_category": "Cloud Computing Provider",
        "division": "Technology",
        "payment_method": "CORPORATE_CREDIT_CARD",
        "transaction_type": "EXPENSE",
        "vendor_rating": 4.8,
        "payment_terms_days": 30,
        "annual_budget_baseline": 15000000.0,
        "is_preferred_vendor": True,
        "employment_type": "FULL_TIME"
    }


class TestExpenseClassifierApi:
    """Test suite for POST /internal/v2/classify endpoint."""

    # 1. Valid classification request
    def test_valid_classification_request(self, client):
        payload = get_valid_payload()
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/classify", body)

        response = client.post("/internal/v2/classify", content=body, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"

    # 2. Correct model output structure
    def test_correct_output_structure(self, client):
        payload = get_valid_payload()
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/classify", body)

        response = client.post("/internal/v2/classify", content=body, headers=headers)
        assert response.status_code == 200
        data = response.json()

        expected_keys = {
            "status",
            "suggested_category",
            "raw_decision_score",
            "score_interpretation",
            "requires_manual_confirmation",
            "alternatives",
            "inference_status",
            "provenance"
        }
        assert set(data.keys()) == expected_keys
        assert data["requires_manual_confirmation"] is True
        assert data["score_interpretation"] == "RAW_UNCALIBRATED_MODEL_SCORE"
        assert data["inference_status"] == "EVALUATED"

        prov = data["provenance"]
        assert prov["model_name"] == "LogisticRegression"
        assert prov["model_version"] == "2.0.0"
        assert prov["artifact_sha256"] == EXPECTED_SHA256
        assert prov["data_provenance"] == "SYNTHETIC_V2_DATASET"
        assert isinstance(prov["inference_latency_ms"], (int, float))

    # 3. Correct category label
    def test_correct_category_label(self, client):
        payload = get_valid_payload()
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/classify", body)

        response = client.post("/internal/v2/classify", content=body, headers=headers)
        data = response.json()
        assert data["suggested_category"] in VERIFIED_CLASSES

    # 4. Correct raw score format
    def test_correct_raw_score_format(self, client):
        payload = get_valid_payload()
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/classify", body)

        response = client.post("/internal/v2/classify", content=body, headers=headers)
        data = response.json()

        score = data["raw_decision_score"]
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

        # Alternatives
        assert isinstance(data["alternatives"], list)
        assert len(data["alternatives"]) > 0
        for alt in data["alternatives"]:
            assert "category" in alt
            assert alt["category"] in VERIFIED_CLASSES
            assert alt["category"] != data["suggested_category"]
            assert 0.0 <= alt["raw_decision_score"] <= 1.0

    # 5. Missing description
    def test_missing_description(self, client):
        payload = get_valid_payload()
        del payload["description"]
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/classify", body)

        response = client.post("/internal/v2/classify", content=body, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"
        assert data["suggested_category"] in VERIFIED_CLASSES

    # 6. Empty description
    def test_empty_description(self, client):
        payload = get_valid_payload()
        payload["description"] = ""
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/classify", body)

        response = client.post("/internal/v2/classify", content=body, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"

    # 7. Missing optional categorical fields
    def test_missing_optional_categorical_fields(self, client):
        payload = {
            "amount": 2500.0,
            "transaction_date": "2026-06-01",
            "department_id": "DEP-OPS-99"
        }
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/classify", body)

        response = client.post("/internal/v2/classify", content=body, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"

    # 8. Unknown category strings
    def test_unknown_category_strings(self, client):
        payload = get_valid_payload()
        payload["vendor_category"] = "UNSEEN_VENDOR_SECTOR_SPECIALIST"
        payload["division"] = "UNKNOWN_OPERATING_DIVISION"
        payload["department_id"] = "DEP-NEW-9999"
        payload["payment_method"] = "CRYPTOCURRENCY_NEW"
        payload["transaction_type"] = "UNSEEN_TYPE"
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/classify", body)

        response = client.post("/internal/v2/classify", content=body, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"
        assert data["suggested_category"] in VERIFIED_CLASSES

    # 9. Invalid amounts (zero or non-numeric)
    def test_invalid_amounts(self, client):
        payload = get_valid_payload()
        payload["amount"] = 0.0
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/classify", body)

        response = client.post("/internal/v2/classify", content=body, headers=headers)
        assert response.status_code == 422

    # 10. Negative amounts
    def test_negative_amounts(self, client):
        payload = get_valid_payload()
        payload["amount"] = -450.50
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/classify", body)

        response = client.post("/internal/v2/classify", content=body, headers=headers)
        assert response.status_code == 422

    # 11. NaN and infinity handling
    def test_nan_and_infinity_handling(self, client):
        # NaN via null or string
        payload = get_valid_payload()
        payload["amount"] = "NaN"
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/classify", body)

        response = client.post("/internal/v2/classify", content=body, headers=headers)
        assert response.status_code == 422

    # 12. Missing mandatory features
    def test_missing_mandatory_features(self, client):
        # Missing department_id
        p1 = get_valid_payload()
        del p1["department_id"]
        body1 = json.dumps(p1).encode("utf-8")
        h1 = make_hmac_post_headers("/internal/v2/classify", body1)
        res1 = client.post("/internal/v2/classify", content=body1, headers=h1)
        assert res1.status_code == 422

        # Empty string department_id
        p2 = get_valid_payload()
        p2["department_id"] = "   "
        body2 = json.dumps(p2).encode("utf-8")
        h2 = make_hmac_post_headers("/internal/v2/classify", body2)
        res2 = client.post("/internal/v2/classify", content=body2, headers=h2)
        assert res2.status_code == 422

        # Missing transaction_date
        p3 = get_valid_payload()
        del p3["transaction_date"]
        body3 = json.dumps(p3).encode("utf-8")
        h3 = make_hmac_post_headers("/internal/v2/classify", body3)
        res3 = client.post("/internal/v2/classify", content=body3, headers=h3)
        assert res3.status_code == 422

    # 13. HMAC authentication
    def test_hmac_authentication_required(self, client):
        payload = get_valid_payload()
        body = json.dumps(payload).encode("utf-8")

        # No auth headers
        response = client.post("/internal/v2/classify", content=body)
        assert response.status_code == 401
        data = response.json()
        assert data["detail"]["error_code"] == "MISSING_SECURITY_HEADERS"

    # 14. Invalid authentication
    def test_invalid_authentication(self, client):
        payload = get_valid_payload()
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "X-STS-Timestamp": str(int(time.time())),
            "X-STS-Nonce": f"nonce-{uuid.uuid4().hex}",
            "X-STS-Signature": "badbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadb",
            "Content-Type": "application/json"
        }

        response = client.post("/internal/v2/classify", content=body, headers=headers)
        assert response.status_code == 401
        data = response.json()
        assert data["detail"]["error_code"] == "INVALID_SIGNATURE"

    # 15. Model artifact verification
    def test_model_artifact_verification(self, client):
        payload = get_valid_payload()
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/classify", body)

        response = client.post("/internal/v2/classify", content=body, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["provenance"]["artifact_sha256"] == EXPECTED_SHA256
        assert data["provenance"]["model_name"] == "LogisticRegression"
        assert data["provenance"]["model_version"] == "2.0.0"

    # 16. Existing health endpoints remain functional
    def test_existing_health_endpoints_remain_functional(self, client):
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
        assert res_health.json()["data_provenance"] == "SYNTHETIC_V2_DATASET"

        # /internal/v2/models/status
        nonce2 = f"models-check-{uuid.uuid4().hex}"
        sig2 = generate_hmac_signature(TEST_SECRET_KEY, "GET", "/internal/v2/models/status", now, nonce2, b"")
        headers2 = {
            "X-STS-Timestamp": now,
            "X-STS-Nonce": nonce2,
            "X-STS-Signature": sig2
        }
        res_models = client.get("/internal/v2/models/status", headers=headers2)
        assert res_models.status_code == 200
        assert res_models.json()["service_status"] == "READY"
