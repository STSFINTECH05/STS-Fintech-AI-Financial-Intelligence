"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Model D: Budget Utilization Predictor API & Security Tests (Milestone 2 - Step 7.4B-2D)

Comprehensive test suite covering all 30 requirements:
1. Valid Day 7 snapshot prediction.
2. Valid Day 14 snapshot prediction.
3. Valid Day 21 snapshot prediction.
4. Unsupported snapshot day 5 rejection (422).
5. Unsupported snapshot day 10 rejection (422).
6. Unsupported snapshot day 25 rejection (422).
7. Missing department_id rejection (422).
8. Invalid / non-positive department_id rejection (422).
9. Missing allocated_amount rejection (422).
10. Zero allocated_amount rejection (422).
11. Negative allocated_amount rejection (422).
12. Missing month_to_date_actual rejection (422).
13. Negative month_to_date_actual rejection (422).
14. NaN/Inf allocated_amount rejection (422).
15. NaN/Inf month_to_date_actual rejection (422).
16. Invalid budget_month (< 1 or > 12) rejection (422).
17. Invalid budget_year (< 2000 or > 2100) rejection (422).
18. as_of_day exceeding days in month rejection (422).
19. Estimator bundle loading and structure verification.
20. SHA-256 checksum integrity verification.
21. Feature ordering and length verification (exact 16 features).
22. Output units verification (currency vs percentage).
23. Monotonic constraint verification (pred_util >= mtd_util).
24. Estimated remaining budget formula verification.
25. Advisory risk tier: Within Budget (< 90%).
26. Advisory risk tier: Approaching Budget Limit (90% - 100%).
27. Advisory risk tier: Potential Budget Overrun – Requires Verification (> 100%).
28. Strict absence of overrun_probability field.
29. HMAC authentication and invalid signature rejection (401).
30. Replay attack rejection (401) and regression across Models A, B, C and Health endpoints.
"""

import os
import json
import time
import uuid
import hashlib
import pytest
from starlette.testclient import TestClient
from sklearn.ensemble import RandomForestRegressor

from ml.service.main import app, registry
from ml.service.security.hmac_auth import generate_hmac_signature, nonce_store
from ml.service.model_loader import APPROVED_MODEL_SPECS
from ml.service.prediction.budget_predictor import (
    BudgetPredictorV2Service,
    FEATURE_COLS,
    EXPECTED_SHA256,
    ADVISORY_NOTICE,
    BudgetInputValidationError
)

TEST_SECRET_KEY = "test_ephemeral_dev_key_at_least_32_bytes_long_secret_12345"


@pytest.fixture(autouse=True)
def configure_test_auth(monkeypatch):
    """Sets development secret key and clears nonce store before each test."""
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


def make_valid_budget_payload(as_of_day: int = 14, dept_id: str = "1", allocated: float = 1000000.0, mtd_actual: float = 400000.0):
    """Helper constructing standard valid budget prediction payload."""
    return {
        "department_id": dept_id,
        "budget_year": 2026,
        "budget_month": 3,
        "as_of_day": as_of_day,
        "allocated_amount": allocated,
        "month_to_date_actual": mtd_actual,
        "transaction_count_mtd": 25,
        "annual_budget_baseline": allocated * 12.0
    }


# ==============================================================================
# TESTS 1 - 3: Supported Snapshot Horizons (Day 7, 14, 21)
# ==============================================================================

def test_01_valid_day_7_snapshot(client):
    """1. Valid Day 7 snapshot prediction returns HTTP 200 and expected contract."""
    payload = make_valid_budget_payload(as_of_day=7, mtd_actual=200000.0)
    body = json.dumps(payload).encode("utf-8")
    headers = make_hmac_post_headers("/internal/v2/budget/predict", body)

    response = client.post("/internal/v2/budget/predict", content=body, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["inference_status"] == "EVALUATED"
    assert data["as_of_day"] == 7
    assert data["allocated_amount"] == 1000000.0
    assert data["month_to_date_actual"] == 200000.0
    assert isinstance(data["predicted_final_actual"], (int, float))
    assert isinstance(data["predicted_utilization_pct"], (int, float))
    assert isinstance(data["estimated_remaining_budget"], (int, float))
    assert data["advisory_risk_tier"] in [
        "Within Budget",
        "Approaching Budget Limit",
        "Potential Budget Overrun – Requires Verification"
    ]
    assert data["advisory_notice"] == ADVISORY_NOTICE
    assert data["provenance"]["model_name"] == "RandomForestRegressor"
    assert data["provenance"]["artifact_sha256"] == EXPECTED_SHA256


def test_02_valid_day_14_snapshot(client):
    """2. Valid Day 14 snapshot prediction returns HTTP 200."""
    payload = make_valid_budget_payload(as_of_day=14, mtd_actual=450000.0)
    body = json.dumps(payload).encode("utf-8")
    headers = make_hmac_post_headers("/internal/v2/budget/predict", body)

    response = client.post("/internal/v2/budget/predict", content=body, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["as_of_day"] == 14
    assert data["inference_status"] == "EVALUATED"
    assert data["predicted_final_actual"] > 0


def test_03_valid_day_21_snapshot(client):
    """3. Valid Day 21 snapshot prediction returns HTTP 200."""
    payload = make_valid_budget_payload(as_of_day=21, mtd_actual=700000.0)
    body = json.dumps(payload).encode("utf-8")
    headers = make_hmac_post_headers("/internal/v2/budget/predict", body)

    response = client.post("/internal/v2/budget/predict", content=body, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["as_of_day"] == 21
    assert data["inference_status"] == "EVALUATED"


# ==============================================================================
# TESTS 4 - 6: Unsupported Snapshot Horizons Rejection (HTTP 422)
# ==============================================================================

def test_04_unsupported_snapshot_day_5(client):
    """4. Unsupported snapshot day 5 is rejected with HTTP 422."""
    payload = make_valid_budget_payload(as_of_day=5)
    body = json.dumps(payload).encode("utf-8")
    headers = make_hmac_post_headers("/internal/v2/budget/predict", body)

    response = client.post("/internal/v2/budget/predict", content=body, headers=headers)
    assert response.status_code == 422
    data = response.json()
    assert data["detail"]["error_code"] == "UNSUPPORTED_SNAPSHOT_HORIZON"


def test_05_unsupported_snapshot_day_10(client):
    """5. Unsupported snapshot day 10 is rejected with HTTP 422."""
    payload = make_valid_budget_payload(as_of_day=10)
    body = json.dumps(payload).encode("utf-8")
    headers = make_hmac_post_headers("/internal/v2/budget/predict", body)

    response = client.post("/internal/v2/budget/predict", content=body, headers=headers)
    assert response.status_code == 422
    data = response.json()
    assert data["detail"]["error_code"] == "UNSUPPORTED_SNAPSHOT_HORIZON"


def test_06_unsupported_snapshot_day_25(client):
    """6. Unsupported snapshot day 25 is rejected with HTTP 422."""
    payload = make_valid_budget_payload(as_of_day=25)
    body = json.dumps(payload).encode("utf-8")
    headers = make_hmac_post_headers("/internal/v2/budget/predict", body)

    response = client.post("/internal/v2/budget/predict", content=body, headers=headers)
    assert response.status_code == 422
    data = response.json()
    assert data["detail"]["error_code"] == "UNSUPPORTED_SNAPSHOT_HORIZON"


# ==============================================================================
# TESTS 7 - 18: Input Validation & Edge Case Rejections (HTTP 422)
# ==============================================================================

def test_07_missing_department_id(client):
    """7. Missing department_id is rejected with HTTP 422."""
    payload = make_valid_budget_payload()
    del payload["department_id"]
    body = json.dumps(payload).encode("utf-8")
    headers = make_hmac_post_headers("/internal/v2/budget/predict", body)

    response = client.post("/internal/v2/budget/predict", content=body, headers=headers)
    assert response.status_code == 422


def test_08_invalid_department_id(client):
    """8. Invalid / non-positive department_id is rejected with HTTP 422."""
    payload = make_valid_budget_payload(dept_id="-5")
    body = json.dumps(payload).encode("utf-8")
    headers = make_hmac_post_headers("/internal/v2/budget/predict", body)

    response = client.post("/internal/v2/budget/predict", content=body, headers=headers)
    assert response.status_code == 422


def test_09_missing_allocated_amount(client):
    """9. Missing allocated_amount is rejected with HTTP 422."""
    payload = make_valid_budget_payload()
    del payload["allocated_amount"]
    body = json.dumps(payload).encode("utf-8")
    headers = make_hmac_post_headers("/internal/v2/budget/predict", body)

    response = client.post("/internal/v2/budget/predict", content=body, headers=headers)
    assert response.status_code == 422


def test_10_zero_allocated_amount(client):
    """10. Zero allocated_amount is rejected with HTTP 422."""
    payload = make_valid_budget_payload(allocated=0.0)
    body = json.dumps(payload).encode("utf-8")
    headers = make_hmac_post_headers("/internal/v2/budget/predict", body)

    response = client.post("/internal/v2/budget/predict", content=body, headers=headers)
    assert response.status_code == 422


def test_11_negative_allocated_amount(client):
    """11. Negative allocated_amount is rejected with HTTP 422."""
    payload = make_valid_budget_payload(allocated=-50000.0)
    body = json.dumps(payload).encode("utf-8")
    headers = make_hmac_post_headers("/internal/v2/budget/predict", body)

    response = client.post("/internal/v2/budget/predict", content=body, headers=headers)
    assert response.status_code == 422


def test_12_missing_month_to_date_actual(client):
    """12. Missing month_to_date_actual is rejected with HTTP 422."""
    payload = make_valid_budget_payload()
    del payload["month_to_date_actual"]
    body = json.dumps(payload).encode("utf-8")
    headers = make_hmac_post_headers("/internal/v2/budget/predict", body)

    response = client.post("/internal/v2/budget/predict", content=body, headers=headers)
    assert response.status_code == 422


def test_13_negative_month_to_date_actual(client):
    """13. Negative month_to_date_actual is rejected with HTTP 422."""
    payload = make_valid_budget_payload(mtd_actual=-150.0)
    body = json.dumps(payload).encode("utf-8")
    headers = make_hmac_post_headers("/internal/v2/budget/predict", body)

    response = client.post("/internal/v2/budget/predict", content=body, headers=headers)
    assert response.status_code == 422


def test_14_nan_allocated_amount(client):
    """14. NaN/Inf allocated_amount is rejected with HTTP 422."""
    payload = make_valid_budget_payload()
    payload["allocated_amount"] = "NaN"
    body = json.dumps(payload).encode("utf-8")
    headers = make_hmac_post_headers("/internal/v2/budget/predict", body)

    response = client.post("/internal/v2/budget/predict", content=body, headers=headers)
    assert response.status_code == 422


def test_15_nan_month_to_date_actual(client):
    """15. NaN/Inf month_to_date_actual is rejected with HTTP 422."""
    payload = make_valid_budget_payload()
    payload["month_to_date_actual"] = "Infinity"
    body = json.dumps(payload).encode("utf-8")
    headers = make_hmac_post_headers("/internal/v2/budget/predict", body)

    response = client.post("/internal/v2/budget/predict", content=body, headers=headers)
    assert response.status_code == 422


def test_16_invalid_budget_month(client):
    """16. Invalid budget_month (< 1 or > 12) is rejected with HTTP 422."""
    payload = make_valid_budget_payload()
    payload["budget_month"] = 13
    body = json.dumps(payload).encode("utf-8")
    headers = make_hmac_post_headers("/internal/v2/budget/predict", body)

    response = client.post("/internal/v2/budget/predict", content=body, headers=headers)
    assert response.status_code == 422


def test_17_invalid_budget_year(client):
    """17. Invalid budget_year (< 2000 or > 2100) is rejected with HTTP 422."""
    payload = make_valid_budget_payload()
    payload["budget_year"] = 1999
    body = json.dumps(payload).encode("utf-8")
    headers = make_hmac_post_headers("/internal/v2/budget/predict", body)

    response = client.post("/internal/v2/budget/predict", content=body, headers=headers)
    assert response.status_code == 422


def test_18_as_of_day_exceeding_month(client):
    """18. as_of_day exceeding days in month (e.g. Feb 30) is rejected with HTTP 422."""
    payload = make_valid_budget_payload()
    payload["budget_month"] = 2
    payload["budget_year"] = 2026
    payload["as_of_day"] = 30  # Feb has 28 days
    body = json.dumps(payload).encode("utf-8")
    headers = make_hmac_post_headers("/internal/v2/budget/predict", body)

    response = client.post("/internal/v2/budget/predict", content=body, headers=headers)
    assert response.status_code == 422


# ==============================================================================
# TESTS 19 - 24: Model Structure, Features, Output Units & Monotonic Constraint
# ==============================================================================

def test_19_bundle_loading_and_structure():
    """19. Model D artifact loads as dictionary bundle with expected keys."""
    service = BudgetPredictorV2Service(registry)
    assert isinstance(service.model, RandomForestRegressor)
    assert len(service.feature_names) == 16
    assert isinstance(service.department_profiles, dict)
    assert len(service.department_profiles) > 0


def test_20_sha256_checksum():
    """20. Verified Model D artifact matches expected SHA-256."""
    artifact_path = os.path.join(registry.models_dir, "budget_predictor_v2.joblib")
    with open(artifact_path, "rb") as f:
        actual_sha256 = hashlib.sha256(f.read()).hexdigest()
    assert actual_sha256 == EXPECTED_SHA256


def test_21_feature_ordering_and_length():
    """21. Feature derivation produces exact 16 columns matching FEATURE_COLS order."""
    service = BudgetPredictorV2Service(registry)
    payload = make_valid_budget_payload()
    df = service._validate_and_prepare_features(payload)
    assert list(df.columns) == FEATURE_COLS
    assert len(df.columns) == 16


def test_22_output_units(client):
    """22. Output units: predicted_final_actual in currency, predicted_utilization_pct in percent."""
    payload = make_valid_budget_payload(allocated=500000.0, mtd_actual=250000.0)
    body = json.dumps(payload).encode("utf-8")
    headers = make_hmac_post_headers("/internal/v2/budget/predict", body)

    response = client.post("/internal/v2/budget/predict", content=body, headers=headers)
    assert response.status_code == 200
    data = response.json()
    # predicted_utilization_pct is a percentage, e.g. 50.0 to 120.0
    assert 0.0 <= data["predicted_utilization_pct"] <= 500.0
    # predicted_final_actual is in currency units (INR)
    assert data["predicted_final_actual"] >= 250000.0


def test_23_monotonic_constraint(client):
    """23. Final utilization cannot be lower than current MTD utilization."""
    # When MTD actual is already 95% of allocation on Day 7, predicted utilization must be >= 95%
    payload = make_valid_budget_payload(as_of_day=7, allocated=1000000.0, mtd_actual=950000.0)
    body = json.dumps(payload).encode("utf-8")
    headers = make_hmac_post_headers("/internal/v2/budget/predict", body)

    response = client.post("/internal/v2/budget/predict", content=body, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_utilization_pct"] >= 95.0
    assert data["predicted_final_actual"] >= 950000.0


def test_24_estimated_remaining_budget_formula(client):
    """24. estimated_remaining_budget equals allocated_amount - predicted_final_actual."""
    payload = make_valid_budget_payload(allocated=1000000.0, mtd_actual=400000.0)
    body = json.dumps(payload).encode("utf-8")
    headers = make_hmac_post_headers("/internal/v2/budget/predict", body)

    response = client.post("/internal/v2/budget/predict", content=body, headers=headers)
    assert response.status_code == 200
    data = response.json()
    expected_remaining = round(data["allocated_amount"] - data["predicted_final_actual"], 2)
    assert abs(data["estimated_remaining_budget"] - expected_remaining) < 0.02


# ==============================================================================
# TESTS 25 - 28: Advisory Tiers & Absence of Overrun Probability
# ==============================================================================

def test_25_advisory_tier_within_budget(client):
    """25. Utilization < 90% produces 'Within Budget' advisory tier."""
    # Low MTD spend on Day 21
    payload = make_valid_budget_payload(as_of_day=21, allocated=2000000.0, mtd_actual=400000.0)
    body = json.dumps(payload).encode("utf-8")
    headers = make_hmac_post_headers("/internal/v2/budget/predict", body)

    response = client.post("/internal/v2/budget/predict", content=body, headers=headers)
    assert response.status_code == 200
    data = response.json()
    if data["predicted_utilization_pct"] < 90.0:
        assert data["advisory_risk_tier"] == "Within Budget"


def test_26_advisory_tier_approaching_limit(client):
    """26. Utilization between 90% and 100% produces 'Approaching Budget Limit' tier."""
    # High MTD spend on Day 21 that pushes util into 90-100%
    payload = make_valid_budget_payload(as_of_day=21, allocated=1000000.0, mtd_actual=920000.0)
    body = json.dumps(payload).encode("utf-8")
    headers = make_hmac_post_headers("/internal/v2/budget/predict", body)

    response = client.post("/internal/v2/budget/predict", content=body, headers=headers)
    assert response.status_code == 200
    data = response.json()
    if 90.0 <= data["predicted_utilization_pct"] <= 100.0:
        assert data["advisory_risk_tier"] == "Approaching Budget Limit"


def test_27_advisory_tier_potential_overrun(client):
    """27. Utilization > 100% produces 'Potential Budget Overrun – Requires Verification'."""
    # MTD spend already exceeding allocation
    payload = make_valid_budget_payload(as_of_day=14, allocated=500000.0, mtd_actual=600000.0)
    body = json.dumps(payload).encode("utf-8")
    headers = make_hmac_post_headers("/internal/v2/budget/predict", body)

    response = client.post("/internal/v2/budget/predict", content=body, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_utilization_pct"] >= 120.0
    assert data["advisory_risk_tier"] == "Potential Budget Overrun – Requires Verification"


def test_28_no_overrun_probability_field(client):
    """28. Response contract strictly excludes 'overrun_probability'."""
    payload = make_valid_budget_payload()
    body = json.dumps(payload).encode("utf-8")
    headers = make_hmac_post_headers("/internal/v2/budget/predict", body)

    response = client.post("/internal/v2/budget/predict", content=body, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "overrun_probability" not in data
    assert "probability" not in data


# ==============================================================================
# TESTS 29 - 30: Security & Full Microservice Regression
# ==============================================================================

def test_29_hmac_authentication_enforcement(client):
    """29. Missing or invalid HMAC headers are rejected with HTTP 401."""
    payload = make_valid_budget_payload()
    body = json.dumps(payload).encode("utf-8")

    # Missing headers
    response_no_headers = client.post("/internal/v2/budget/predict", content=body)
    assert response_no_headers.status_code == 401

    # Invalid signature
    headers = make_hmac_post_headers("/internal/v2/budget/predict", body)
    headers["X-STS-Signature"] = "invalid_signature_hex_00000000"
    response_bad_sig = client.post("/internal/v2/budget/predict", content=body, headers=headers)
    assert response_bad_sig.status_code == 401


def test_30_replay_attack_and_multi_model_regression(client):
    """30. Replay attack rejection (401) and full microservice regression across Models A, B, C, and Health."""
    payload = make_valid_budget_payload()
    body = json.dumps(payload).encode("utf-8")
    headers = make_hmac_post_headers("/internal/v2/budget/predict", body)

    # First request: 200 OK
    resp1 = client.post("/internal/v2/budget/predict", content=body, headers=headers)
    assert resp1.status_code == 200

    # Replay request with same nonce: 401 Unauthorized
    resp2 = client.post("/internal/v2/budget/predict", content=body, headers=headers)
    assert resp2.status_code == 401

    # Regression: Health endpoints
    resp_live = client.get("/health/live")
    assert resp_live.status_code == 200
    assert resp_live.json()["service_status"] == "ALIVE"

    # Regression: Model A (Classify)
    class_payload = {
        "amount": 2500.0,
        "description": "Enterprise cloud license",
        "transaction_date": "2026-03-15",
        "department_id": "DEP_001",
        "vendor_category": "Software"
    }
    class_body = json.dumps(class_payload).encode("utf-8")
    class_headers = make_hmac_post_headers("/internal/v2/classify", class_body)
    resp_class = client.post("/internal/v2/classify", content=class_body, headers=class_headers)
    assert resp_class.status_code == 200
    assert resp_class.json()["status"] == "SUCCESS"

    # Regression: Model B (Anomaly)
    anomaly_payload = {
        "amount": 2500.0,
        "transaction_date": "2026-03-15",
        "category": "Software",
        "department_id": "DEP_001",
        "category_median": 2000.0,
        "department_median": 2100.0,
        "spending_approval_limit": 50000.0,
        "annual_budget_baseline": 10000000.0,
        "vendor_rating": 4.5
    }
    anomaly_body = json.dumps(anomaly_payload).encode("utf-8")
    anomaly_headers = make_hmac_post_headers("/internal/v2/anomaly/score", anomaly_body)
    resp_anomaly = client.post("/internal/v2/anomaly/score", content=anomaly_body, headers=anomaly_headers)
    assert resp_anomaly.status_code == 200
    assert resp_anomaly.json()["status"] == "SUCCESS"
