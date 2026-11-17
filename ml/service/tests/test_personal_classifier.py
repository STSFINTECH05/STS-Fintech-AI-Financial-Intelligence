"""
STS FINTECH - PERSONAL FINANCE CLASSIFIER TEST SUITE
Milestone 2 - Step 7.4C-3D (Experimental Microservice)

Contains 38 comprehensive tests verifying:
- Model loading, class inventory, and cryptographic integrity.
- Safe deserialization using PersonalFinanceNumpyUnpickler (and rejection by standard pickle).
- Income classification (Consulting, Retainer, Investment & Yield).
- Expense classification across verified 8 categories.
- Direction-aware guardrails:
  - Salary received vs. Payroll outgoing.
  - Loan disbursement vs. Loan EMI debit.
  - Refund credit vs. Refund expense fee.
- Boundary conditions:
  - Ambiguous / multi-category phrases.
  - Generic / insufficient UPI strings.
  - Out-of-distribution (OOD) amounts.
  - Invalid calendar dates and formats.
  - Text-only invariant (amount does not shift prediction).
  - Manual confirmation invariant (requires_manual_confirmation: true).
- Detailed context-aware PII sanitization:
  - Phone numbers, UPI VPAs, account numbers.
  - Full card numbers and card-ending references ('card ending 1234', 'card ending in 1234', 'card XXXX1234').
  - Preserves surrounding merchant context and non-card numbers (invoices, dates).
- Model-unavailable handling and error resilience.
- Cryptographic HMAC-SHA256 verification:
  - Valid canonical signatures accepted.
  - Missing security headers rejected (401).
  - Invalid signature rejected (401).
  - Nonce replay rejected (401).
  - Path mismatch rejected (401).
"""

import os
import sys
import time
import json
import uuid
import pickle
import hashlib
import pytest
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, BASE_DIR)

from ml.training.train_personal_finance_v1 import pii_sanitizer_transform
from ml.service.prediction.personal_classifier import (
    PersonalFinanceClassifierService,
    PersonalFinanceNumpyUnpickler,
    safe_load_personal_bundle,
    sanitize_input_text,
    BUNDLE_PATH,
    EXPECTED_BUNDLE_SHA256,
    RANKING_THRESHOLD,
    TOP_TWO_MARGIN_THRESHOLD
)
from ml.service.personal_schemas import PersonalClassifyRequest, PersonalClassifyResponse
from ml.service.security.hmac_auth import generate_hmac_signature, nonce_store
from ml.service.main import create_app
from ml.service.routes.personal_routes import router as personal_router

# Test secret key (>= 32 chars)
TEST_KEY = "test_secret_key_0123456789_abcdefghij"
os.environ["STS_INTERNAL_SERVICE_KEY"] = TEST_KEY


@pytest.fixture(scope="module")
def service():
    """Initializes and returns the singleton classifier service."""
    # Reset singleton to ensure fresh load with new unpickler
    PersonalFinanceClassifierService._instance = None
    return PersonalFinanceClassifierService.get_instance()


@pytest.fixture(scope="module")
def client():
    """Creates a test client with personal routes included."""
    app = create_app()
    app.include_router(personal_router)
    return TestClient(app)


def build_auth_headers(method: str, path: str, body_bytes: bytes, key: str = TEST_KEY, timestamp_str: str = None, nonce: str = None):
    """Helper to generate valid HMAC headers for testing."""
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


# =============================================================================
# 1. Model Loading, Deserialization & Class Inventory Tests (5 tests)
# =============================================================================
def test_01_model_loading_and_integrity(service):
    """Verifies artifact SHA-256 and successful load."""
    assert service.is_loaded is True
    assert service.load_error is None


def test_02_standard_pickle_rejection():
    """Verifies that standard pickle fails on the joblib-specific artifact."""
    with open(BUNDLE_PATH, "rb") as f:
        with pytest.raises(Exception):
            pickle.load(f)


def test_03_custom_unpickler_successful_load():
    """Verifies that PersonalFinanceNumpyUnpickler successfully deserializes the bundle."""
    bundle = safe_load_personal_bundle(BUNDLE_PATH)
    assert "income_pipeline" in bundle
    assert "expense_pipeline" in bundle
    assert bundle["dataset_sha256"] == "11cf2d14da863707a7b69406582ef51d18d5d46c73e153a7b6c0f9ea5105e243"


def test_04_income_class_inventory(service):
    """Verifies that income classes exactly match the approved 3 categories."""
    expected = ["Consulting", "Investment & Yield", "Retainer"]
    assert sorted(service.income_classes) == sorted(expected)


def test_05_expense_class_inventory(service):
    """Verifies that expense classes exactly match the 8 ExpenseTrack categories."""
    expected = [
        "General", "Infrastructure & Cloud", "Legal & Compliance", "Office Rent",
        "Operations & Travel", "Payroll & Salaries", "Software Subscription", "Utilities & Bills"
    ]
    assert sorted(service.expense_classes) == sorted(expected)


# =============================================================================
# 2. Supervised Classification Tests (6 tests)
# =============================================================================
def test_06_income_consulting_classification(service):
    res = service.classify_transaction(
        transaction_direction="income",
        description="Fee for smart contract security consultation",
        amount_inr=50000.0,
        transaction_date="2026-11-17"
    )
    assert res["status"] == "SUCCESS"
    assert res["category"] == "Consulting"
    assert res["ranking_score"] >= RANKING_THRESHOLD


def test_07_income_retainer_classification(service):
    res = service.classify_transaction(
        transaction_direction="income",
        description="Monthly advisory retainer for Q4 technical guidance",
        amount_inr=80000.0,
        transaction_date="2026-11-17"
    )
    assert res["status"] == "SUCCESS"
    assert res["category"] == "Retainer"
    assert res["ranking_score"] >= RANKING_THRESHOLD


def test_08_income_investment_classification(service):
    res = service.classify_transaction(
        transaction_direction="income",
        description="Quarterly dividend payout from TCS equity holdings",
        amount_inr=15000.0,
        transaction_date="2026-11-17"
    )
    assert res["status"] == "SUCCESS"
    assert res["category"] == "Investment & Yield"
    assert res["ranking_score"] >= RANKING_THRESHOLD


def test_09_expense_infrastructure_classification(service):
    res = service.classify_transaction(
        transaction_direction="expense",
        description="Hetzner cloud server hosting monthly billing",
        amount_inr=3200.0,
        transaction_date="2026-11-17"
    )
    assert res["status"] == "SUCCESS"
    assert res["category"] == "Infrastructure & Cloud"
    assert res["ranking_score"] >= RANKING_THRESHOLD


def test_10_expense_office_rent_classification(service):
    res = service.classify_transaction(
        transaction_direction="expense",
        description="WeWork dedicated desk rent Koramangala Bangalore",
        amount_inr=18000.0,
        transaction_date="2026-11-17"
    )
    assert res["status"] == "SUCCESS"
    assert res["category"] == "Office Rent"
    assert res["ranking_score"] >= RANKING_THRESHOLD


def test_11_expense_operations_travel_classification(service):
    res = service.classify_transaction(
        transaction_direction="expense",
        description="IndiGo flight ticket booking BLR to DEL",
        amount_inr=6500.0,
        transaction_date="2026-11-17"
    )
    assert res["status"] == "SUCCESS"
    assert res["category"] == "Operations & Travel"
    assert res["ranking_score"] >= RANKING_THRESHOLD


# =============================================================================
# 3. Direction-Aware Guardrail Tests (8 tests)
# =============================================================================
def test_12_guardrail_incoming_salary_abstains(service):
    """Salary received on income must abstain as UNSUPPORTED_INCOME_SALARY."""
    res = service.classify_transaction(
        transaction_direction="income",
        description="Monthly salary credited by Infosys Limited",
        amount_inr=95000.0,
        transaction_date="2026-11-17"
    )
    assert res["status"] == "ABSTAIN"
    assert res["category"] is None
    assert res["abstention_reason"] == "UNSUPPORTED_INCOME_SALARY"


def test_13_guardrail_outgoing_payroll_permitted(service):
    """Outgoing payroll expense must be classified under Payroll & Salaries."""
    res = service.classify_transaction(
        transaction_direction="expense",
        description="Monthly payroll salary disbursement to engineering team",
        amount_inr=250000.0,
        transaction_date="2026-11-17"
    )
    assert res["status"] == "SUCCESS"
    assert res["category"] == "Payroll & Salaries"


def test_14_guardrail_incoming_loan_abstains(service):
    """Personal loan disbursement received on income must abstain."""
    res = service.classify_transaction(
        transaction_direction="income",
        description="Personal loan disbursement from HDFC Bank",
        amount_inr=200000.0,
        transaction_date="2026-11-17"
    )
    assert res["status"] == "ABSTAIN"
    assert res["category"] is None
    assert res["abstention_reason"] == "UNSUPPORTED_INCOME_LOAN"


def test_15_guardrail_outgoing_loan_emi_permitted(service):
    """Outgoing loan EMI repayment reaches model without guardrail rejection."""
    res = service.classify_transaction(
        transaction_direction="expense",
        description="Home loan EMI auto debit payment HDFC",
        amount_inr=35000.0,
        transaction_date="2026-11-17"
    )
    assert res["abstention_reason"] != "UNSUPPORTED_INCOME_LOAN"
    assert res["status"] == "ABSTAIN"
    assert res["abstention_reason"] == "LOW_SCORE"


def test_16_guardrail_incoming_refund_abstains(service):
    """Merchant refund on income must abstain."""
    res = service.classify_transaction(
        transaction_direction="income",
        description="Amazon merchant refund credited back to account",
        amount_inr=1499.0,
        transaction_date="2026-11-17"
    )
    assert res["status"] == "ABSTAIN"
    assert res["category"] is None
    assert res["abstention_reason"] == "UNSUPPORTED_INCOME_REFUND"


def test_17_guardrail_outgoing_refund_charge_permitted(service):
    """Refund processing fee expense reaches model without guardrail rejection."""
    res = service.classify_transaction(
        transaction_direction="expense",
        description="Gateway refund processing fee charge",
        amount_inr=50.0,
        transaction_date="2026-11-17"
    )
    assert res["abstention_reason"] != "UNSUPPORTED_INCOME_REFUND"
    assert res["status"] == "ABSTAIN"
    assert res["abstention_reason"] == "LOW_SCORE"


def test_18_guardrail_legitimate_consulting_credited_by_company(service):
    """Valid consulting credited by a private company must NOT be rejected by salary rule."""
    res = service.classify_transaction(
        transaction_direction="income",
        description="Consulting charges for architecture audit from Acme Tech",
        amount_inr=75000.0,
        transaction_date="2026-11-17"
    )
    assert res["status"] == "SUCCESS"
    assert res["category"] == "Consulting"


def test_19_guardrail_legitimate_retainer_credited_by_client(service):
    """Valid retainer credited by a client must reach Retainer."""
    res = service.classify_transaction(
        transaction_direction="income",
        description="Monthly UX retainer payment received from client",
        amount_inr=60000.0,
        transaction_date="2026-11-17"
    )
    assert res["status"] == "SUCCESS"
    assert res["category"] == "Retainer"


# =============================================================================
# 4. Boundary & Ambiguity Tests (6 tests)
# =============================================================================
def test_20_insufficient_information_empty_narrative(service):
    res = service.classify_transaction(
        transaction_direction="expense",
        description="   ",
        amount_inr=500.0,
        transaction_date="2026-11-17"
    )
    assert res["status"] == "ABSTAIN"
    assert res["abstention_reason"] == "INSUFFICIENT_INFORMATION"


def test_21_insufficient_information_generic_upi(service):
    res = service.classify_transaction(
        transaction_direction="expense",
        description="UPI",
        amount_inr=250.0,
        transaction_date="2026-11-17"
    )
    assert res["status"] == "ABSTAIN"
    assert res["abstention_reason"] == "INSUFFICIENT_INFORMATION"


def test_22_insufficient_information_generic_transfer(service):
    res = service.classify_transaction(
        transaction_direction="income",
        description="IMPS transfer",
        amount_inr=5000.0,
        transaction_date="2026-11-17"
    )
    assert res["status"] == "ABSTAIN"
    assert res["abstention_reason"] == "INSUFFICIENT_INFORMATION"


def test_23_multi_category_ambiguity_explicit_conjunction(service):
    """Narrative combining cloud hosting and office rent must abstain as AMBIGUOUS."""
    res = service.classify_transaction(
        transaction_direction="expense",
        description="AWS cloud hosting plus purchase of office desk rent",
        amount_inr=22000.0,
        transaction_date="2026-11-17"
    )
    assert res["status"] == "ABSTAIN"
    assert res["abstention_reason"] == "AMBIGUOUS"


def test_24_ood_amount_below_minimum(service):
    """Amount < 1.0 INR must abstain as OUT_OF_DISTRIBUTION_AMOUNT."""
    res = service.classify_transaction(
        transaction_direction="expense",
        description="AWS cloud server micro charge",
        amount_inr=0.50,
        transaction_date="2026-11-17"
    )
    assert res["status"] == "ABSTAIN"
    assert res["abstention_reason"] == "OUT_OF_DISTRIBUTION_AMOUNT"


def test_25_ood_amount_above_maximum(service):
    """Amount > 1 Crore INR must abstain as OUT_OF_DISTRIBUTION_AMOUNT."""
    res = service.classify_transaction(
        transaction_direction="expense",
        description="Office space lease payment",
        amount_inr=15000000.0,
        transaction_date="2026-11-17"
    )
    assert res["status"] == "ABSTAIN"
    assert res["abstention_reason"] == "OUT_OF_DISTRIBUTION_AMOUNT"


# =============================================================================
# 5. Pipeline Invariants, Card Endings & PII Redaction Tests (6 tests)
# =============================================================================
def test_26_text_only_invariant(service):
    """Varying amount within valid bounds must yield identical category and score."""
    res1 = service.classify_transaction("expense", "WeWork desk rent", 5000.0, "2026-11-17")
    res2 = service.classify_transaction("expense", "WeWork desk rent", 500000.0, "2026-11-17")
    assert res1["category"] == res2["category"]
    assert res1["ranking_score"] == res2["ranking_score"]


def test_27_manual_confirmation_invariant(service):
    """requires_manual_confirmation must always be True."""
    res_ok = service.classify_transaction("income", "Fee for cloud architecture consultation", 10000.0, "2026-11-17")
    res_abs = service.classify_transaction("income", "salary credit", 10000.0, "2026-11-17")
    assert res_ok["requires_manual_confirmation"] is True
    assert res_abs["requires_manual_confirmation"] is True


def test_28_pii_sanitization_in_pipeline(service):
    """PII in narrative (phone, UPI) should not crash pipeline and produce clean inference."""
    res = service.classify_transaction(
        transaction_direction="expense",
        description="Uber cab trip call +91 9876543210 UPI trip.pay@icici",
        amount_inr=450.0,
        transaction_date="2026-11-17"
    )
    assert res["status"] == "SUCCESS"
    assert res["category"] == "Operations & Travel"


def test_29_card_ending_and_sensitive_redaction():
    """Verifies that card ending phrases and digits are completely redacted."""
    raw = "Payment to cab call +91 9876543210 upi test@okaxis card ending 1234 acc 123456789012"
    cleaned = sanitize_input_text(raw)
    assert "1234" not in cleaned
    assert "9876543210" not in cleaned
    assert "test@okaxis" not in cleaned
    assert "123456789012" not in cleaned
    assert "[CARD]" in cleaned
    assert "[PHONE]" in cleaned
    assert "[UPI]" in cleaned
    assert "[ACCT]" in cleaned


def test_30_card_ending_variations_and_ordinary_numbers():
    """Verifies card suffix variants while preserving invoice numbers and dates."""
    c1 = sanitize_input_text("Payment using card ending in 5678 at Starbucks")
    assert "5678" not in c1
    assert "Starbucks" in c1
    assert "[CARD]" in c1

    c2 = sanitize_input_text("Txn card XXXX1234 settled for AWS hosting")
    assert "1234" not in c2
    assert "AWS hosting" in c2
    assert "[CARD]" in c2

    # Preserves ordinary non-card numbers
    normal = sanitize_input_text("Invoice 2024 for cloud hosting dated 2026-11-17")
    assert "Invoice 2024" in normal
    assert "2026-11-17" in normal
    assert "[CARD]" not in normal


def test_31_model_unavailable_resilience():
    """Service initialized with invalid path returns NOT_EVALUATED gracefully."""
    fake_service = PersonalFinanceClassifierService(bundle_path="/tmp/non_existent.joblib")
    res = fake_service.classify_transaction("expense", "AWS invoice", 1000.0, "2026-11-17")
    assert res["status"] == "NOT_EVALUATED"
    assert res["abstention_reason"] == "MODEL_UNAVAILABLE"


# =============================================================================
# 6. HTTP API Endpoint & HMAC Security Tests (7 tests)
# =============================================================================
ENDPOINT_PATH = "/internal/v1/personal/classify"


def test_32_endpoint_valid_hmac_request_success(client):
    payload = {
        "transaction_direction": "expense",
        "description": "AWS cloud hosting monthly invoice",
        "amount_inr": 4500.0,
        "transaction_date": "2026-11-17"
    }
    body_bytes = json.dumps(payload).encode("utf-8")
    headers = build_auth_headers("POST", ENDPOINT_PATH, body_bytes)

    response = client.post(ENDPOINT_PATH, content=body_bytes, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["category"] == "Infrastructure & Cloud"
    assert data["score_semantics"] == "INTERNAL_RANKING_ONLY"
    assert data["requires_manual_confirmation"] is True


def test_33_endpoint_missing_hmac_headers_rejected(client):
    payload = {
        "transaction_direction": "expense",
        "description": "AWS cloud hosting monthly invoice",
        "amount_inr": 4500.0,
        "transaction_date": "2026-11-17"
    }
    response = client.post(ENDPOINT_PATH, json=payload)
    assert response.status_code == 401
    assert response.json()["detail"]["error_code"] == "MISSING_SECURITY_HEADERS"


def test_34_endpoint_invalid_signature_rejected(client):
    payload = {
        "transaction_direction": "expense",
        "description": "AWS cloud hosting monthly invoice",
        "amount_inr": 4500.0,
        "transaction_date": "2026-11-17"
    }
    body_bytes = json.dumps(payload).encode("utf-8")
    headers = build_auth_headers("POST", ENDPOINT_PATH, body_bytes)
    headers["X-STS-Signature"] = "deadbeef" * 8  # Corrupt signature

    response = client.post(ENDPOINT_PATH, content=body_bytes, headers=headers)
    assert response.status_code == 401
    assert response.json()["detail"]["error_code"] == "INVALID_SIGNATURE"


def test_35_endpoint_replay_attack_rejected(client):
    payload = {
        "transaction_direction": "expense",
        "description": "AWS cloud hosting monthly invoice",
        "amount_inr": 4500.0,
        "transaction_date": "2026-11-17"
    }
    body_bytes = json.dumps(payload).encode("utf-8")
    nonce = str(uuid.uuid4())
    headers = build_auth_headers("POST", ENDPOINT_PATH, body_bytes, nonce=nonce)

    # First request succeeds
    r1 = client.post(ENDPOINT_PATH, content=body_bytes, headers=headers)
    assert r1.status_code == 200

    # Second request with same nonce is rejected as replay
    r2 = client.post(ENDPOINT_PATH, content=body_bytes, headers=headers)
    assert r2.status_code == 401
    assert r2.json()["detail"]["error_code"] == "REPLAY_DETECTED"


def test_36_endpoint_path_mismatch_rejected(client):
    """Signing for a different path must fail HMAC validation."""
    payload = {
        "transaction_direction": "expense",
        "description": "AWS cloud hosting monthly invoice",
        "amount_inr": 4500.0,
        "transaction_date": "2026-11-17"
    }
    body_bytes = json.dumps(payload).encode("utf-8")
    headers = build_auth_headers("POST", "/internal/v2/classify", body_bytes)

    response = client.post(ENDPOINT_PATH, content=body_bytes, headers=headers)
    assert response.status_code == 401
    assert response.json()["detail"]["error_code"] == "INVALID_SIGNATURE"


def test_37_endpoint_invalid_date_validation_error(client):
    """Invalid calendar date format must return 422 Unprocessable Entity."""
    payload = {
        "transaction_direction": "expense",
        "description": "AWS cloud hosting monthly invoice",
        "amount_inr": 4500.0,
        "transaction_date": "2026-02-31"  # Invalid calendar day
    }
    body_bytes = json.dumps(payload).encode("utf-8")
    headers = build_auth_headers("POST", ENDPOINT_PATH, body_bytes)

    response = client.post(ENDPOINT_PATH, content=body_bytes, headers=headers)
    assert response.status_code == 422


def test_38_endpoint_unsupported_direction_validation_error(client):
    """Invalid transaction_direction must return 422 Unprocessable Entity."""
    payload = {
        "transaction_direction": "transfer",  # Invalid direction
        "description": "AWS cloud hosting monthly invoice",
        "amount_inr": 4500.0,
        "transaction_date": "2026-11-17"
    }
    body_bytes = json.dumps(payload).encode("utf-8")
    headers = build_auth_headers("POST", ENDPOINT_PATH, body_bytes)

    response = client.post(ENDPOINT_PATH, content=body_bytes, headers=headers)
    assert response.status_code == 422
